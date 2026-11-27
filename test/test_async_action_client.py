# Copyright 2026 Kiwibot
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from collections.abc import Generator, Iterator
import contextlib
import os
import threading
import time
from typing import Any, List

from action_msgs.msg import GoalStatus
from action_msgs.srv import CancelGoal
import pytest

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import rclpy_async
from rclpy_async import _rcl_events, AsyncActionClient, AsyncClientGoalHandle, AsyncNode

from test_msgs.action import Fibonacci
from unique_identifier_msgs.msg import UUID

KINDS = ('goal_response', 'cancel_response', 'result_response', 'feedback', 'status')


@pytest.fixture(autouse=True)
def rclpy_context() -> Generator[None, None, None]:
    """Initialize and shut down rclpy for each test."""
    with rclpy_async.init():
        yield


@contextlib.contextmanager
def fibonacci_server(
    action_name: str,
    *,
    accept: bool = True,
    step_s: float = 0.0,
    goal_gate: threading.Event | None = None,
) -> Iterator[List[Any]]:
    """
    Serve Fibonacci with a plain rclpy ActionServer spun on a background thread.

    Yields the list of goal requests the server accepted, in arrival order.
    """
    accepted: List[Any] = []

    def on_goal(request: Fibonacci.Goal) -> GoalResponse:
        if goal_gate is not None:
            goal_gate.wait(timeout=5)
        if accept:
            accepted.append(request)
            return GoalResponse.ACCEPT
        return GoalResponse.REJECT

    def execute(goal_handle: Any) -> Fibonacci.Result:
        sequence = [0, 1]
        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return Fibonacci.Result(sequence=sequence)
            sequence.append(sequence[i] + sequence[i - 1])
            goal_handle.publish_feedback(Fibonacci.Feedback(sequence=sequence))
            time.sleep(step_s)
        goal_handle.succeed()
        return Fibonacci.Result(sequence=sequence)

    node = rclpy.create_node(f'{action_name.strip("/")}_server')
    server = ActionServer(
        node, Fibonacci, action_name, execute,
        goal_callback=on_goal,
        cancel_callback=lambda _: CancelResponse.ACCEPT,
        callback_group=ReentrantCallbackGroup())
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        yield accepted
    finally:
        if goal_gate is not None:
            goal_gate.set()
            # Let a goal released here finish before its executor goes away.
            time.sleep(0.2)
        executor.shutdown()
        thread.join(timeout=5)
        server.destroy()
        node.destroy_node()


@pytest.mark.asyncio
async def test_goal_feedback_and_result() -> None:
    """A goal is accepted, streams feedback to a sync callback and returns its result."""
    feedback: List[List[int]] = []

    with fibonacci_server('/test_fib', step_s=0.02):
        async with AsyncNode('test_fib_client') as node:
            client = node.create_action_client(Fibonacci, '/test_fib')
            async with asyncio.timeout(10):
                await client.wait_for_server()
                goal_handle = await client.send_goal(
                    Fibonacci.Goal(order=6),
                    feedback_callback=lambda msg: feedback.append(list(msg.feedback.sequence)))
                assert isinstance(goal_handle, AsyncClientGoalHandle)
                assert goal_handle.accepted
                response = await goal_handle.get_result()

    assert response.status == GoalStatus.STATUS_SUCCEEDED
    assert list(response.result.sequence) == [0, 1, 1, 2, 3, 5, 8]
    assert goal_handle.status == GoalStatus.STATUS_SUCCEEDED
    assert feedback, 'no feedback reached the callback'
    assert feedback[-1] == list(response.result.sequence)[:len(feedback[-1])]


@pytest.mark.asyncio
async def test_async_feedback_callback() -> None:
    """An async def feedback callback is awaited."""
    feedback_seen = asyncio.Event()

    async def on_feedback(msg: Any) -> None:
        await asyncio.sleep(0)
        feedback_seen.set()

    with fibonacci_server('/test_fib_async_fb', step_s=0.02):
        async with AsyncNode('test_fib_async_fb_client') as node:
            client = node.create_action_client(Fibonacci, '/test_fib_async_fb')
            async with asyncio.timeout(10):
                await client.wait_for_server()
                goal_handle = await client.send_goal(
                    Fibonacci.Goal(order=5), feedback_callback=on_feedback)
                await goal_handle.get_result()
    assert feedback_seen.is_set()


@pytest.mark.asyncio
async def test_rejected_goal() -> None:
    """A rejected goal returns a handle with accepted False and keeps no feedback callback."""
    with fibonacci_server('/test_fib_reject', accept=False):
        async with AsyncNode('test_fib_reject_client') as node:
            client = node.create_action_client(Fibonacci, '/test_fib_reject')
            async with asyncio.timeout(10):
                await client.wait_for_server()
                goal_handle = await client.send_goal(
                    Fibonacci.Goal(order=3), feedback_callback=lambda msg: None)
    assert not goal_handle.accepted
    assert not client._feedback_callbacks


@pytest.mark.asyncio
async def test_cancel_goal() -> None:
    """Canceling a running goal is acknowledged and its result reports CANCELED."""
    with fibonacci_server('/test_fib_cancel', step_s=0.05):
        async with AsyncNode('test_fib_cancel_client') as node:
            client = node.create_action_client(Fibonacci, '/test_fib_cancel')
            async with asyncio.timeout(10):
                await client.wait_for_server()
                goal_handle = await client.send_goal(Fibonacci.Goal(order=100))
                cancel_response = await goal_handle.cancel_goal()
                response = await goal_handle.get_result()

    assert cancel_response.return_code == CancelGoal.Response.ERROR_NONE
    assert len(cancel_response.goals_canceling) == 1
    assert bytes(cancel_response.goals_canceling[0].goal_id.uuid) == bytes(
        goal_handle.goal_id.uuid)
    assert response.status == GoalStatus.STATUS_CANCELED


@pytest.mark.asyncio
async def test_concurrent_goals_are_demuxed() -> None:
    """Concurrent goals on one client each get their own result."""
    orders = (3, 5, 8)
    expected = {3: [0, 1, 1, 2], 5: [0, 1, 1, 2, 3, 5], 8: [0, 1, 1, 2, 3, 5, 8, 13, 21]}

    async def run(client: AsyncActionClient, order: int) -> List[int]:
        goal_handle = await client.send_goal(Fibonacci.Goal(order=order))
        return list((await goal_handle.get_result()).result.sequence)

    with fibonacci_server('/test_fib_concurrent', step_s=0.01):
        async with AsyncNode('test_fib_concurrent_client') as node:
            client = node.create_action_client(Fibonacci, '/test_fib_concurrent')
            async with asyncio.timeout(10):
                await client.wait_for_server()
                results = await asyncio.gather(*(run(client, order) for order in orders))

    assert dict(zip(orders, results)) == expected


@pytest.mark.asyncio
async def test_goal_uuid_is_kept() -> None:
    """A caller supplied goal id is the one the server sees."""
    goal_id = UUID(uuid=list(range(16)))
    with fibonacci_server('/test_fib_uuid'):
        async with AsyncNode('test_fib_uuid_client') as node:
            client = node.create_action_client(Fibonacci, '/test_fib_uuid')
            async with asyncio.timeout(10):
                await client.wait_for_server()
                goal_handle = await client.send_goal(Fibonacci.Goal(order=2), goal_uuid=goal_id)
                await goal_handle.get_result()
    assert bytes(goal_handle.goal_id.uuid) == bytes(range(16))


@pytest.mark.asyncio
async def test_wait_for_server_timeout() -> None:
    """Timeout is raised when no action server exists."""
    async with AsyncNode('test_fib_no_server') as node:
        client = node.create_action_client(Fibonacci, '/test_fib_nonexistent')
        assert not client.server_is_ready()
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.5):
                await client.wait_for_server()


@pytest.mark.asyncio
async def test_wrong_goal_type_raises() -> None:
    """A goal that is not the action's Goal type is refused before anything is sent."""
    async with AsyncNode('test_fib_bad_goal') as node:
        client = node.create_action_client(Fibonacci, '/test_fib_bad_goal')
        with pytest.raises(TypeError):
            await client.send_goal(Fibonacci.Result())


@pytest.mark.asyncio
async def test_send_goal_on_destroyed_client() -> None:
    """Sending a goal on a destroyed client raises RuntimeError."""
    async with AsyncNode('test_fib_destroyed') as node:
        client = node.create_action_client(Fibonacci, '/test_fib_destroyed')
        client.destroy()
        assert client.destroyed
        with pytest.raises(RuntimeError):
            await client.send_goal(Fibonacci.Goal(order=1))


@pytest.mark.asyncio
async def test_destroy_cancels_in_flight_goal() -> None:
    """destroy() while a goal awaits the server's answer cancels the await."""
    gate = threading.Event()
    with fibonacci_server('/test_fib_destroy_inflight', goal_gate=gate):
        async with AsyncNode('test_fib_destroy_inflight_client') as node:
            client = node.create_action_client(Fibonacci, '/test_fib_destroy_inflight')
            async with asyncio.timeout(10):
                await client.wait_for_server()
            task = asyncio.ensure_future(client.send_goal(Fibonacci.Goal(order=2)))
            await asyncio.sleep(0.1)
            client.destroy()
            with pytest.raises(asyncio.CancelledError):
                await task


@pytest.mark.asyncio
async def test_timed_out_goal_then_next_goal_succeeds() -> None:
    """A send_goal abandoned by a timeout does not break later goals on the same client."""
    gate = threading.Event()
    with fibonacci_server('/test_fib_timeout', goal_gate=gate):
        async with AsyncNode('test_fib_timeout_client') as node:
            client = node.create_action_client(Fibonacci, '/test_fib_timeout')
            async with asyncio.timeout(10):
                await client.wait_for_server()
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(0.2):
                    await client.send_goal(
                        Fibonacci.Goal(order=2), feedback_callback=lambda msg: None)
            assert not client._pending_goals
            assert not client._feedback_callbacks
            gate.set()
            async with asyncio.timeout(10):
                goal_handle = await client.send_goal(Fibonacci.Goal(order=3))
                response = await goal_handle.get_result()
    assert list(response.result.sequence) == [0, 1, 1, 2]


@pytest.mark.asyncio
async def test_destroy_node_destroys_action_client() -> None:
    """The node owns its action clients and destroys them with itself."""
    async with AsyncNode('test_fib_owner') as node:
        client = node.create_action_client(Fibonacci, '/test_fib_owner')
        assert client in node._entities
    assert client.destroyed


@pytest.mark.asyncio
async def test_action_eventfds_are_registered_then_released() -> None:
    """All five middleware callbacks get an eventfd, and destroy releases every one."""
    async with AsyncNode('test_fib_eventfd') as node:
        client = node.create_action_client(Fibonacci, '/test_fib_eventfd')
        await asyncio.sleep(0)  # let the bridge task register its callbacks

        if not _rcl_events.uses_ctypes(client.handle):
            pytest.skip('this build has no handle pointer, so no eventfd exists')

        keys = [(client.handle.pointer, kind) for kind in KINDS]
        wakeups = [_rcl_events._alive[key] for key in keys]
        for wakeup in wakeups:
            os.fstat(wakeup.fd)  # raises if the descriptor is not open

        client.destroy()
        await asyncio.sleep(0.05)
        for key, wakeup in zip(keys, wakeups):
            assert key not in _rcl_events._alive, f'the {key[1]} eventfd outlived the client'
            with pytest.raises(OSError):
                os.fstat(wakeup.fd)


@pytest.mark.asyncio
async def test_action_client_churn_does_not_leak_callbacks() -> None:
    """Creating and destroying many action clients leaves no registered callbacks behind."""
    baseline = len(_rcl_events._alive)
    async with AsyncNode('test_fib_churn') as node:
        for i in range(20):
            client = node.create_action_client(Fibonacci, f'/test_fib_churn_{i}')
            await asyncio.sleep(0)
            client.destroy()
        await asyncio.sleep(0.1)
    leaked = len(_rcl_events._alive) - baseline
    assert leaked == 0, f'{leaked} middleware callbacks were never cleared'
