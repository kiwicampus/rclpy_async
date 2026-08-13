# Copyright 2026 Open Source Robotics Foundation, Inc.
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

"""
Does the asyncio actually work, or does it only look like it?

The upstream suite checks that the API behaves. These tests check the properties that a fake
implementation would get wrong: real elapsed time, a loop that keeps running during an await,
callbacks on the loop thread, genuine overlap under concurrent dispatch, and no deadlock when a
ROS call is awaited from inside another ROS callback.
"""

import asyncio
from collections.abc import Generator
import threading
import time

import pytest

import rclpy_async
from rclpy_async import AsyncNode

from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy

from std_msgs.msg import String
from test_msgs.srv import BasicTypes as BasicTypesSrv

TEST_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)


@pytest.fixture(autouse=True)
def rclpy_context() -> Generator[None, None, None]:
    with rclpy_async.init():
        yield


class Ticker:
    """A background task that counts loop turns, to prove the loop is not blocked."""

    def __init__(self, period: float = 0.01) -> None:
        self.count = 0
        self._period = period
        self._task: 'asyncio.Task[None]' = None

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self._period)
            self.count += 1

    def __enter__(self) -> 'Ticker':
        self._task = asyncio.get_running_loop().create_task(self._run())
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._task.cancel()


@pytest.mark.asyncio
async def test_asyncio_sleep_in_timer_really_sleeps() -> None:
    """A timer callback that awaits asyncio.sleep takes at least that long on the wall clock."""
    SLEEP = 0.3
    elapsed = []
    done = asyncio.Event()

    async def callback() -> None:
        if done.is_set():
            return
        start = time.monotonic()
        await asyncio.sleep(SLEEP)
        elapsed.append(time.monotonic() - start)
        done.set()

    async with AsyncNode('test_sleep_in_timer') as node:
        node.create_timer(0.05, callback)
        async with asyncio.timeout(5):
            await done.wait()

    assert len(elapsed) == 1
    assert elapsed[0] >= SLEEP, f'slept {elapsed[0]:.3f}s, expected at least {SLEEP}s'
    # A blocking time.sleep would also pass the line above, so bound it from the other side too.
    assert elapsed[0] < SLEEP + 1.0


@pytest.mark.asyncio
async def test_event_loop_keeps_running_during_a_timer_await() -> None:
    """While a timer callback awaits, the loop still runs everything else."""
    SLEEP = 0.4
    ticks_during_sleep = []
    done = asyncio.Event()

    with Ticker(0.01) as ticker:

        async def callback() -> None:
            if done.is_set():
                return
            before = ticker.count
            await asyncio.sleep(SLEEP)
            ticks_during_sleep.append(ticker.count - before)
            done.set()

        async with AsyncNode('test_loop_alive') as node:
            node.create_timer(0.05, callback)
            async with asyncio.timeout(5):
                await done.wait()

    # 0.4s of 0.01s ticks is ~40. Anything above 20 proves the loop was never blocked.
    assert ticks_during_sleep[0] > 20, (
        f'loop advanced only {ticks_during_sleep[0]} ticks during a {SLEEP}s await')


@pytest.mark.asyncio
async def test_callbacks_run_on_the_event_loop_thread() -> None:
    """
    User callbacks run on the loop thread, not on the middleware thread.

    The middleware delivers on its own thread. The bridge must hop to the loop with
    call_soon_threadsafe before any user code runs.
    """
    loop_thread = threading.get_ident()
    callback_threads = []
    received = asyncio.Event()

    async def on_message(msg: String) -> None:
        callback_threads.append(threading.get_ident())
        received.set()

    async with AsyncNode('test_cb_thread') as node:
        pub = node.create_publisher(String, '/test_cb_thread_topic', TEST_QOS)
        node.create_subscription(String, '/test_cb_thread_topic', on_message, TEST_QOS)
        pub.publish(String(data='hi'))
        async with asyncio.timeout(5):
            await received.wait()

    assert callback_threads == [loop_thread]


@pytest.mark.asyncio
async def test_sequential_dispatch_serializes_and_backpressures() -> None:
    """With concurrent=False, callbacks never overlap and the total time is the sum."""
    NUM = 3
    SLEEP = 0.2
    in_flight = 0
    max_in_flight = 0
    finished = asyncio.Event()
    order = []

    async def on_message(msg: String) -> None:
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(SLEEP)
        order.append(msg.data)
        in_flight -= 1
        if len(order) == NUM:
            finished.set()

    async with AsyncNode('test_seq_dispatch') as node:
        pub = node.create_publisher(String, '/test_seq_topic', TEST_QOS)
        node.create_subscription(String, '/test_seq_topic', on_message, TEST_QOS)

        start = time.monotonic()
        for i in range(NUM):
            pub.publish(String(data=f'm{i}'))
        async with asyncio.timeout(10):
            await finished.wait()
        total = time.monotonic() - start

    assert max_in_flight == 1, f'{max_in_flight} callbacks overlapped under sequential dispatch'
    assert total >= NUM * SLEEP, f'{NUM} serialized {SLEEP}s callbacks took only {total:.3f}s'
    assert order == ['m0', 'm1', 'm2']


@pytest.mark.asyncio
async def test_concurrent_dispatch_really_overlaps() -> None:
    """With concurrent=True, callbacks overlap and the total is one sleep, not the sum."""
    NUM = 3
    SLEEP = 0.3
    in_flight = 0
    max_in_flight = 0
    done_count = 0
    finished = asyncio.Event()

    async def on_message(msg: String) -> None:
        nonlocal in_flight, max_in_flight, done_count
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(SLEEP)
        in_flight -= 1
        done_count += 1
        if done_count == NUM:
            finished.set()

    async with AsyncNode('test_conc_dispatch2') as node:
        pub = node.create_publisher(String, '/test_conc_topic2', TEST_QOS)
        node.create_subscription(
            String, '/test_conc_topic2', on_message, TEST_QOS, concurrent=True)

        start = time.monotonic()
        for i in range(NUM):
            pub.publish(String(data=f'm{i}'))
        async with asyncio.timeout(10):
            await finished.wait()
        total = time.monotonic() - start

    assert max_in_flight == NUM, f'only {max_in_flight} callbacks ran at once, expected {NUM}'
    assert total < NUM * SLEEP, f'concurrent dispatch took {total:.3f}s, as slow as serial'


@pytest.mark.asyncio
async def test_await_service_call_from_inside_a_timer_callback() -> None:
    """
    Awaiting a service call from inside a timer callback works.

    With a SingleThreadedExecutor this is the classic rclpy deadlock. On the event loop it is
    ordinary code.
    """
    result = []
    done = asyncio.Event()

    async def handler(request, response):
        await asyncio.sleep(0.05)
        response.bool_value = not request.bool_value
        return response

    async with AsyncNode('test_call_from_timer') as node:
        node.create_service(BasicTypesSrv, '/test_call_from_timer_svc', handler)
        client = node.create_client(BasicTypesSrv, '/test_call_from_timer_svc')

        async def callback() -> None:
            if done.is_set():
                return
            response = await client.call(BasicTypesSrv.Request(bool_value=False))
            result.append(response.bool_value)
            done.set()

        node.create_timer(0.05, callback)
        async with asyncio.timeout(10):
            await done.wait()

    assert result == [True]


@pytest.mark.asyncio
async def test_third_party_asyncio_library_inside_a_callback() -> None:
    """
    An unrelated asyncio library composes inside a ROS callback.

    A TCP round trip over asyncio streams stands in for aiohttp or serial_asyncio: it needs a
    live, unblocked event loop, so it cannot work under an rclpy executor.
    """
    replies = []
    received = asyncio.Event()

    async def echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        data = await reader.readline()
        writer.write(b'echo:' + data)
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(echo, '127.0.0.1', 0)
    port = server.sockets[0].getsockname()[1]

    async def on_message(msg: String) -> None:
        reader, writer = await asyncio.open_connection('127.0.0.1', port)
        writer.write(msg.data.encode() + b'\n')
        await writer.drain()
        replies.append((await reader.readline()).decode().strip())
        writer.close()
        received.set()

    try:
        async with AsyncNode('test_third_party') as node:
            pub = node.create_publisher(String, '/test_third_party_topic', TEST_QOS)
            node.create_subscription(
                String, '/test_third_party_topic', on_message, TEST_QOS)
            pub.publish(String(data='payload'))
            async with asyncio.timeout(10):
                await received.wait()
    finally:
        server.close()
        await server.wait_closed()

    assert replies == ['echo:payload']


@pytest.mark.asyncio
async def test_timer_period_holds_while_the_callback_awaits() -> None:
    """A callback that awaits less than the period does not change the firing rate."""
    PERIOD = 0.1
    TICKS = 6
    stamps = []
    done = asyncio.Event()

    async def callback() -> None:
        if done.is_set():
            return
        stamps.append(time.monotonic())
        await asyncio.sleep(0.02)
        if len(stamps) == TICKS:
            done.set()

    async with AsyncNode('test_period_holds') as node:
        node.create_timer(PERIOD, callback)
        async with asyncio.timeout(10):
            await done.wait()

    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    assert min(gaps) >= PERIOD * 0.8, f'timer fired too fast: {min(gaps):.3f}s < {PERIOD}s'
    assert max(gaps) < PERIOD * 3, f'timer drifted badly: {max(gaps):.3f}s'


@pytest.mark.asyncio
async def test_slow_timer_callback_never_reenters() -> None:
    """A callback slower than the period is not started again while it still runs."""
    in_flight = 0
    max_in_flight = 0
    calls = 0
    done = asyncio.Event()

    async def callback() -> None:
        nonlocal in_flight, max_in_flight, calls
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.2)
        in_flight -= 1
        calls += 1
        if calls == 3:
            done.set()

    async with AsyncNode('test_no_reentry') as node:
        node.create_timer(0.02, callback)
        async with asyncio.timeout(10):
            await done.wait()

    assert max_in_flight == 1, f'timer re-entered its own callback ({max_in_flight} at once)'


@pytest.mark.asyncio
async def test_clock_sleep_is_real_wall_clock_time() -> None:
    """node.get_clock().sleep() waits for real, and does not block the loop."""
    SLEEP = 0.3

    with Ticker(0.01) as ticker:
        async with AsyncNode('test_clock_sleep_real') as node:
            before = ticker.count
            start = time.monotonic()
            await node.get_clock().sleep(SLEEP)
            elapsed = time.monotonic() - start
            ticks = ticker.count - before

    assert elapsed >= SLEEP, f'clock.sleep({SLEEP}) returned after {elapsed:.3f}s'
    assert ticks > 15, f'loop advanced only {ticks} ticks during clock.sleep'


@pytest.mark.asyncio
async def test_gather_runs_ros_operations_concurrently() -> None:
    """asyncio.gather over service calls overlaps them, like any other coroutine."""
    NUM = 4
    HANDLER_SLEEP = 0.2

    async def handler(request, response):
        await asyncio.sleep(HANDLER_SLEEP)
        response.int32_value = request.int32_value * 2
        return response

    async with AsyncNode('test_gather') as node:
        node.create_service(
            BasicTypesSrv, '/test_gather_svc', handler, concurrent=True)
        client = node.create_client(BasicTypesSrv, '/test_gather_svc')

        async with asyncio.timeout(10):
            await client.wait_for_service()

        start = time.monotonic()
        async with asyncio.timeout(10):
            responses = await asyncio.gather(
                *(client.call(BasicTypesSrv.Request(int32_value=i)) for i in range(NUM)))
        total = time.monotonic() - start

    assert sorted(r.int32_value for r in responses) == [0, 2, 4, 6]
    assert total < NUM * HANDLER_SLEEP, (
        f'{NUM} gathered calls took {total:.3f}s, no better than serial')


@pytest.mark.asyncio
async def test_publishing_from_inside_a_subscription_callback() -> None:
    """A callback may publish, and a second subscription picks it up on the same loop."""
    relayed = []
    done = asyncio.Event()

    async with AsyncNode('test_relay') as node:
        out = node.create_publisher(String, '/test_relay_out', TEST_QOS)

        async def on_in(msg: String) -> None:
            await asyncio.sleep(0.01)
            out.publish(String(data=msg.data.upper()))

        async def on_out(msg: String) -> None:
            relayed.append(msg.data)
            done.set()

        node.create_subscription(String, '/test_relay_in', on_in, TEST_QOS)
        node.create_subscription(String, '/test_relay_out', on_out, TEST_QOS)
        pub_in = node.create_publisher(String, '/test_relay_in', TEST_QOS)

        pub_in.publish(String(data='hello'))
        async with asyncio.timeout(10):
            await done.wait()

    assert relayed == ['HELLO']


@pytest.mark.asyncio
async def test_cancellation_reaches_a_sleeping_callback() -> None:
    """Destroying the node cancels a callback that is parked on an await."""
    entered = asyncio.Event()
    cancelled = []

    async def callback() -> None:
        entered.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    async with AsyncNode('test_cancel_sleep') as node:
        node.create_timer(0.02, callback)
        async with asyncio.timeout(5):
            await entered.wait()

    assert cancelled == [True], 'a callback parked on await was not cancelled at shutdown'
