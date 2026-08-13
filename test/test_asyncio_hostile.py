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
Hostile tests: the weird things a real asyncio program does.

These go after the parts most likely to be quietly wrong — the ctypes callback lifetime, the
claim that data is pushed rather than polled, coexistence with the rest of the asyncio ecosystem,
and reentrancy.
"""

import asyncio
from collections.abc import Generator
import gc
import os
import signal
import sys
import time

import pytest

import rclpy_async
from rclpy_async import AsyncNode
from rclpy_async import _rcl_events

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
BULK_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_ALL,
)


@pytest.fixture(autouse=True)
def rclpy_context() -> Generator[None, None, None]:
    with rclpy_async.init():
        yield


@pytest.mark.asyncio
async def test_delivery_latency_proves_push_not_poll() -> None:
    """
    A message reaches its callback in milliseconds.

    A polling loop would still pass every functional test, but it would show latency near its
    poll interval. Push shows latency near zero.
    """
    latencies = []
    got = asyncio.Event()
    sent_at = 0.0

    async def on_message(msg: String) -> None:
        latencies.append(time.monotonic() - sent_at)
        got.set()

    async with AsyncNode('test_latency') as node:
        pub = node.create_publisher(String, '/test_latency_topic', TEST_QOS)
        node.create_subscription(String, '/test_latency_topic', on_message, TEST_QOS)
        await asyncio.sleep(0.3)  # let discovery settle

        for _ in range(10):
            got.clear()
            sent_at = time.monotonic()
            pub.publish(String(data='x'))
            async with asyncio.timeout(5):
                await got.wait()

    worst = max(latencies)
    assert worst < 0.02, f'worst delivery latency {worst * 1e3:.1f} ms suggests polling'


@pytest.mark.asyncio
async def test_idle_node_burns_no_cpu() -> None:
    """An idle node spends no CPU. A busy-wait read loop would show up here."""
    async with AsyncNode('test_idle_cpu') as node:
        node.create_subscription(String, '/test_idle_topic', lambda m: None, TEST_QOS)
        node.create_service(
            BasicTypesSrv, '/test_idle_svc', lambda rq, rs: rs)
        await asyncio.sleep(0.3)

        before = time.process_time()
        await asyncio.sleep(1.0)
        cpu = time.process_time() - before

    assert cpu < 0.15, f'idle node used {cpu:.3f}s of CPU in 1.0s of wall clock'


@pytest.mark.asyncio
async def test_ctypes_callback_survives_aggressive_gc() -> None:
    """
    The middleware holds a pointer into a ctypes object. It must not be collected.

    If the registry that keeps it alive were missing, this would segfault or silently stop
    delivering rather than fail an assertion.
    """
    received = []
    done = asyncio.Event()

    async def on_message(msg: String) -> None:
        received.append(msg.data)
        if len(received) == 5:
            done.set()

    async with AsyncNode('test_gc') as node:
        pub = node.create_publisher(String, '/test_gc_topic', TEST_QOS)
        node.create_subscription(String, '/test_gc_topic', on_message, TEST_QOS)
        await asyncio.sleep(0.3)

        for i in range(5):
            gc.collect()
            gc.collect()
            pub.publish(String(data=f'm{i}'))
            await asyncio.sleep(0.05)

        async with asyncio.timeout(5):
            await done.wait()

    assert received == ['m0', 'm1', 'm2', 'm3', 'm4']


@pytest.mark.asyncio
async def test_wakeup_eventfd_is_registered_then_released() -> None:
    """
    The bridge registers an eventfd per entity and releases it on destroy.

    rcl is handed glibc's eventfd_write, not a Python function, so nothing Python-owned is
    reachable from a middleware thread. What must be tracked is the descriptor: the middleware
    holds it as its user data, so it has to outlive the registration and be closed after it.
    """
    async with AsyncNode('test_wakeup') as node:
        sub = node.create_subscription(
            String, '/test_wakeup_topic', lambda m: None, TEST_QOS)
        await asyncio.sleep(0)  # let the reader task register the callback

        if not _rcl_events.uses_ctypes(sub.handle):
            pytest.skip('this build uses the native rclpy setters, so no eventfd exists')

        key = (sub.handle.pointer, 'message')
        assert key in _rcl_events._alive, 'no eventfd was registered for the subscription'
        wakeup = _rcl_events._alive[key]
        os.fstat(wakeup.fd)  # raises if the descriptor is not open

        sub.destroy()
        await asyncio.sleep(0.05)
        assert key not in _rcl_events._alive, 'the eventfd outlived the entity'
        with pytest.raises(OSError):
            os.fstat(wakeup.fd)


@pytest.mark.asyncio
async def test_entity_churn_does_not_leak_file_descriptors() -> None:
    """One eventfd per entity means entity churn is a descriptor leak waiting to happen."""
    def open_fds() -> int:
        return len(os.listdir('/proc/self/fd'))

    async with AsyncNode('test_fd_churn') as node:
        # Create and drop one first, so any one-off allocation is outside the measurement.
        node.create_subscription(
            String, '/test_fd_warmup', lambda m: None, TEST_QOS).destroy()
        await asyncio.sleep(0.05)

        before = open_fds()
        for i in range(30):
            sub = node.create_subscription(
                String, f'/test_fd_{i}', lambda m: None, TEST_QOS)
            await asyncio.sleep(0)
            sub.destroy()
        await asyncio.sleep(0.1)
        growth = open_fds() - before

    assert growth < 5, f'{growth} file descriptors leaked over 30 create/destroy cycles'


@pytest.mark.asyncio
async def test_entity_churn_does_not_leak_callback_registry() -> None:
    """Creating and destroying many entities leaves no registered callbacks behind."""
    baseline = len(_rcl_events._alive)

    async with AsyncNode('test_churn') as node:
        for i in range(30):
            sub = node.create_subscription(
                String, f'/test_churn_{i}', lambda m: None, TEST_QOS)
            await asyncio.sleep(0)  # let the reader task register its callback
            sub.destroy()
        await asyncio.sleep(0.1)

    leaked = len(_rcl_events._alive) - baseline
    assert leaked == 0, f'{leaked} middleware callbacks were never cleared'


@pytest.mark.asyncio
async def test_offload_blocking_work_with_to_thread() -> None:
    """A callback offloads blocking work to a thread and the loop keeps turning."""
    ticks = 0
    result = []
    done = asyncio.Event()

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    def blocking_work() -> str:
        time.sleep(0.3)   # a real blocking call, on a worker thread
        return 'from thread'

    tick_task = asyncio.get_running_loop().create_task(ticker())
    try:
        async with AsyncNode('test_to_thread') as node:
            async def on_message(msg: String) -> None:
                before = ticks
                result.append(await asyncio.to_thread(blocking_work))
                result.append(ticks - before)
                done.set()

            pub = node.create_publisher(String, '/test_to_thread_topic', TEST_QOS)
            node.create_subscription(
                String, '/test_to_thread_topic', on_message, TEST_QOS)
            pub.publish(String(data='go'))
            async with asyncio.timeout(10):
                await done.wait()
    finally:
        tick_task.cancel()

    assert result[0] == 'from thread'
    assert result[1] > 15, f'loop advanced only {result[1]} ticks during a threaded 0.3s block'


@pytest.mark.asyncio
async def test_asyncio_subprocess_inside_a_callback() -> None:
    """A callback spawns a child process and awaits it, which needs the loop's child watcher."""
    output = []
    done = asyncio.Event()

    async def on_message(msg: String) -> None:
        proc = await asyncio.create_subprocess_exec(
            '/bin/echo', msg.data, stdout=asyncio.subprocess.PIPE)
        stdout, _ = await proc.communicate()
        output.append(stdout.decode().strip())
        done.set()

    async with AsyncNode('test_subprocess') as node:
        pub = node.create_publisher(String, '/test_subprocess_topic', TEST_QOS)
        node.create_subscription(String, '/test_subprocess_topic', on_message, TEST_QOS)
        pub.publish(String(data='child-process-ok'))
        async with asyncio.timeout(10):
            await done.wait()

    assert output == ['child-process-ok']


@pytest.mark.asyncio
async def test_asyncio_signal_handler_coexists_with_rclpy() -> None:
    """
    An asyncio signal handler still works while rclpy owns SIGINT and SIGTERM.

    rclpy installs its own handlers at init. SIGUSR1 must still reach the loop.
    """
    fired = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGUSR1, fired.set)

    try:
        async with AsyncNode('test_signal'):
            os.kill(os.getpid(), signal.SIGUSR1)
            async with asyncio.timeout(5):
                await fired.wait()
    finally:
        loop.remove_signal_handler(signal.SIGUSR1)


@pytest.mark.asyncio
async def test_queue_pipeline_keeps_every_message_in_order() -> None:
    """A callback feeds an asyncio.Queue and a consumer drains it, losing nothing."""
    COUNT = 200
    queue: 'asyncio.Queue[str]' = asyncio.Queue()
    drained = []
    done = asyncio.Event()

    async def consumer() -> None:
        while True:
            drained.append(await queue.get())
            if len(drained) == COUNT:
                done.set()

    async with AsyncNode('test_queue') as node:
        pub = node.create_publisher(String, '/test_queue_topic', BULK_QOS)
        node.create_subscription(
            String, '/test_queue_topic', lambda m: queue.put_nowait(m.data), BULK_QOS)
        consumer_task = asyncio.get_running_loop().create_task(consumer())
        await asyncio.sleep(0.3)

        try:
            for i in range(COUNT):
                pub.publish(String(data=str(i)))
            async with asyncio.timeout(15):
                await done.wait()
        finally:
            consumer_task.cancel()

    assert drained == [str(i) for i in range(COUNT)]


@pytest.mark.asyncio
async def test_semaphore_bounds_concurrent_dispatch() -> None:
    """An asyncio.Semaphore inside a concurrent callback limits real overlap."""
    LIMIT = 2
    COUNT = 6
    semaphore = asyncio.Semaphore(LIMIT)
    in_flight = 0
    max_in_flight = 0
    finished = 0
    done = asyncio.Event()

    async def on_message(msg: String) -> None:
        nonlocal in_flight, max_in_flight, finished
        async with semaphore:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.1)
            in_flight -= 1
        finished += 1
        if finished == COUNT:
            done.set()

    async with AsyncNode('test_semaphore') as node:
        pub = node.create_publisher(String, '/test_sem_topic', BULK_QOS)
        node.create_subscription(
            String, '/test_sem_topic', on_message, BULK_QOS, concurrent=True)
        await asyncio.sleep(0.3)
        for i in range(COUNT):
            pub.publish(String(data=str(i)))
        async with asyncio.timeout(15):
            await done.wait()

    assert max_in_flight == LIMIT, f'semaphore of {LIMIT} allowed {max_in_flight} at once'


@pytest.mark.asyncio
async def test_cancelled_call_does_not_leak_a_pending_request() -> None:
    """A client call cancelled by a timeout removes its own bookkeeping."""
    async def handler(request, response):
        await asyncio.sleep(5)
        return response

    async with AsyncNode('test_leak') as node:
        node.create_service(BasicTypesSrv, '/test_leak_svc', handler, concurrent=True)
        client = node.create_client(BasicTypesSrv, '/test_leak_svc')
        async with asyncio.timeout(10):
            await client.wait_for_service()

        for _ in range(5):
            # Builtin TimeoutError, not asyncio.TimeoutError. See the test below.
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(0.1):
                    await client.call(BasicTypesSrv.Request())

        assert client._pending_requests == {}, (
            f'{len(client._pending_requests)} cancelled requests were left behind')


@pytest.mark.asyncio
async def test_many_entities_on_one_loop() -> None:
    """Fifty subscriptions on one node all receive, so the TaskGroup scales."""
    COUNT = 50
    seen = set()
    done = asyncio.Event()

    def make_callback(index: int):
        async def on_message(msg: String) -> None:
            seen.add(index)
            if len(seen) == COUNT:
                done.set()
        return on_message

    async with AsyncNode('test_many') as node:
        pubs = []
        for i in range(COUNT):
            pubs.append(node.create_publisher(String, f'/test_many_{i}', TEST_QOS))
            node.create_subscription(
                String, f'/test_many_{i}', make_callback(i), TEST_QOS)
        await asyncio.sleep(0.5)

        for i, pub in enumerate(pubs):
            pub.publish(String(data=str(i)))
        async with asyncio.timeout(15):
            await done.wait()

    assert len(seen) == COUNT


@pytest.mark.asyncio
async def test_timer_recovers_after_the_loop_is_blocked() -> None:
    """A synchronous block stalls everything, and the timer resumes right after."""
    stamps = []
    done = asyncio.Event()

    async def callback() -> None:
        stamps.append(time.monotonic())
        if len(stamps) == 6:
            done.set()

    async with AsyncNode('test_recover') as node:
        node.create_timer(0.05, callback)
        await asyncio.sleep(0.15)
        time.sleep(0.4)          # a rude synchronous block, on purpose
        async with asyncio.timeout(10):
            await done.wait()

    assert len(stamps) >= 6


@pytest.mark.asyncio
async def test_timeout_raises_the_builtin_timeouterror_on_python_310() -> None:
    """
    Pin a portability trap that bites real user code.

    Python 3.11 merged asyncio.TimeoutError into the builtin TimeoutError. On the Python 3.10
    that Humble ships they are still unrelated classes, and the backported asyncio.timeout
    follows 3.11 and raises the builtin one. So ``except asyncio.TimeoutError`` silently misses
    a timeout here, while ``except TimeoutError`` works on every version.
    """
    async with AsyncNode('test_timeout_type'):
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.01):
                await asyncio.sleep(5)

    if sys.version_info < (3, 11):
        assert asyncio.TimeoutError is not TimeoutError
        # The trap itself: the 3.10 asyncio class does not catch what asyncio.timeout raises.
        assert not issubclass(TimeoutError, asyncio.TimeoutError)
    else:
        assert asyncio.TimeoutError is TimeoutError


@pytest.mark.asyncio
async def test_wait_for_still_raises_asyncio_timeouterror() -> None:
    """asyncio.wait_for keeps 3.10 semantics, so the two spellings really do differ."""
    async with AsyncNode('test_wait_for_type'):
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.sleep(5), timeout=0.01)
