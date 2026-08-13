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

import asyncio
import inspect
from typing import Any, Callable, Optional

from rclpy.clock import ClockChange, JumpHandle, JumpThreshold, TimeJump
from rclpy.context import Context
from rclpy.duration import Duration
from rclpy.executors import await_or_execute
from rclpy.timer import Timer

from ._entity import AsyncEntityMixin, InvalidHandle
from .clock import AsyncClock
from .exceptions import TimeSourceChangedError
from .timer_info import TimerInfo


class AsyncTimer(AsyncEntityMixin, Timer):
    """
    A periodic timer.

    Humble's rcl has no ``rcl_timer_set_on_reset_callback`` and no ``rcl_timer_call_with_info``.
    Resets are therefore observed at the Python level, and the call info is derived from the
    clock and from ``time_until_next_call``.

    .. admonition:: Experimental

       This API is experimental.
    """

    def __init__(
        self,
        timer_period_ns: int,
        clock: AsyncClock,
        context: Context,
        callback: Callable[..., Any],
        autostart: bool = True,
        on_destroy: Optional[Callable[['AsyncTimer'], None]] = None,
        tg: Optional[Any] = None,
    ) -> None:
        self._callback: Optional[Callable[..., Any]] = None
        self._pass_info = False
        super().__init__(callback, None, timer_period_ns, clock, context=context)
        self._init_async_entity(on_destroy)
        self._task: Optional['asyncio.Task[None]'] = None
        self._reset_event = asyncio.Event()
        self._sleep_waiter: Optional['asyncio.Future[None]'] = None
        self._jump_handle: Optional[JumpHandle] = None
        if not autostart:
            Timer.cancel(self)
        if tg is not None:
            self._task = tg.create_task(self._run())

    @property
    def callback(self) -> Optional[Callable[..., Any]]:
        return self._callback

    @callback.setter
    def callback(self, cb: Optional[Callable[..., Any]]) -> None:
        self._callback = cb
        self._pass_info = False if cb is None else AsyncTimer._detect_wants_info(cb)

    @staticmethod
    def _detect_wants_info(callback: Optional[Callable[..., Any]]) -> bool:
        try:
            inspect.signature(callback).bind()
            return False
        except TypeError:
            pass
        try:
            inspect.signature(callback).bind(object())
            return True
        except TypeError:
            pass
        raise RuntimeError(
            'Timer callback must accept zero arguments '
            'or one argument (TimerInfo)')

    def _destroy(self) -> None:
        if self._task is not None:
            self._task.cancel()
        if self._jump_handle is not None:
            self._jump_handle.unregister()
        try:
            Timer.destroy(self)
        except InvalidHandle:
            pass

    def cancel(self) -> None:
        super().cancel()
        self._resolve()

    def reset(self) -> None:
        super().reset()
        self._on_reset()

    def _on_reset(self) -> None:
        self._resolve()
        self._reset_event.set()

    def _resolve(self) -> None:
        if self._sleep_waiter is not None and not self._sleep_waiter.done():
            self._sleep_waiter.set_result(None)

    def _reject(self) -> None:
        if self._sleep_waiter is not None and not self._sleep_waiter.done():
            self._sleep_waiter.set_exception(TimeSourceChangedError())

    def _on_jump(self, time_jump: TimeJump) -> None:
        if time_jump.clock_change in (
            ClockChange.ROS_TIME_ACTIVATED,
            ClockChange.ROS_TIME_DEACTIVATED,
        ):
            self._reject()
        elif time_jump.clock_change == ClockChange.ROS_TIME_NO_CHANGE and self.is_ready():
            self._resolve()

    async def _wait(self) -> None:
        duration_ns = self.time_until_next_call()
        if duration_ns is None or duration_ns <= 0:
            await asyncio.sleep(0)
            return

        loop = asyncio.get_running_loop()
        future: 'asyncio.Future[None]' = loop.create_future()
        self._sleep_waiter = future
        timer_handle: Optional[asyncio.TimerHandle] = None

        if not self._clock.ros_time_is_active:
            timer_handle = loop.call_later(duration_ns / 1e9, self._resolve)

        try:
            await future
        finally:
            self._sleep_waiter = None
            if timer_handle is not None:
                timer_handle.cancel()

    async def _call(self) -> None:
        if self.callback is None:
            raise RuntimeError('AsyncTimer cannot run with callback=None')

        # rcl_timer_call_with_info does not exist in Humble. The expected call time is the
        # deadline the timer is firing for, which is "now plus whatever is left until it".
        now_ns = self._clock.now().nanoseconds
        until_next_ns = self.time_until_next_call()
        expected_ns = now_ns if until_next_ns is None else now_ns + until_next_ns
        with self.handle:
            self.handle.call_timer()

        if self._pass_info:
            timer_info = TimerInfo(
                expected_call_time=max(expected_ns, 0),
                actual_call_time=max(now_ns, 0),
                clock_type=self._clock.clock_type)
            await await_or_execute(self.callback, timer_info)
        else:
            await await_or_execute(self.callback)

    async def _run(self) -> None:
        threshold = JumpThreshold(
            min_forward=Duration(nanoseconds=1),
            min_backward=None,
            on_clock_change=True,
        )
        self._jump_handle = self._clock.create_jump_callback(
            threshold, post_callback=self._on_jump)
        try:
            while not self._destroyed:
                try:
                    await self._wait()
                except TimeSourceChangedError:
                    continue

                if self._reset_event.is_set():
                    self._reset_event.clear()
                    continue
                if self.is_canceled():
                    await self._reset_event.wait()
                    self._reset_event.clear()
                    continue

                await self._call()
        except InvalidHandle:
            pass
        finally:
            self._task = None
            self.destroy()
