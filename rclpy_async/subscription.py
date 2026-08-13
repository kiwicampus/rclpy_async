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
from typing import Any, AsyncGenerator, Callable, Coroutine, Optional

from rclpy.executors import await_or_execute
from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from rclpy.qos import QoSProfile
from rclpy.qos_event import SubscriptionEventCallbacks
from rclpy.subscription import Subscription

from . import _rcl_events
from ._entity import AsyncEntityMixin, InvalidHandle


class AsyncSubscription(AsyncEntityMixin, Subscription):
    """
    A subscription to a ROS topic.

    .. admonition:: Experimental

       This API is experimental.
    """

    def __init__(
        self,
        subscription_impl: _rclpy.Subscription,
        msg_type: type,
        topic: str,
        callback: Callable[..., Any],
        qos_profile: QoSProfile,
        on_destroy: Optional[Callable[['AsyncSubscription'], None]] = None,
        raw: bool = False,
        concurrent: bool = False,
        tg: Optional[Any] = None,
    ) -> None:
        super().__init__(
            subscription_impl, msg_type, topic, callback, None, qos_profile, raw,
            event_callbacks=SubscriptionEventCallbacks(use_default_callbacks=False))
        self._init_async_entity(on_destroy)
        self._wants_info = _detect_wants_info(callback)
        self._concurrent = concurrent
        self._task: Optional['asyncio.Task[None]'] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._read_event = asyncio.Event()
        if tg is not None:
            self._task = tg.create_task(self._run())

    def _on_new_message(self, _num_waiting: int) -> None:
        assert self._loop is not None
        self._loop.call_soon_threadsafe(self._read_event.set)

    def _destroy(self) -> None:
        if self._task is not None:
            self._task.cancel()
        try:
            with self.handle:
                _rcl_events.clear_callback(self.handle, 'message')
        except InvalidHandle:
            pass
        Subscription.destroy(self)

    async def _messages(self) -> AsyncGenerator[Any, None]:
        """Async generator yielding (msg, msg_info) from DDS."""
        with self.handle:
            _rcl_events.set_callback(self.handle, 'message', self._on_new_message)
        while not self._destroyed:
            with self.handle:
                msg_and_info = self.handle.take_message(self.msg_type, self.raw)
            if msg_and_info is not None:
                yield msg_and_info
            else:
                self._read_event.clear()
                await self._read_event.wait()

    def _make_callback(self, msg_and_info: Any) -> Coroutine[Any, Any, None]:
        """Create a callback coroutine from a (msg, msg_info) tuple."""
        if self._wants_info:
            return await_or_execute(self.callback, *msg_and_info)
        return await_or_execute(self.callback, msg_and_info[0])

    async def _run(self) -> None:
        """DDS bridge read loop for subscriptions."""
        self._loop = asyncio.get_running_loop()
        try:
            if self._concurrent:
                async with asyncio.TaskGroup() as tg:
                    async for msg_and_info in self._messages():
                        tg.create_task(self._make_callback(msg_and_info))
            else:
                async for msg_and_info in self._messages():
                    await self._make_callback(msg_and_info)
        except InvalidHandle:
            pass
        finally:
            self._task = None
            self.destroy()


def _detect_wants_info(callback: Callable[..., Any]) -> bool:
    """Report whether the callback takes (msg, msg_info) instead of (msg)."""
    try:
        inspect.signature(callback).bind(object())
        return False
    except TypeError:
        pass
    try:
        inspect.signature(callback).bind(object(), object())
        return True
    except TypeError:
        pass
    raise RuntimeError(
        'Subscription callback should be either be callable with one argument'
        '(to get only the message) or two (to get message and message info)')
