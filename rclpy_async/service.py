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
from typing import Any, AsyncGenerator, Callable, Optional, Tuple

from rclpy.executors import await_or_execute
from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from rclpy.qos import QoSProfile
from rclpy.service import Service

from . import _rcl_events
from ._entity import AsyncEntityMixin, InvalidHandle


class AsyncService(AsyncEntityMixin, Service):
    """
    A server for a ROS service.

    .. admonition:: Experimental

       This API is experimental.
    """

    def __init__(
        self,
        service_impl: _rclpy.Service,
        srv_type: type,
        srv_name: str,
        callback: Callable[..., Any],
        qos_profile: QoSProfile,
        on_destroy: Optional[Callable[['AsyncService'], None]] = None,
        concurrent: bool = False,
        tg: Optional[Any] = None,
    ) -> None:
        super().__init__(service_impl, srv_type, srv_name, callback, None, qos_profile)
        self._init_async_entity(on_destroy)
        self._concurrent = concurrent
        self._task: Optional['asyncio.Task[None]'] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._read_event = asyncio.Event()
        if tg is not None:
            self._task = tg.create_task(self._run())

    def _on_new_request(self, _num_waiting: int) -> None:
        assert self._loop is not None
        self._loop.call_soon_threadsafe(self._read_event.set)

    def _destroy(self) -> None:
        if self._task is not None:
            self._task.cancel()
        try:
            with self.handle:
                _rcl_events.clear_callback(self.handle, 'request')
        except InvalidHandle:
            pass
        Service.destroy(self)

    async def _handle_request(self, request: Any, header: Any) -> None:
        response = await await_or_execute(
            self.callback, request, self.srv_type.Response())
        self.send_response(response, header)

    async def _requests(self) -> AsyncGenerator[Tuple[Any, Any], None]:
        """Async generator yielding (request, header) from DDS."""
        with self.handle:
            _rcl_events.set_callback(self.handle, 'request', self._on_new_request)
        while not self._destroyed:
            with self.handle:
                request_and_header = self.handle.service_take_request(self.srv_type.Request)
            if request_and_header != (None, None):
                yield request_and_header
            else:
                self._read_event.clear()
                await self._read_event.wait()

    async def _run(self) -> None:
        """DDS bridge read loop for services."""
        self._loop = asyncio.get_running_loop()
        try:
            if self._concurrent:
                async with asyncio.TaskGroup() as tg:
                    async for request, header in self._requests():
                        tg.create_task(self._handle_request(request, header))
            else:
                async for request, header in self._requests():
                    await self._handle_request(request, header)
        except InvalidHandle:
            pass
        finally:
            self._task = None
            self.destroy()
