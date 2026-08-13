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
from typing import Any, AsyncGenerator, Callable, Dict, Optional, Tuple

from rclpy.client import Client
from rclpy.context import Context
from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from rclpy.qos import QoSProfile

from . import _rcl_events
from ._entity import AsyncEntityMixin, InvalidHandle


class AsyncClient(AsyncEntityMixin, Client):
    """
    A client of a ROS service.

    .. admonition:: Experimental

       This API is experimental.
    """

    def __init__(
        self,
        context: Context,
        client_impl: _rclpy.Client,
        srv_type: type,
        srv_name: str,
        qos_profile: QoSProfile,
        on_destroy: Optional[Callable[['AsyncClient'], None]] = None,
        tg: Optional[Any] = None,
    ) -> None:
        super().__init__(context, client_impl, srv_type, srv_name, qos_profile, None)
        self._init_async_entity(on_destroy)
        # Client.__init__ makes this a dict of rclpy Futures; asyncio Futures replace them.
        self._pending_requests: Dict[int, 'asyncio.Future[Any]'] = {}
        self._task: Optional['asyncio.Task[None]'] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._read_event = asyncio.Event()
        if tg is not None:
            self._task = tg.create_task(self._run())

    def _on_new_response(self, _num_waiting: int) -> None:
        assert self._loop is not None
        self._loop.call_soon_threadsafe(self._read_event.set)

    async def wait_for_service(self, *, check_interval: float = 0.1) -> None:
        """
        Wait for a service server to become ready.

        To apply a timeout, wrap the call with ``async with asyncio.timeout()``.

        :param check_interval: Seconds between checks. Defaults to 0.1.
        """
        while not self.service_is_ready():
            await asyncio.sleep(check_interval)

    def _destroy(self) -> None:
        if self._task is not None:
            self._task.cancel()
        try:
            with self.handle:
                _rcl_events.clear_callback(self.handle, 'response')
        except InvalidHandle:
            pass
        for future in self._pending_requests.values():
            future.cancel()
        Client.destroy(self)

    async def call(self, request: Any) -> Any:
        """Send a service request and await the response."""
        if self._destroyed:
            raise RuntimeError('Calling a destroyed client is forbidden')
        loop = asyncio.get_running_loop()
        future: 'asyncio.Future[Any]' = loop.create_future()
        with self.handle:
            sequence_number = self.handle.send_request(request)
        self._pending_requests[sequence_number] = future
        try:
            return await future
        finally:
            self._pending_requests.pop(sequence_number)

    async def _responses(self) -> AsyncGenerator[Tuple[Any, Any], None]:
        """Async generator yielding (header, response) from DDS."""
        with self.handle:
            _rcl_events.set_callback(self.handle, 'response', self._on_new_response)
        while not self._destroyed:
            with self.handle:
                header_and_response = self.handle.take_response(self.srv_type.Response)
            if header_and_response != (None, None):
                yield header_and_response
            else:
                self._read_event.clear()
                await self._read_event.wait()

    async def _run(self) -> None:
        """DDS bridge response loop for clients."""
        self._loop = asyncio.get_running_loop()
        try:
            async for header, response in self._responses():
                future = self._pending_requests.get(header.request_id.sequence_number)
                if future is not None and not future.done():
                    future.set_result(response)
        except InvalidHandle:
            pass
        finally:
            self._task = None
            self.destroy()
