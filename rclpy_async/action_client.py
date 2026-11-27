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
from typing import Any, Callable, Dict, Optional, Tuple
import uuid
import weakref

from action_msgs.msg import GoalStatus
from action_msgs.srv import CancelGoal
from rclpy.executors import await_or_execute
from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from unique_identifier_msgs.msg import UUID

from . import _rcl_events
from ._entity import AsyncEntityMixin, InvalidHandle

_EVENT_KINDS = ('goal_response', 'cancel_response', 'result_response', 'feedback', 'status')
_TERMINAL_STATUSES = (
    GoalStatus.STATUS_SUCCEEDED,
    GoalStatus.STATUS_CANCELED,
    GoalStatus.STATUS_ABORTED,
)


class AsyncClientGoalHandle:
    """
    A goal sent by an :class:`AsyncActionClient`.

    .. admonition:: Experimental

       This API is experimental.
    """

    def __init__(self, action_client: 'AsyncActionClient', goal_id: UUID,
                 goal_response: Any) -> None:
        self._action_client = action_client
        self._goal_id = goal_id
        self._goal_response = goal_response
        self._status = GoalStatus.STATUS_UNKNOWN

    def __repr__(self) -> str:
        return (f'AsyncClientGoalHandle <id={bytes(self._goal_id.uuid).hex()}, '
                f'accepted={self.accepted}, status={self.status}>')

    @property
    def goal_id(self) -> UUID:
        return self._goal_id

    @property
    def stamp(self) -> Any:
        return self._goal_response.stamp

    @property
    def accepted(self) -> bool:
        return self._goal_response.accepted

    @property
    def status(self) -> int:
        """Return the last ``action_msgs/GoalStatus`` value published for this goal."""
        return self._status

    async def get_result(self) -> Any:
        """Await the goal's ``GetResult`` response, which carries ``status`` and ``result``."""
        return await self._action_client._get_result(self)

    async def cancel_goal(self) -> CancelGoal.Response:
        """Ask the server to cancel the goal and await its answer."""
        return await self._action_client._cancel_goal(self)


class AsyncActionClient(AsyncEntityMixin):
    """
    A client of a ROS action.

    rclpy's ``ActionClient`` is an executor ``Waitable``, so this class wraps the same
    ``_rclpy.ActionClient`` handle instead of subclassing it. Create it with
    :meth:`AsyncNode.create_action_client`.

    .. admonition:: Experimental

       This API is experimental.
    """

    def __init__(
        self,
        client_impl: _rclpy.ActionClient,
        action_type: Any,
        action_name: str,
        logger: Any,
        on_destroy: Optional[Callable[['AsyncActionClient'], None]] = None,
        tg: Optional[Any] = None,
    ) -> None:
        self._init_async_entity(on_destroy)
        self._client_handle = client_impl
        self._action_type = action_type
        self._action_name = action_name
        self._logger = logger
        self._pending_goals: Dict[int, Tuple['asyncio.Future[Any]', UUID]] = {}
        self._pending_cancels: Dict[int, 'asyncio.Future[Any]'] = {}
        self._pending_results: Dict[int, 'asyncio.Future[Any]'] = {}
        self._feedback_callbacks: Dict[bytes, Callable[..., Any]] = {}
        self._goal_handles: Dict[bytes, 'weakref.ref[AsyncClientGoalHandle]'] = {}
        self._task: Optional['asyncio.Task[None]'] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._read_event = asyncio.Event()
        if tg is not None:
            self._task = tg.create_task(self._run())

    @property
    def handle(self) -> _rclpy.ActionClient:
        return self._client_handle

    @property
    def action_type(self) -> Any:
        return self._action_type

    @property
    def action_name(self) -> str:
        return self._action_name

    def server_is_ready(self) -> bool:
        """Report whether an action server is matched on all three services and both topics."""
        with self.handle:
            return self.handle.is_action_server_available()

    async def wait_for_server(self, *, check_interval: float = 0.1) -> None:
        """
        Wait for an action server to become ready.

        To apply a timeout, wrap the call with ``async with asyncio.timeout()``.

        :param check_interval: Seconds between checks. Defaults to 0.1.
        """
        while not self.server_is_ready():
            await asyncio.sleep(check_interval)

    async def send_goal(
        self,
        goal: Any,
        feedback_callback: Optional[Callable[..., Any]] = None,
        goal_uuid: Optional[UUID] = None,
    ) -> AsyncClientGoalHandle:
        """
        Send a goal and await the server's acceptance or rejection.

        To wait for the outcome, await :meth:`AsyncClientGoalHandle.get_result` on the handle.

        :param goal: An instance of the action's ``Goal`` type.
        :param feedback_callback: Called with each ``FeedbackMessage`` for this goal until it
            finishes. May be a sync ``def`` or ``async def`` function.
        :param goal_uuid: The goal id, or ``None`` for a random one.
        :return: The goal handle; check ``accepted`` before awaiting its result.
        :raises TypeError: If ``goal`` is not the action's ``Goal`` type.
        :raises RuntimeError: If the client has been destroyed.
        """
        if self._destroyed:
            raise RuntimeError('Sending a goal on a destroyed action client is forbidden')
        if not isinstance(goal, self._action_type.Goal):
            raise TypeError(
                f'Expected {self._action_type.Goal.__name__}, got {type(goal).__name__}')
        request = self._action_type.Impl.SendGoalService.Request()
        request.goal_id = goal_uuid if goal_uuid is not None else _random_uuid()
        request.goal = goal
        key = bytes(request.goal_id.uuid)
        future: 'asyncio.Future[AsyncClientGoalHandle]' = (
            asyncio.get_running_loop().create_future())
        # Registered before sending, so no feedback for this goal can arrive unclaimed.
        if feedback_callback is not None:
            self._feedback_callbacks[key] = feedback_callback
        with self.handle:
            sequence_number = self.handle.send_goal_request(request)
        self._pending_goals[sequence_number] = (future, request.goal_id)
        try:
            goal_handle = await future
        except BaseException:
            self._feedback_callbacks.pop(key, None)
            raise
        finally:
            self._pending_goals.pop(sequence_number, None)
        if not goal_handle.accepted:
            self._feedback_callbacks.pop(key, None)
        return goal_handle

    async def _get_result(self, goal_handle: AsyncClientGoalHandle) -> Any:
        if self._destroyed:
            raise RuntimeError('Requesting a result on a destroyed action client is forbidden')
        request = self._action_type.Impl.GetResultService.Request()
        request.goal_id = goal_handle.goal_id
        future: 'asyncio.Future[Any]' = asyncio.get_running_loop().create_future()
        with self.handle:
            sequence_number = self.handle.send_result_request(request)
        self._pending_results[sequence_number] = future
        try:
            response = await future
        finally:
            self._pending_results.pop(sequence_number, None)
        self._feedback_callbacks.pop(bytes(goal_handle.goal_id.uuid), None)
        goal_handle._status = response.status
        return response

    async def _cancel_goal(self, goal_handle: AsyncClientGoalHandle) -> CancelGoal.Response:
        if self._destroyed:
            raise RuntimeError('Canceling a goal on a destroyed action client is forbidden')
        request = CancelGoal.Request()
        request.goal_info.goal_id = goal_handle.goal_id
        future: 'asyncio.Future[CancelGoal.Response]' = (
            asyncio.get_running_loop().create_future())
        with self.handle:
            sequence_number = self.handle.send_cancel_request(request)
        self._pending_cancels[sequence_number] = future
        try:
            return await future
        finally:
            self._pending_cancels.pop(sequence_number, None)

    def _on_event(self, _num_waiting: int) -> None:
        assert self._loop is not None
        self._loop.call_soon_threadsafe(self._read_event.set)

    def _destroy(self) -> None:
        if self._task is not None:
            self._task.cancel()
        for kind in _EVENT_KINDS:
            try:
                with self.handle:
                    _rcl_events.clear_callback(self.handle, kind)
            except InvalidHandle:
                pass
        for future, _ in self._pending_goals.values():
            future.cancel()
        for pending in (self._pending_cancels, self._pending_results):
            for future in pending.values():
                future.cancel()
        self._feedback_callbacks.clear()
        self._goal_handles.clear()
        self.handle.destroy_when_not_in_use()

    async def _run(self) -> None:
        """DDS bridge loop: takes every response and message, then sleeps until the next one."""
        self._loop = asyncio.get_running_loop()
        try:
            with self.handle:
                for kind in _EVENT_KINDS:
                    _rcl_events.set_callback(self.handle, kind, self._on_event)
            while not self._destroyed:
                if not await self._take_all():
                    self._read_event.clear()
                    await self._read_event.wait()
        except InvalidHandle:
            pass
        finally:
            self._task = None
            self.destroy()

    async def _take_all(self) -> bool:
        """Take one of each kind that is waiting and dispatch it; report whether any was."""
        impl = self._action_type.Impl
        with self.handle:
            goal = self.handle.take_goal_response(impl.SendGoalService.Response)
            cancel = self.handle.take_cancel_response(impl.CancelGoalService.Response)
            result = self.handle.take_result_response(impl.GetResultService.Response)
            feedback = self.handle.take_feedback(impl.FeedbackMessage)
            status = self.handle.take_status(impl.GoalStatusMessage)

        if goal[0] is not None:
            self._on_goal_response(*goal)
        if cancel[0] is not None:
            _resolve(self._pending_cancels, *cancel, self._unexpected, 'cancel')
        if result[0] is not None:
            _resolve(self._pending_results, *result, self._unexpected, 'result')
        if status is not None:
            self._on_status(status)
        if feedback is not None:
            callback = self._feedback_callbacks.get(bytes(feedback.goal_id.uuid))
            if callback is not None:
                await await_or_execute(callback, feedback)
        taken = (goal[0], cancel[0], result[0], feedback, status)
        return any(item is not None for item in taken)

    def _on_goal_response(self, sequence_number: int, response: Any) -> None:
        pending = self._pending_goals.get(sequence_number)
        if pending is None:
            self._unexpected('goal')
            return
        future, goal_id = pending
        if future.done():
            return
        goal_handle = AsyncClientGoalHandle(self, goal_id, response)
        if goal_handle.accepted:
            key = bytes(goal_id.uuid)
            if key in self._goal_handles and self._goal_handles[key]() is not None:
                future.set_exception(
                    RuntimeError(f'Two goals were accepted with the same ID ({goal_handle})'))
                return
            self._goal_handles[key] = weakref.ref(goal_handle)
        future.set_result(goal_handle)

    def _on_status(self, status_array: Any) -> None:
        for status in status_array.status_list:
            key = bytes(status.goal_info.goal_id.uuid)
            ref = self._goal_handles.get(key)
            goal_handle = ref() if ref is not None else None
            if goal_handle is None:
                self._goal_handles.pop(key, None)
                continue
            goal_handle._status = status.status
            if status.status in _TERMINAL_STATUSES:
                del self._goal_handles[key]

    def _unexpected(self, what: str) -> None:
        self._logger.warning(
            f'Ignoring unexpected {what} response. There may be more than one action server '
            f"for the action '{self._action_name}'")


def _resolve(pending: Dict[int, 'asyncio.Future[Any]'], sequence_number: int, response: Any,
             unexpected: Callable[[str], None], what: str) -> None:
    future = pending.get(sequence_number)
    if future is None:
        unexpected(what)
    elif not future.done():
        future.set_result(response)


def _random_uuid() -> UUID:
    return UUID(uuid=list(uuid.uuid4().bytes))
