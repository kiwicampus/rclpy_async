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
The DDS listener bridge, without any change to rclpy.

Neither Humble's nor Jazzy's ``_rclpy`` extension has ``set_on_new_message_callback``, but the
``rcl`` C library has ``rcl_subscription_set_on_new_message_callback``, and the pybind11 entity
classes expose the raw ``rcl_*_t`` pointer as ``handle.pointer``. This module calls the C
function through ctypes.

The callback given to rcl is not a Python function. It is glibc's ``eventfd_write``, whose
signature ``int eventfd_write(int fd, uint64_t value)`` matches the ``(user_data,
number_of_events)`` pair that rcl passes, so the middleware writes the event count straight into
an eventfd and the event loop wakes up on it through ``loop.add_reader()``.

Handing rcl a Python callback instead looks simpler but deadlocks. That callback runs on a
middleware thread, which must take the GIL to enter Python, while the thread that destroys any
entity holds the GIL for the whole of ``rcl_*_fini``, and under rmw_cyclonedds that fini waits
for the middleware thread to make progress. Destroying a client while a service still holds a
Python listener callback is enough to hang the process. The eventfd never enters Python, so no
middleware thread ever waits for the GIL.

Action clients take the same path through ``rcl_action``'s five setters, one per inner client
and subscription. They have no native rclpy setter, so a build without ``pointer`` cannot serve
them.

A build whose entity handles expose no ``pointer`` falls back to the native rclpy setters, with
the deadlock above as a known risk.
"""

import asyncio
import ctypes
import os
from typing import Any, Callable, Dict, Tuple

RCL_RET_OK = 0

# void (*)(const void * user_data, size_t number_of_events)
_CALLBACK_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_size_t)

_RCL = 'librcl.so'
_RCL_ACTION = 'librcl_action.so'

# kind -> (library, rcl symbol, native rclpy setter, native rclpy clearer)
_KINDS = {
    'message': (
        _RCL,
        'rcl_subscription_set_on_new_message_callback',
        'set_on_new_message_callback',
        'clear_on_new_message_callback',
    ),
    'request': (
        _RCL,
        'rcl_service_set_on_new_request_callback',
        'set_on_new_request_callback',
        'clear_on_new_request_callback',
    ),
    'response': (
        _RCL,
        'rcl_client_set_on_new_response_callback',
        'set_on_new_response_callback',
        'clear_on_new_response_callback',
    ),
    'goal_response': (_RCL_ACTION, 'rcl_action_client_set_goal_client_callback', '', ''),
    'cancel_response': (_RCL_ACTION, 'rcl_action_client_set_cancel_client_callback', '', ''),
    'result_response': (_RCL_ACTION, 'rcl_action_client_set_result_client_callback', '', ''),
    'feedback': (_RCL_ACTION, 'rcl_action_client_set_feedback_subscription_callback', '', ''),
    'status': (_RCL_ACTION, 'rcl_action_client_set_status_subscription_callback', '', ''),
}

_libraries: Dict[str, ctypes.CDLL] = {}
_eventfd_write = None
# The middleware keeps the file descriptor as its user data, so the eventfd must outlive the
# call. Closing it early is harmless -- a write to a closed descriptor only returns an error --
# but a wakeup that lands nowhere stalls the entity, so the descriptor stays open until the
# callback is cleared.
_alive: Dict[Tuple[int, str], '_Wakeup'] = {}


class RclEventsUnavailable(RuntimeError):
    """The running ROS 2 build exposes no way to register a middleware callback."""


class _Wakeup:
    """
    An eventfd that the middleware writes to and the event loop reads.

    ``callback(number_of_events)`` runs on the event loop thread, once per readable event.
    """

    def __init__(self, callback: Callable[[int], None]) -> None:
        self._callback = callback
        self._loop = asyncio.get_running_loop()
        self.fd = os.eventfd(0, os.EFD_CLOEXEC | os.EFD_NONBLOCK)
        self._loop.add_reader(self.fd, self._on_readable)

    def _on_readable(self) -> None:
        self._callback(os.eventfd_read(self.fd))

    def close(self) -> None:
        if not self._loop.is_closed():
            self._loop.remove_reader(self.fd)
        os.close(self.fd)


def _load(library: str) -> ctypes.CDLL:
    """Load ``library`` once and type every setter it provides."""
    lib = _libraries.get(library)
    if lib is None:
        try:
            lib = ctypes.CDLL(library)
        except OSError as ex:
            raise RclEventsUnavailable(
                f'cannot load {library}; source the ROS 2 setup file first') from ex
        for kind_library, symbol, _, _ in _KINDS.values():
            if kind_library != library:
                continue
            try:
                fn = getattr(lib, symbol)
            except AttributeError as ex:
                raise RclEventsUnavailable(
                    f'{library} has no {symbol}; this ROS 2 version is too old') from ex
            fn.argtypes = [ctypes.c_void_p, _CALLBACK_TYPE, ctypes.c_void_p]
            fn.restype = ctypes.c_int
        _libraries[library] = lib
    return lib


def _load_eventfd_write() -> Any:
    """Return glibc's ``eventfd_write``, seen as an rcl event callback."""
    global _eventfd_write
    if _eventfd_write is None:
        try:
            libc = ctypes.CDLL('libc.so.6')
            _eventfd_write = ctypes.cast(libc.eventfd_write, _CALLBACK_TYPE)
        except (OSError, AttributeError) as ex:
            raise RclEventsUnavailable('libc.so.6 has no eventfd_write') from ex
    return _eventfd_write


def _native_setter(handle: Any, kind: str) -> str:
    """Return the native rclpy method name if this build has one, else an empty string."""
    _, _, setter, _ = _KINDS[kind]
    return setter if setter and hasattr(handle, setter) else ''


def set_callback(handle: Any, kind: str, callback: Callable[[int], None]) -> None:
    """
    Ask the middleware to call ``callback(number_of_events)`` when data arrives.

    :param handle: An ``_rclpy`` Subscription, Service, Client or ActionClient.
    :param kind: A key of ``_KINDS``: ``message``, ``request``, ``response`` or one of the
        action client kinds.
    :param callback: Runs on the event loop thread, so it may touch the loop freely.
    """
    if not uses_ctypes(handle):
        setter = _native_setter(handle, kind)
        if not setter:
            raise RclEventsUnavailable(
                f'{type(handle).__name__} has no pointer and no native setter for {kind}')
        getattr(handle, setter)(callback)
        return

    library, symbol, _, _ = _KINDS[kind]
    lib = _load(library)
    write_event = _load_eventfd_write()
    pointer = handle.pointer
    wakeup = _Wakeup(callback)
    ret = getattr(lib, symbol)(
        ctypes.c_void_p(pointer), write_event, ctypes.c_void_p(wakeup.fd))
    if ret != RCL_RET_OK:
        wakeup.close()
        raise RuntimeError(f'{symbol} failed with rcl_ret_t {ret}')
    # Store only after the call succeeds, so a failure leaves nothing registered.
    _alive[(pointer, kind)] = wakeup


def clear_callback(handle: Any, kind: str) -> None:
    """
    Remove the middleware callback.

    Always call this before the handle is destroyed. A callback that fires against a destroyed
    entity reads freed memory.
    """
    if not uses_ctypes(handle):
        _, _, _, clearer = _KINDS[kind]
        if clearer:
            getattr(handle, clearer)()
        return

    library, symbol, _, _ = _KINDS[kind]
    pointer = handle.pointer
    wakeup = _alive.pop((pointer, kind), None)
    if wakeup is None:
        return
    lib = _load(library)
    # ctypes rejects None for a CFUNCTYPE argument, so pass a NULL function pointer.
    ret = getattr(lib, symbol)(ctypes.c_void_p(pointer), _CALLBACK_TYPE(), None)
    # rcl no longer holds the descriptor, so the eventfd can go.
    wakeup.close()
    if ret != RCL_RET_OK:
        raise RuntimeError(f'{symbol} failed to clear with rcl_ret_t {ret}')


def uses_ctypes(handle: Any) -> bool:
    """Report whether this build takes the ctypes and eventfd path. Useful in bug reports."""
    return hasattr(handle, 'pointer') and hasattr(os, 'eventfd')
