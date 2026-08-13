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

# kind -> (rcl symbol, native rclpy setter, native rclpy clearer)
_KINDS = {
    'message': (
        'rcl_subscription_set_on_new_message_callback',
        'set_on_new_message_callback',
        'clear_on_new_message_callback',
    ),
    'request': (
        'rcl_service_set_on_new_request_callback',
        'set_on_new_request_callback',
        'clear_on_new_request_callback',
    ),
    'response': (
        'rcl_client_set_on_new_response_callback',
        'set_on_new_response_callback',
        'clear_on_new_response_callback',
    ),
}

_librcl = None
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


def _load() -> ctypes.CDLL:
    global _librcl
    if _librcl is None:
        try:
            lib = ctypes.CDLL('librcl.so')
        except OSError as ex:
            raise RclEventsUnavailable(
                'cannot load librcl.so; source the ROS 2 setup file first') from ex
        for symbol, _, _ in _KINDS.values():
            try:
                fn = getattr(lib, symbol)
            except AttributeError as ex:
                raise RclEventsUnavailable(
                    f'librcl.so has no {symbol}; this ROS 2 version is too old') from ex
            fn.argtypes = [ctypes.c_void_p, _CALLBACK_TYPE, ctypes.c_void_p]
            fn.restype = ctypes.c_int
        _librcl = lib
    return _librcl


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
    _, setter, _ = _KINDS[kind]
    return setter if hasattr(handle, setter) else ''


def set_callback(handle: Any, kind: str, callback: Callable[[int], None]) -> None:
    """
    Ask the middleware to call ``callback(number_of_events)`` when data arrives.

    :param handle: An ``_rclpy`` Subscription, Service or Client.
    :param kind: One of ``message``, ``request`` or ``response``.
    :param callback: Runs on the event loop thread, so it may touch the loop freely.
    """
    if not uses_ctypes(handle):
        getattr(handle, _native_setter(handle, kind))(callback)
        return

    symbol, _, _ = _KINDS[kind]
    lib = _load()
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
        _, _, clearer = _KINDS[kind]
        getattr(handle, clearer)()
        return

    symbol, _, _ = _KINDS[kind]
    pointer = handle.pointer
    wakeup = _alive.pop((pointer, kind), None)
    if wakeup is None:
        return
    lib = _load()
    # ctypes rejects None for a CFUNCTYPE argument, so pass a NULL function pointer.
    ret = getattr(lib, symbol)(ctypes.c_void_p(pointer), _CALLBACK_TYPE(), None)
    # rcl no longer holds the descriptor, so the eventfd can go.
    wakeup.close()
    if ret != RCL_RET_OK:
        raise RuntimeError(f'{symbol} failed to clear with rcl_ret_t {ret}')


def uses_ctypes(handle: Any) -> bool:
    """Report whether this build takes the ctypes and eventfd path. Useful in bug reports."""
    return hasattr(handle, 'pointer') and hasattr(os, 'eventfd')
