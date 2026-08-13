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
Python 3.11 asyncio features backported onto the Python 3.10 that ROS 2 Humble ships.

Importing this module installs :class:`asyncio.TaskGroup`, :func:`asyncio.timeout` and the
builtin ``ExceptionGroup`` when the running interpreter is older than 3.11. On 3.11 and later
it does nothing.
"""

import asyncio
import builtins
import sys

__all__ = ['HAS_NATIVE_TASKGROUP']

HAS_NATIVE_TASKGROUP = sys.version_info >= (3, 11)


def _install() -> None:
    if HAS_NATIVE_TASKGROUP:
        return

    from exceptiongroup import BaseExceptionGroup, ExceptionGroup
    if not hasattr(builtins, 'ExceptionGroup'):
        builtins.ExceptionGroup = ExceptionGroup
        builtins.BaseExceptionGroup = BaseExceptionGroup

    import taskgroup
    if not hasattr(asyncio, 'TaskGroup'):
        asyncio.TaskGroup = taskgroup.TaskGroup
    if not hasattr(asyncio, 'timeout'):
        asyncio.timeout = taskgroup.timeout
        asyncio.timeout_at = taskgroup.timeout_at
        asyncio.Timeout = taskgroup.Timeout

    # The backported TaskGroup and timeout need Task.uncancel(), which arrived in 3.11.
    # taskgroup ships a Task subclass that provides it; make it the default for every loop.
    _patch_task_factory()


def _patch_task_factory() -> None:
    import taskgroup.tasks

    task_factory = getattr(taskgroup.tasks, 'task_factory', None)
    if task_factory is None:
        return

    base_new_event_loop = asyncio.BaseEventLoop.__init__

    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        base_new_event_loop(self, *args, **kwargs)
        self.set_task_factory(task_factory)

    asyncio.BaseEventLoop.__init__ = __init__

    # Loops created before this import (pytest-asyncio makes one early) get patched on use.
    try:
        loop = asyncio.get_event_loop_policy().get_event_loop()
    except Exception:
        return
    if loop.get_task_factory() is None:
        loop.set_task_factory(task_factory)


_install()
