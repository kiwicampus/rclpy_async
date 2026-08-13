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

from types import TracebackType
from typing import List, Optional, Type

import rclpy
from rclpy.context import Context


class _InitContextManager:
    """Shut the context down again when the with block exits."""

    def __init__(self, context: Optional[Context]) -> None:
        self._context = context

    def __enter__(self) -> Optional[Context]:
        return self._context

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        rclpy.try_shutdown(context=self._context)


def init(*, args: Optional[List[str]] = None, context: Optional[Context] = None,
         **kwargs: object) -> _InitContextManager:
    """
    Initialize rclpy and return a context manager that shuts it down again.

    Humble's rclpy.init() returns nothing, so with rclpy.init(): does not work there.
    This helper gives newer rclpy releases' behaviour without patching rclpy.

    For the default context, rclpy.try_shutdown() also replaces the context object, so a
    later init() succeeds.
    """
    rclpy.init(args=args, context=context, **kwargs)
    return _InitContextManager(context)
