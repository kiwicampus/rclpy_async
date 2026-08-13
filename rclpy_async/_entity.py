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

from typing import Callable, Optional

from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy

InvalidHandle = _rclpy.InvalidHandle


class AsyncEntityMixin:
    """
    Destruction bookkeeping shared by every async entity.

    ``destroy()`` is idempotent, tells the owning node to forget the entity, and then runs the
    entity specific ``_destroy()``.
    """

    def _init_async_entity(self, on_destroy: Optional[Callable[..., None]]) -> None:
        self._on_destroy = on_destroy
        self._destroyed = False

    @property
    def destroyed(self) -> bool:
        return self._destroyed

    def destroy(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        if self._on_destroy is not None:
            self._on_destroy(self)
        self._destroy()

    def _destroy(self) -> None:
        raise NotImplementedError
