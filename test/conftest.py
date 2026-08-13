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

"""Test-only shims. rclpy.experimental installs TaskGroup, timeout and ExceptionGroup."""

import asyncio




if not hasattr(asyncio, 'Barrier'):
    class Barrier:
        """The subset of asyncio.Barrier (3.11+) that these tests use."""

        def __init__(self, parties: int) -> None:
            if parties < 1:
                raise ValueError('parties must be > 0')
            self._parties = parties
            self._count = 0
            self._event = asyncio.Event()

        @property
        def parties(self) -> int:
            return self._parties

        @property
        def n_waiting(self) -> int:
            return self._count

        async def wait(self) -> int:
            index = self._count
            self._count += 1
            if self._count >= self._parties:
                self._event.set()
            await self._event.wait()
            return index

    asyncio.Barrier = Barrier
