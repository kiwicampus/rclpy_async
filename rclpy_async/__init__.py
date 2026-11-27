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
asyncio support for rclpy, as a plain ROS 2 package.

This is ros2/rclpy#1620 ("Feature: async node") for ROS 2 Humble. It patches nothing: the
middleware bridge reaches rcl through ctypes, and every entity subclasses the rclpy class it
extends.

Example::

    import rclpy_async
    from rclpy_async import AsyncNode

    async def main():
        with rclpy_async.init():
            async with AsyncNode('my_node') as node:
                client = node.create_client(SetBool, '/set_bool')
                await client.wait_for_service()
                response = await client.call(SetBool.Request(data=True))
"""

from . import _compat  # noqa: F401  installs the 3.11 asyncio API on Python 3.10
from ._rcl_events import RclEventsUnavailable
from .action_client import AsyncActionClient, AsyncClientGoalHandle
from .client import AsyncClient
from .clock import AsyncClock
from .exceptions import TimeSourceChangedError
from .message_info import MessageInfo
from .node import AsyncNode
from .publisher import AsyncPublisher
from .service import AsyncService
from .subscription import AsyncSubscription
from .timer import AsyncTimer
from .timer_info import TimerInfo
from .utilities import init

__all__ = [
    'AsyncActionClient',
    'AsyncClient',
    'AsyncClientGoalHandle',
    'AsyncClock',
    'AsyncNode',
    'AsyncPublisher',
    'AsyncService',
    'AsyncSubscription',
    'AsyncTimer',
    'MessageInfo',
    'RclEventsUnavailable',
    'TimeSourceChangedError',
    'TimerInfo',
    'init',
]
