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

from rclpy.clock import ClockType
from rclpy.time import Time


class TimerInfo:
    """
    The call information of a timer.

    Humble's rcl has no rcl_timer_call_with_info, so AsyncTimer derives these two times from
    the clock instead of reading them from rcl.
    """

    def __init__(
        self, *,
        expected_call_time: int = 0,
        actual_call_time: int = 0,
        clock_type: ClockType = ClockType.SYSTEM_TIME,
    ) -> None:
        if not isinstance(clock_type, ClockType):
            raise TypeError('Clock type must be a ClockType enum')
        if expected_call_time < 0 or actual_call_time < 0:
            raise ValueError('call time values must not be negative')
        self._expected_call_time = Time(
            nanoseconds=expected_call_time, clock_type=clock_type)
        self._actual_call_time = Time(
            nanoseconds=actual_call_time, clock_type=clock_type)

    @property
    def expected_call_time(self) -> Time:
        """:return: the time the timer should have fired."""
        return self._expected_call_time

    @property
    def actual_call_time(self) -> Time:
        """:return: the time the timer did fire."""
        return self._actual_call_time
