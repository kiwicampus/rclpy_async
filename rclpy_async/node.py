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
from types import TracebackType
from typing import Any, Callable, List, Optional, Set, Tuple, Type, Union

from rclpy.clock import ClockType
from rclpy.context import Context
from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_action_status_default
from rclpy.qos import qos_profile_services_default
from rclpy.qos import QoSProfile
from rclpy.type_support import check_for_type_support
from rclpy.type_support import check_is_valid_msg_type, check_is_valid_srv_type

from .action_client import AsyncActionClient
from .client import AsyncClient
from .clock import AsyncClock
from .publisher import AsyncPublisher
from .service import AsyncService
from .subscription import AsyncSubscription
from .timer import AsyncTimer


class AsyncNode(Node):
    """
    A Node in the ROS graph that runs on an asyncio event loop.

    Like :class:`Node`, ``AsyncNode`` is the primary entrypoint for communication — it creates
    publishers, subscriptions, services, clients, and timers. Unlike :class:`Node`, it does not
    require an rclpy executor: the node owns its own :class:`asyncio.TaskGroup`, and DDS callbacks
    are dispatched directly onto the running event loop.

    .. admonition:: Experimental

       This API is experimental.

    Example::

        async with AsyncNode('my_node') as node:
            node.create_subscription(String, '/topic', callback, 10)
            await my_async_task()

    See :meth:`run` for a shorthand when the node is the only foreground task.
    """

    def __init__(
        self,
        node_name: str,
        *,
        context: Optional[Context] = None,
        cli_args: Optional[List[str]] = None,
        namespace: Optional[str] = None,
        use_global_arguments: bool = True,
        enable_rosout: bool = True,
        start_parameter_services: bool = True,
        parameter_overrides: Optional[List[Parameter]] = None,
        allow_undeclared_parameters: bool = False,
        automatically_declare_parameters_from_overrides: bool = False,
    ) -> None:
        """
        Create an AsyncNode.

        :param node_name: A name to give to this node. Validated by :func:`validate_node_name`.
        :param context: The context to be associated with, or ``None`` for the default global
            context.
        :param cli_args: A list of strings of command line args to be used only by this node.
        :param namespace: The namespace to which relative topic and service names will be
            prefixed. Validated by :func:`validate_namespace`.
        :param use_global_arguments: ``False`` if the node should ignore process-wide command
            line args.
        :param enable_rosout: ``False`` if the node should ignore rosout logging.
        :param start_parameter_services: ``False`` if the node should not create parameter
            services.
        :param parameter_overrides: A list of overrides for initial values for parameters
            declared on the node.
        :param allow_undeclared_parameters: True if undeclared parameters are allowed.
        :param automatically_declare_parameters_from_overrides: If True, the parameter overrides
            will be used to implicitly declare parameters on the node during creation.
        """
        self._async_clock = AsyncClock(clock_type=ClockType.ROS_TIME)
        self._tg: Optional[Any] = None
        self._entities: Set[Any] = set()
        self._destroyed = asyncio.Event()

        super().__init__(
            node_name,
            context=context,
            cli_args=cli_args,
            namespace=namespace,
            use_global_arguments=use_global_arguments,
            enable_rosout=enable_rosout,
            start_parameter_services=start_parameter_services,
            parameter_overrides=parameter_overrides,
            allow_undeclared_parameters=allow_undeclared_parameters,
            automatically_declare_parameters_from_overrides=(
                automatically_declare_parameters_from_overrides),
        )

        # Node.__init__ builds a plain ROSClock. Swap in the async one and move the time source
        # over so sim time still drives it.
        plain_clock = self._clock
        if plain_clock in self._time_source._associated_clocks:
            self._time_source._associated_clocks.remove(plain_clock)
        self._clock = self._async_clock
        self._time_source.attach_clock(self._clock)
        plain_clock.handle.destroy_when_not_in_use()

    async def __aenter__(self) -> 'AsyncNode':
        if self._destroyed.is_set():
            raise RuntimeError('Node has been destroyed and cannot be reentered')
        if self._tg is not None:
            raise RuntimeError('Node is already running')
        _install_task_factory()
        tg = asyncio.TaskGroup()
        self._tg = await tg.__aenter__()
        for entity in self._entities:
            if hasattr(entity, '_run') and entity._task is None:
                entity._task = self._tg.create_task(entity._run())
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        try:
            self.destroy_node()
            if self._tg:
                await self._tg.__aexit__(exc_type, exc_val, exc_tb)
        finally:
            self._tg = None

    def get_clock(self) -> AsyncClock:
        """Get the async clock used by the node."""
        return self._async_clock

    def destroy_node(self) -> None:
        """Destroy the node."""
        if self._destroyed.is_set():
            return
        self._destroyed.set()
        # entity.destroy() calls back into _entities.discard, so iterate a copy.
        for entity in list(self._entities):
            entity.destroy()
        self._async_clock._destroy()
        super().destroy_node()

    async def wait_for_node(
        self,
        fully_qualified_node_name: str,
        *,
        check_interval: float = 0.1
    ) -> None:
        """
        Wait until node name is present in the system.

        To apply a timeout, wrap the call with ``async with asyncio.timeout()``.

        :param fully_qualified_node_name: Fully qualified name of the node to wait for.
        :param check_interval: Seconds between checks. Defaults to 0.1.
        """
        if not fully_qualified_node_name.startswith('/'):
            fully_qualified_node_name = f'/{fully_qualified_node_name}'

        while fully_qualified_node_name not in self.get_fully_qualified_node_names():
            await asyncio.sleep(check_interval)

    async def run(self) -> None:
        """
        Run the node until destroy_node() is called.

        Mutually exclusive with ``async with``.

        :raises RuntimeError: If the node is already running under a context manager.
        """
        async with self:
            await self._destroyed.wait()

    # --- graph helpers that Humble's Node does not have ------------------------------

    def get_fully_qualified_node_names(self) -> List[str]:
        """
        Get a list of fully qualified names for discovered nodes.

        :return: List of fully qualified node names.
        """
        return [
            namespace + name if namespace.endswith('/') else namespace + '/' + name
            for name, namespace in self.get_node_names_and_namespaces()
        ]

    def get_action_names_and_types(self) -> List[Tuple[str, List[str]]]:
        """
        Get a list of action names and types in the ROS graph.

        :return: List of tuples. The first element of each tuple is the action name, and the
          second element is a list of action types.
        """
        with self.handle:
            return self.handle.get_action_names_and_types()

    # --- entity factories -------------------------------------------------------------

    def create_publisher(
        self,
        msg_type: type,
        topic: str,
        qos_profile: Union[QoSProfile, int],
        **kwargs: Any,
    ) -> AsyncPublisher:
        """
        Create a new publisher.

        :param msg_type: The type of ROS messages the publisher will publish.
        :param topic: The name of the topic the publisher will publish to.
        :param qos_profile: A QoSProfile or a history depth to apply to the publisher.
        :return: The new publisher.
        :raises RuntimeError: If the node has been destroyed.
        """
        if self._destroyed.is_set():
            raise RuntimeError('Cannot create publisher on a destroyed node')
        qos_profile = self._validate_qos_or_depth_parameter(qos_profile)
        check_is_valid_msg_type(msg_type)

        try:
            with self.handle:
                publisher_impl = _rclpy.Publisher(
                    self.handle, msg_type, topic, qos_profile.get_c_qos_profile())
        except ValueError:
            self._validate_topic_or_service_name(topic)
            raise

        pub = AsyncPublisher(
            publisher_impl, msg_type, topic, qos_profile,
            on_destroy=self._entities.discard)
        self._entities.add(pub)
        return pub

    def create_subscription(
        self,
        msg_type: type,
        topic: str,
        callback: Callable[..., Any],
        qos_profile: Union[QoSProfile, int],
        *,
        raw: bool = False,
        concurrent: bool = False,
        **kwargs: Any,
    ) -> AsyncSubscription:
        """
        Create a new subscription.

        :param msg_type: The type of ROS messages the subscription will subscribe to.
        :param topic: The name of the topic the subscription will subscribe to.
        :param callback: A user-defined callback invoked for each received message. May be a
            sync ``def`` or ``async def`` function.
        :param qos_profile: A QoSProfile or a history depth to apply to the subscription.
        :param raw: If ``True``, received messages are delivered in raw binary representation.
        :param concurrent: If ``True``, each message is taken as soon as it arrives and its
            callback is dispatched as an independent task, so callbacks may run concurrently.
            If ``False`` (default), the next message is not taken from DDS until the previous
            callback has finished.
        :return: The new subscription.
        :raises RuntimeError: If the node has been destroyed.
        """
        if self._destroyed.is_set():
            raise RuntimeError('Cannot create subscription on a destroyed node')
        qos_profile = self._validate_qos_or_depth_parameter(qos_profile)
        check_is_valid_msg_type(msg_type)

        try:
            with self.handle:
                subscription_impl = _rclpy.Subscription(
                    self.handle, msg_type, topic, qos_profile.get_c_qos_profile())
        except ValueError:
            self._validate_topic_or_service_name(topic)
            raise

        sub = AsyncSubscription(
            subscription_impl, msg_type, topic, callback, qos_profile,
            on_destroy=self._entities.discard,
            raw=raw, concurrent=concurrent, tg=self._tg)
        self._entities.add(sub)
        return sub

    def create_service(
        self,
        srv_type: type,
        srv_name: str,
        callback: Callable[..., Any],
        *,
        qos_profile: QoSProfile = qos_profile_services_default,
        concurrent: bool = False,
        **kwargs: Any,
    ) -> AsyncService:
        """
        Create a new service server.

        :param srv_type: The service type.
        :param srv_name: The name of the service.
        :param callback: A user-defined callback invoked for each received request; its return
            value is sent as the response. May be a sync ``def`` or ``async def`` function.
        :param qos_profile: The quality of service profile to apply to the service server.
        :param concurrent: If ``True``, handlers may run concurrently.
        :return: The new service.
        :raises RuntimeError: If the node has been destroyed.
        """
        if self._destroyed.is_set():
            raise RuntimeError('Cannot create service on a destroyed node')
        check_is_valid_srv_type(srv_type)

        try:
            with self.handle:
                service_impl = _rclpy.Service(
                    self.handle, srv_type, srv_name, qos_profile.get_c_qos_profile())
        except ValueError:
            self._validate_topic_or_service_name(srv_name, is_service=True)
            raise

        srv = AsyncService(
            service_impl, srv_type, srv_name, callback, qos_profile,
            on_destroy=self._entities.discard,
            concurrent=concurrent, tg=self._tg)
        self._entities.add(srv)
        return srv

    def create_client(
        self,
        srv_type: type,
        srv_name: str,
        *,
        qos_profile: QoSProfile = qos_profile_services_default,
        **kwargs: Any,
    ) -> AsyncClient:
        """
        Create a new service client.

        :param srv_type: The service type.
        :param srv_name: The name of the service.
        :param qos_profile: The quality of service profile to apply to the service client.
        :return: The new client.
        :raises RuntimeError: If the node has been destroyed.
        """
        if self._destroyed.is_set():
            raise RuntimeError('Cannot create client on a destroyed node')
        check_is_valid_srv_type(srv_type)

        try:
            with self.handle:
                client_impl = _rclpy.Client(
                    self.handle, srv_type, srv_name, qos_profile.get_c_qos_profile())
        except ValueError:
            self._validate_topic_or_service_name(srv_name, is_service=True)
            raise

        client = AsyncClient(
            self.context, client_impl, srv_type, srv_name, qos_profile,
            on_destroy=self._entities.discard, tg=self._tg)
        self._entities.add(client)
        return client

    def create_action_client(
        self,
        action_type: type,
        action_name: str,
        *,
        goal_service_qos_profile: QoSProfile = qos_profile_services_default,
        result_service_qos_profile: QoSProfile = qos_profile_services_default,
        cancel_service_qos_profile: QoSProfile = qos_profile_services_default,
        feedback_sub_qos_profile: QoSProfile = QoSProfile(depth=10),
        status_sub_qos_profile: QoSProfile = qos_profile_action_status_default,
        **kwargs: Any,
    ) -> AsyncActionClient:
        """
        Create a new action client.

        :param action_type: The action type.
        :param action_name: The name of the action; its services and topics hang off it.
        :param goal_service_qos_profile: QoS profile for the goal service.
        :param result_service_qos_profile: QoS profile for the result service.
        :param cancel_service_qos_profile: QoS profile for the cancel service.
        :param feedback_sub_qos_profile: QoS profile for the feedback subscription.
        :param status_sub_qos_profile: QoS profile for the status subscription.
        :return: The new action client.
        :raises RuntimeError: If the node has been destroyed.
        """
        if self._destroyed.is_set():
            raise RuntimeError('Cannot create action client on a destroyed node')
        check_for_type_support(action_type)

        with self.handle:
            client_impl = _rclpy.ActionClient(
                self.handle,
                action_type,
                action_name,
                goal_service_qos_profile.get_c_qos_profile(),
                result_service_qos_profile.get_c_qos_profile(),
                cancel_service_qos_profile.get_c_qos_profile(),
                feedback_sub_qos_profile.get_c_qos_profile(),
                status_sub_qos_profile.get_c_qos_profile())

        client = AsyncActionClient(
            client_impl, action_type, action_name,
            self.get_logger().get_child('action_client'),
            on_destroy=self._entities.discard, tg=self._tg)
        self._entities.add(client)
        return client

    def create_timer(
        self,
        timer_period_sec: float,
        callback: Callable[..., Any],
        autostart: bool = True,
        **kwargs: Any,
    ) -> AsyncTimer:
        """
        Create a new timer.

        If autostart is ``True`` (the default), the timer starts immediately and calls the
        callback every ``timer_period_sec`` seconds. If autostart is ``False``, the timer is
        created but not started; start it by calling ``reset()`` on the timer object.

        :param timer_period_sec: The period (in seconds) of the timer.
        :param callback: A user-defined callback invoked when the timer expires. May be a
            sync ``def`` or ``async def`` function.
        :param autostart: Whether to start the timer after creation. Defaults to ``True``.
        :return: The new timer.
        :raises RuntimeError: If the node has been destroyed.
        """
        if self._destroyed.is_set():
            raise RuntimeError('Cannot create timer on a destroyed node')

        timer_period_ns = int(float(timer_period_sec) * 1e9)
        timer = AsyncTimer(
            timer_period_ns, self._async_clock, self.context, callback, autostart,
            on_destroy=self._entities.discard, tg=self._tg)
        self._entities.add(timer)
        return timer

    # --- destruction ------------------------------------------------------------------
    #
    # Node makes entities for itself -- the parameter services, the rosout and
    # parameter_event publishers -- and they are plain rclpy objects held in
    # Node's own lists, not in ``_entities``. Node.destroy_node() drains those
    # lists with ``while self._services: self.destroy_service(self._services[0])``,
    # and only the base method removes the entry. So an override that always
    # takes the async path never shrinks the list and that loop never ends.
    # Anything in a Node list therefore goes back to Node.

    def destroy_publisher(self, publisher: AsyncPublisher) -> bool:
        if publisher in self._publishers:
            return super().destroy_publisher(publisher)
        publisher.destroy()
        return True

    def destroy_subscription(self, subscription: AsyncSubscription) -> bool:
        if subscription in self._subscriptions:
            return super().destroy_subscription(subscription)
        subscription.destroy()
        return True

    def destroy_service(self, service: AsyncService) -> bool:
        if service in self._services:
            return super().destroy_service(service)
        service.destroy()
        return True

    def destroy_client(self, client: AsyncClient) -> bool:
        if client in self._clients:
            return super().destroy_client(client)
        client.destroy()
        return True

    def destroy_timer(self, timer: AsyncTimer) -> bool:
        if timer in self._timers:
            return super().destroy_timer(timer)
        timer.destroy()
        return True


def _install_task_factory() -> None:
    """Give the running loop the uncancel-capable Task on Python 3.10."""
    from ._compat import HAS_NATIVE_TASKGROUP
    if HAS_NATIVE_TASKGROUP:
        return
    from taskgroup.tasks import task_factory
    loop = asyncio.get_running_loop()
    if loop.get_task_factory() is None:
        loop.set_task_factory(task_factory)
