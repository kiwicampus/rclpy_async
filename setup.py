from setuptools import setup

package_name = 'rclpy_async'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        # Python 3.10 has no asyncio.TaskGroup, asyncio.timeout or ExceptionGroup.
        # Neither backport is packaged for jammy, so these come from pip.
        'taskgroup>=0.2.2',
        'exceptiongroup>=1.2',
    ],
    zip_safe=True,
    maintainer='pepis',
    maintainer_email='pedro@kiwibot.com',
    description='asyncio support for rclpy on ROS 2 Humble (ros2/rclpy#1620 backport)',
    license='Apache-2.0',
    tests_require=['pytest'],
)
