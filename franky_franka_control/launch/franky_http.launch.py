from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    robot_ip = LaunchConfiguration("robot_ip")
    server_host = LaunchConfiguration("server_host")
    server_port = LaunchConfiguration("server_port")
    relative_dynamics_factor = LaunchConfiguration("relative_dynamics_factor")

    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_ip", description="Hostname or IP address of the Franka arm"),
            DeclareLaunchArgument(
                "server_host",
                default_value="0.0.0.0",
                description="Host interface used by the franky HTTP bridge",
            ),
            DeclareLaunchArgument(
                "server_port",
                default_value="5000",
                description="TCP port used by the franky HTTP bridge",
            ),
            DeclareLaunchArgument(
                "relative_dynamics_factor",
                default_value="0.2",
                description="Default franky velocity/acceleration/jerk scaling factor",
            ),
            Node(
                package="franky_franka_control",
                executable="franky_http_server.py",
                output="screen",
                arguments=[
                    "--robot-ip",
                    robot_ip,
                    "--host",
                    server_host,
                    "--port",
                    server_port,
                    "--relative-dynamics-factor",
                    relative_dynamics_factor,
                ],
            ),
        ]
    )
