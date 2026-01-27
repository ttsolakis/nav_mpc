# nav_mpc/launch/nav_mpc_hdw.launch.py

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    mpc = Node(
        package="nav_mpc_ros",
        executable="nav_mpc_node",
        name="nav_mpc_node",
        output="screen",
        parameters=[{
            "dt_mpc": 0.1,
            "N": 25,
            "embedded": True,
            "debugging": False,
            "velocity_ref": 0.5,
        }],
    )

    return LaunchDescription([mpc])
