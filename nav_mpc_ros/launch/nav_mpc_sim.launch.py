import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def _venv_python_and_env():
    repo_root = os.path.expanduser("~/dev_ws/src/nav_mpc")
    venv_python = os.path.join(repo_root, ".venv", "bin", "python")
    env = {"PYTHONPATH": repo_root + ":" + os.environ.get("PYTHONPATH", "")}
    return venv_python, env


def _py_node(module: str, name: str, params: dict | None = None):
    venv_python, env = _venv_python_and_env()
    return Node(
        package="nav_mpc_ros",
        executable=venv_python,
        name=name,
        output="screen",
        additional_env=env,
        arguments=["-m", module],
        parameters=[params or {}],
    )


def generate_launch_description():
    # Optional: toggle RViz from CLI: use_rviz:=false
    use_rviz = LaunchConfiguration("use_rviz")

    # Optional: toggle URDF + RobotModel from CLI: use_urdf:=false
    use_urdf = LaunchConfiguration("use_urdf")

    # --- rates ---
    dt_mpc = 0.1
    dt_sim = 0.002
    dt_lidar = 0.1

    # --- shared config ---
    map_path = "map.png"
    x_init = [-1.0, -2.0, 1.57079632679, 0.0, 0.0]
    x_goal = [2.0, 2.0, 0.0, 0.0, 0.0]

    path_node = _py_node(
        "nav_mpc_ros.nodes.path_node",
        "nav_mpc_path_node",
        params={
            "map_path": map_path,
            "x_init": x_init,
            "x_goal": x_goal,
            "world_width_m": 5.0,
            "occupied_threshold": 127,
            "invert_map": False,
            "inflation_radius_m": 0.25,
            "rrt_max_iters": 6000,
            "rrt_step_size": 0.10,
            "rrt_neighbor_radius": 0.30,
            "rrt_goal_sample_rate": 0.10,
            "rrt_collision_check_step": 0.02,
        },
    )

    sim_node = _py_node(
        "nav_mpc_ros.nodes.sim_node",
        "nav_mpc_sim_node",
        params={
            "dt_sim": dt_sim,
            "x_init": x_init,
            # make sure your sim_node publishes tf: map -> base_link
            # (and pose/twist topics if you added them)
        },
    )

    lidar_node = _py_node(
        "nav_mpc_ros.nodes.lidar_node",
        "nav_mpc_lidar_node",
        params={
            "dt_lidar": dt_lidar,
            "map_path": map_path,
            "world_width_m": 5.0,
            "occupied_threshold": 127,
            "invert_map": False,
            "lidar_range_max": 8.0,
            "lidar_angle_increment_deg": 0.72,
            "publish_pointcloud": True,
            "cloud_frame_id": "map",
        },
    )

    nav_mpc_node = _py_node(
        "nav_mpc_ros.nodes.nav_mpc_node",
        "nav_mpc_controller_node",
        params={
            "dt_mpc": dt_mpc,
            "N": 25,
            "embedded": True,
            "debugging": False,
            "x_goal": x_goal,
            "velocity_ref": 0.5,
        },
    )

    # --- URDF + robot_state_publisher (optional) ---
    pkg_share = get_package_share_directory("nav_mpc_ros")
    urdf_path = os.path.join(pkg_share, "urdf", "simple_rover.urdf")

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="nav_mpc_robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": open(urdf_path, "r").read()}],
        condition=IfCondition(use_urdf),
    )

    # --- RViz ---
    rviz_config = os.path.join(pkg_share, "rviz", "nav_mpc_sim.rviz")
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="nav_mpc_rviz",
        output="screen",
        arguments=["-d", rviz_config],
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument("use_urdf", default_value="true"),
        path_node,
        sim_node,
        lidar_node,
        nav_mpc_node,
        robot_state_publisher_node,
        rviz_node,
    ])
