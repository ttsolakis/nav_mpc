from launch import LaunchDescription
from launch.actions import ExecuteProcess
import os


def generate_launch_description():
    nav_mpc_root = os.path.expanduser("~/dev_ws/src/nav_mpc")
    venv_python = os.path.join(nav_mpc_root, ".venv", "bin", "python")

    env = os.environ.copy()
    env["PYTHONPATH"] = nav_mpc_root + (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    sim = ExecuteProcess(
        cmd=[venv_python, "-m", "nav_mpc_ros.nodes.sim_node"],
        additional_env=env,
        output="screen",
    )

    mpc = ExecuteProcess(
        cmd=[venv_python, "-m", "nav_mpc_ros.nodes.nav_mpc_node"],
        additional_env=env,
        output="screen",
    )

    return LaunchDescription([sim, mpc])
