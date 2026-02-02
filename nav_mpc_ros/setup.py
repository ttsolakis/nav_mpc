import os
from glob import glob
from setuptools import setup, find_packages

package_name = "nav_mpc_ros"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/rviz", glob("rviz/*.rviz")),
        (f"share/{package_name}/urdf", ["urdf/simple_robot.urdf"]),

    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="tasos",
    maintainer_email="tasos@todo.com",
    description="ROS2 wrapper for nav_mpc",
    license="MIT",
    entry_points={
        "console_scripts": [
            "sim_node = nav_mpc_ros.nodes.sim_node:main",
            "lidar_node = nav_mpc_ros.nodes.lidar_node:main",
            "path_node = nav_mpc_ros.nodes.path_node:main",
            "nav_mpc_node = nav_mpc_ros.nodes.nav_mpc_node:main",
        ],
    },
)
