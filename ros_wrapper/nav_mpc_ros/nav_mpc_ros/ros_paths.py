# nav_mpc_ros/ros_paths.py
from __future__ import annotations

import sys
from pathlib import Path

def add_nav_mpc_repo_to_syspath() -> Path:
    """
    Make the nav_mpc repo (which contains core/, simulation/, utils/) importable
    when running ROS nodes.

    Assumes this file lives at:
      nav_mpc/ros_wrapper/nav_mpc_ros/nav_mpc_ros/ros_paths.py

    and we want to add nav_mpc/ to sys.path.
    """
    here = Path(__file__).resolve()
    # .../nav_mpc/ros_wrapper/nav_mpc_ros/nav_mpc_ros/ros_paths.py
    nav_mpc_root = here.parents[4]  # -> .../nav_mpc
    if str(nav_mpc_root) not in sys.path:
        sys.path.insert(0, str(nav_mpc_root))
    return nav_mpc_root
