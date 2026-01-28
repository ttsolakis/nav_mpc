# nav_mpc_ros/nodes/path_node.py
from __future__ import annotations

import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray

from nav_mpc_ros.ros_paths import add_nav_mpc_repo_to_syspath
add_nav_mpc_repo_to_syspath()

from nav_mpc_ros.ros_conversions import np_to_f32multi

from simulation.environment.occupancy_map import OccupancyMapConfig, OccupancyMap2D
from simulation.path_following import (
    RRTStarConfig,
    rrt_star_plan,
    smooth_and_resample_path,
)


class PathNode(Node):
    """
    Global path node.

    Publishes (latched):
      /nav_mpc/path_xy  Float32MultiArray flattened [x0,y0,x1,y1,...]
    """

    def __init__(self) -> None:
        super().__init__("nav_mpc_path_node")

        # ---------------- Params ----------------
        self.declare_parameter("map_path", "map.png")
        self.declare_parameter("world_width_m", 5.0)
        self.declare_parameter("occupied_threshold", 127)
        self.declare_parameter("invert_map", False)

        # Match launch: full state length 5, we use [:2]
        self.declare_parameter("x_init", [-1.0, -2.0, float(np.pi/2), 0.0, 0.0])
        self.declare_parameter("x_goal", [2.0, 2.0, 0.0, 0.0, 0.0])

        self.declare_parameter("rrt_max_iters", 6000)
        self.declare_parameter("rrt_step_size", 0.10)
        self.declare_parameter("rrt_neighbor_radius", 0.30)
        self.declare_parameter("rrt_goal_sample_rate", 0.10)
        self.declare_parameter("rrt_collision_check_step", 0.02)
        self.declare_parameter("inflation_radius_m", 0.25)

        # IMPORTANT: latched QoS
        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub_path = self.create_publisher(Float32MultiArray, "/nav_mpc/path_xy", latched_qos)

        # ---------------- Compute path once ----------------
        t0 = time.perf_counter()

        occ_cfg = OccupancyMapConfig(
            map_path=str(self.get_parameter("map_path").value),
            world_width_m=float(self.get_parameter("world_width_m").value),
            occupied_threshold=int(self.get_parameter("occupied_threshold").value),
            invert=bool(self.get_parameter("invert_map").value),
        )
        occ_map = OccupancyMap2D.from_png(occ_cfg)

        x_init = np.array(self.get_parameter("x_init").value, dtype=float)
        x_goal = np.array(self.get_parameter("x_goal").value, dtype=float)
        x_init_xy = x_init[:2].copy()
        x_goal_xy = x_goal[:2].copy()

        rrt_cfg = RRTStarConfig(
            max_iters=int(self.get_parameter("rrt_max_iters").value),
            step_size=float(self.get_parameter("rrt_step_size").value),
            neighbor_radius=float(self.get_parameter("rrt_neighbor_radius").value),
            goal_sample_rate=float(self.get_parameter("rrt_goal_sample_rate").value),
            collision_check_step=float(self.get_parameter("rrt_collision_check_step").value),
            seed=1,
        )

        path = rrt_star_plan(
            occ_map=occ_map,
            start_xy=x_init_xy,
            goal_xy=x_goal_xy,
            inflation_radius_m=float(self.get_parameter("inflation_radius_m").value),
            cfg=rrt_cfg,
        )
        path = smooth_and_resample_path(path, ds=0.05, smoothing=0.01, k=3)

        t1 = time.perf_counter()
        self.get_logger().info(f"Path computed in {t1 - t0:.2f}s, publishing latched /nav_mpc/path_xy")

        # Publish latched
        self._path_xy = path[:, :2].copy()
        self.pub_path.publish(np_to_f32multi(self._path_xy))

        # Optional republish
        self.declare_parameter("republish_period_s", 5.0)
        republish_period = float(self.get_parameter("republish_period_s").value)
        if republish_period > 0.0:
            self.create_timer(republish_period, self._republish)

    def _republish(self) -> None:
        self.pub_path.publish(np_to_f32multi(self._path_xy))


def main(args=None):
    rclpy.init(args=args)
    node = PathNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
