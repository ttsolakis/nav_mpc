# nav_mpc_ros/nodes/sim_node.py
from __future__ import annotations

import time
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from nav_mpc_ros.ros_paths import add_nav_mpc_repo_to_syspath
add_nav_mpc_repo_to_syspath()

from nav_mpc_ros.ros_conversions import np_to_f32multi, f32multi_to_np

# nav_mpc imports (repo-root style, same as main.py)
from simulation.simulator import ContinuousSimulator, SimulatorConfig
from simulation.environment.occupancy_map import OccupancyMapConfig, OccupancyMap2D
from simulation.lidar import LidarSimulator2D, LidarConfig
from simulation.path_following import (
    RRTStarConfig,
    rrt_star_plan,
    smooth_and_resample_path,
)

class SimNode(Node):
    def __init__(self) -> None:
        super().__init__("nav_mpc_sim_node")

        # ---------------- Params ----------------
        self.declare_parameter("dt_sim", 0.1)
        self.declare_parameter("map_path", "map.png")
        self.declare_parameter("world_width_m", 5.0)
        self.declare_parameter("occupied_threshold", 127)
        self.declare_parameter("invert_map", False)

        # For your unicycle example (length 5)
        self.declare_parameter("x_init", [-1.0, -2.0, float(np.pi / 2), 0.0, 0.0])
        self.declare_parameter("x_goal", [2.0, 2.0, 0.0, 0.0, 0.0])

        # Lidar
        self.declare_parameter("lidar_range_max", 8.0)
        self.declare_parameter("lidar_angle_increment_deg", 0.72)

        # Global planner
        self.declare_parameter("rrt_max_iters", 6000)
        self.declare_parameter("rrt_step_size", 0.10)
        self.declare_parameter("rrt_neighbor_radius", 0.30)
        self.declare_parameter("rrt_goal_sample_rate", 0.10)
        self.declare_parameter("rrt_collision_check_step", 0.02)
        self.declare_parameter("inflation_radius_m", 0.25)

        # ---------------- Publishers/Subscribers ----------------
        self.pub_state = self.create_publisher(Float32MultiArray, "/nav_mpc/state", 10)
        self.pub_obstacles = self.create_publisher(Float32MultiArray, "/nav_mpc/obstacles_xy", 10)
        self.pub_global_path = self.create_publisher(Float32MultiArray, "/nav_mpc/global_path_xy", 1)

        self.sub_cmd = self.create_subscription(
            Float32MultiArray, "/nav_mpc/cmd", self._cmd_cb, 10
        )

        # ---------------- Init sim world ----------------
        dt = float(self.get_parameter("dt_sim").value)
        map_path = str(self.get_parameter("map_path").value)

        x_init = np.array(self.get_parameter("x_init").value, dtype=float)
        x_goal = np.array(self.get_parameter("x_goal").value, dtype=float)

        # system comes from nav_mpc core setup (but sim node doesn't need objective/QP)
        from core.problem_setup import setup_path_tracking_unicycle
        _, system, _, _, _, _ = setup_path_tracking_unicycle.setup_problem()

        sim_cfg = SimulatorConfig(dt=dt, method="rk4", substeps=10)
        self.sim = ContinuousSimulator(system, sim_cfg)

        occ_cfg = OccupancyMapConfig(
            map_path=map_path,
            world_width_m=float(self.get_parameter("world_width_m").value),
            occupied_threshold=int(self.get_parameter("occupied_threshold").value),
            invert=bool(self.get_parameter("invert_map").value),
        )
        self.occ_map = OccupancyMap2D.from_png(occ_cfg)

        lidar_cfg = LidarConfig(
            range_max=float(self.get_parameter("lidar_range_max").value),
            angle_increment=np.deg2rad(float(self.get_parameter("lidar_angle_increment_deg").value)),
            seed=1,
            noise_std=0.0,
            drop_prob=0.0,
            ray_step=None,
        )
        self.lidar = LidarSimulator2D(occ_map=self.occ_map, cfg=lidar_cfg)

        # Global path once
        rrt_cfg = RRTStarConfig(
            max_iters=int(self.get_parameter("rrt_max_iters").value),
            step_size=float(self.get_parameter("rrt_step_size").value),
            neighbor_radius=float(self.get_parameter("rrt_neighbor_radius").value),
            goal_sample_rate=float(self.get_parameter("rrt_goal_sample_rate").value),
            collision_check_step=float(self.get_parameter("rrt_collision_check_step").value),
            seed=1,
        )

        t0 = time.perf_counter()
        path = rrt_star_plan(
            occ_map=self.occ_map,
            start_xy=x_init[:2],
            goal_xy=x_goal[:2],
            inflation_radius_m=float(self.get_parameter("inflation_radius_m").value),
            cfg=rrt_cfg,
        )
        self.global_path = smooth_and_resample_path(path, ds=0.05, smoothing=0.01, k=3)
        self.get_logger().info(f"Global path computed in {time.perf_counter() - t0:.2f}s")

        # Publish global path
        self.pub_global_path.publish(np_to_f32multi(self.global_path[:, :2]))

        # State + command memory (sample-and-hold)
        self.x = x_init.copy()
        self.u_latest = np.zeros(system.input_dim, dtype=float)

        self.timer = self.create_timer(dt, self._tick)

    def _cmd_cb(self, msg: Float32MultiArray) -> None:
        u = f32multi_to_np(msg, dtype=np.float64)
        self.u_latest = u.copy()

    def _tick(self) -> None:
        # 1) step sim (sample-and-hold latest command)
        self.x = self.sim.step(self.x, self.u_latest)

        # 2) lidar -> obstacles
        pose = np.array([self.x[0], self.x[1], self.x[2]], dtype=float)
        scan = self.lidar.scan(pose)
        obstacles_xy = self.lidar.points_world_from_scan(scan, pose).astype(float, copy=False)

        # 3) publish
        self.pub_state.publish(np_to_f32multi(self.x))
        self.pub_obstacles.publish(np_to_f32multi(obstacles_xy))

def main(args=None):
    import rclpy
    rclpy.init(args=args)

    node = SimNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
