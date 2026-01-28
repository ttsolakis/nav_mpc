# nav_mpc_ros/nodes/lidar_node.py
from __future__ import annotations

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import Float32MultiArray

from nav_mpc_ros.ros_paths import add_nav_mpc_repo_to_syspath
add_nav_mpc_repo_to_syspath()

from nav_mpc_ros.ros_conversions import np_to_f32multi, f32multi_to_np

from simulation.environment.occupancy_map import OccupancyMapConfig, OccupancyMap2D
from simulation.lidar import LidarSimulator2D, LidarConfig


class LidarNode(Node):
    """
    Lidar sensor node (slow rate, realistic).

    Subscribes:
      /nav_mpc/state         Float32MultiArray (nx)

    Publishes:
      /nav_mpc/obstacles_xy  Float32MultiArray flattened [x0,y0,x1,y1,...]
    """

    def __init__(self) -> None:
        super().__init__("nav_mpc_lidar_node")

        # ---------------- Params ----------------
        self.declare_parameter("dt_lidar", 0.1)  # 10 Hz
        self.declare_parameter("map_path", "map.png")
        self.declare_parameter("world_width_m", 5.0)
        self.declare_parameter("occupied_threshold", 127)
        self.declare_parameter("invert_map", False)

        self.declare_parameter("lidar_range_max", 8.0)
        self.declare_parameter("lidar_angle_increment_deg", 0.72)

        dt_lidar = float(self.get_parameter("dt_lidar").value)

        # ---------------- QoS ----------------
        qos = QoSProfile(depth=10)

        # ---------------- pubs/subs ----------------
        self.pub_obstacles = self.create_publisher(Float32MultiArray, "/nav_mpc/obstacles_xy", qos)
        self.sub_state = self.create_subscription(Float32MultiArray, "/nav_mpc/state", self._state_cb, qos)

        # ---------------- map + lidar ----------------
        occ_cfg = OccupancyMapConfig(
            map_path=str(self.get_parameter("map_path").value),
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

        self.x_latest: np.ndarray | None = None

        # publish empty initially
        self.pub_obstacles.publish(np_to_f32multi(np.zeros((0, 2), dtype=float)))

        self.timer = self.create_timer(dt_lidar, self._tick)
        self.get_logger().info(f"LidarNode started: dt_lidar={dt_lidar}s (~{1.0/dt_lidar:.1f} Hz)")

    def _state_cb(self, msg: Float32MultiArray) -> None:
        x = f32multi_to_np(msg, dtype=np.float64)
        if x.size > 0:
            self.x_latest = x

    def _tick(self) -> None:
        if self.x_latest is None:
            return

        x = self.x_latest
        pose = np.array([x[0], x[1], x[2]], dtype=float)

        scan = self.lidar.scan(pose)
        obstacles_xy = self.lidar.points_world_from_scan(scan, pose).astype(float, copy=False)

        self.pub_obstacles.publish(np_to_f32multi(obstacles_xy))


def main(args=None):
    rclpy.init(args=args)
    node = LidarNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
