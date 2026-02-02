# nav_mpc_ros/nodes/path_node.py

import os
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray

from nav_msgs.msg import Path, OccupancyGrid, MapMetaData
from geometry_msgs.msg import PoseStamped, Pose

from PIL import Image  # <-- IMPORTANT

from nav_mpc_ros.ros_paths import add_nav_mpc_repo_to_syspath
add_nav_mpc_repo_to_syspath()

from nav_mpc_ros.ros_conversions import np_to_f32multi

from simulation.environment.occupancy_map import OccupancyMapConfig, OccupancyMap2D
from simulation.path_following import (
    RRTStarConfig,
    rrt_star_plan,
    smooth_and_resample_path,
)


def _resolve_map_path(map_path: str) -> str:
    """
    Resolve map path robustly.

    - If absolute -> use as is
    - Else try:
        1) cwd/map_path
        2) nav_mpc repo: <repo_root>/simulation/environment/maps/map_path
    """
    if os.path.isabs(map_path):
        return map_path

    # 1) relative to current working directory
    cand = os.path.abspath(map_path)
    if os.path.exists(cand):
        return cand

    # 2) relative to nav_mpc repo layout
    repo_root = os.path.expanduser("~/dev_ws/src/nav_mpc")
    cand2 = os.path.join(repo_root, "simulation", "environment", "maps", map_path)
    if os.path.exists(cand2):
        return cand2

    # fallback (lets Image.open raise a useful error)
    return cand


def png_to_occupancygrid(
    map_path: str,
    world_width_m: float,
    occupied_threshold: int,
    invert: bool,
    frame_id: str,
    stamp,
) -> OccupancyGrid:
    """
    Convert a grayscale PNG into nav_msgs/OccupancyGrid.

    Conventions:
      - PNG pixel (0,0) is top-left
      - OccupancyGrid data starts at map origin (bottom-left), row-major
      - We set map origin at (-world_width/2, -world_width/2)
      - resolution = world_width_m / image_width_px  (assumes square world)
      - Occupied if pixel < occupied_threshold (dark = occupied), optionally inverted
    """
    map_path = _resolve_map_path(map_path)

    img = Image.open(map_path).convert("L")      # grayscale
    arr = np.array(img, dtype=np.uint8)          # (H,W)

    h, w = arr.shape
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid map image size: {arr.shape}")

    resolution = float(world_width_m) / float(w)

    # dark pixels = occupied (typical)
    occ = arr < np.uint8(occupied_threshold)
    if invert:
        occ = ~occ

    # flip vertically: image top-left -> map bottom-left
    occ = np.flipud(occ)

    # build occupancy values: 0 free, 100 occupied
    data = np.where(occ, 100, 0).astype(np.int8).reshape(-1).tolist()

    msg = OccupancyGrid()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id

    info = MapMetaData()
    info.map_load_time = stamp
    info.resolution = resolution
    info.width = int(w)
    info.height = int(h)

    origin = Pose()
    origin.position.x = -0.5 * float(world_width_m)
    origin.position.y = -0.5 * float(world_width_m)
    origin.position.z = 0.0
    origin.orientation.w = 1.0
    info.origin = origin

    msg.info = info
    msg.data = data
    return msg


class PathNode(Node):
    """
    Global path node.

    Publishes (latched):
      /nav_mpc/path_xy   Float32MultiArray (N,2)
      /nav_mpc/path      nav_msgs/Path
      /nav_mpc/map       nav_msgs/OccupancyGrid
    """

    def __init__(self) -> None:
        super().__init__("nav_mpc_path_node")

        # ---------------- Params ----------------
        self.declare_parameter("map_path", "map.png")
        self.declare_parameter("world_width_m", 5.0)
        self.declare_parameter("occupied_threshold", 127)
        self.declare_parameter("invert_map", False)

        self.declare_parameter("x_init", [-1.0, -2.0, float(np.pi / 2), 0.0, 0.0])
        self.declare_parameter("x_goal", [2.0, 2.0, 0.0, 0.0, 0.0])

        self.declare_parameter("rrt_max_iters", 6000)
        self.declare_parameter("rrt_step_size", 0.10)
        self.declare_parameter("rrt_neighbor_radius", 0.30)
        self.declare_parameter("rrt_goal_sample_rate", 0.10)
        self.declare_parameter("rrt_collision_check_step", 0.02)
        self.declare_parameter("inflation_radius_m", 0.25)

        self.declare_parameter("frame_id", "map")
        self.declare_parameter("republish_period_s", 10.0)

        # ---------------- QoS (latched) ----------------
        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # Publishers
        self.pub_path_xy = self.create_publisher(Float32MultiArray, "/nav_mpc/path_xy", latched_qos)
        self.pub_path_rviz = self.create_publisher(Path, "/nav_mpc/path", latched_qos)
        self.pub_map = self.create_publisher(OccupancyGrid, "/nav_mpc/map", latched_qos)

        # ---------------- Compute path once ----------------
        t0 = time.perf_counter()

        map_path = str(self.get_parameter("map_path").value)
        world_width_m = float(self.get_parameter("world_width_m").value)
        occupied_threshold = int(self.get_parameter("occupied_threshold").value)
        invert = bool(self.get_parameter("invert_map").value)

        occ_cfg = OccupancyMapConfig(
            map_path=_resolve_map_path(map_path),
            world_width_m=world_width_m,
            occupied_threshold=occupied_threshold,
            invert=invert,
        )
        self.occ_map = OccupancyMap2D.from_png(occ_cfg)

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
            occ_map=self.occ_map,
            start_xy=x_init_xy,
            goal_xy=x_goal_xy,
            inflation_radius_m=float(self.get_parameter("inflation_radius_m").value),
            cfg=rrt_cfg,
        )
        path = smooth_and_resample_path(path, ds=0.05, smoothing=0.01, k=3)
        self._path_xy = path[:, :2].copy()

        t1 = time.perf_counter()
        self.get_logger().info(
            f"PathNode started, path computed in {t1 - t0:.2f}s, "
            f"publishing latched /nav_mpc/path_xy, /nav_mpc/path, /nav_mpc/map"
        )

        # Publish once (latched)
        self._publish_all()

        # Optional republish
        republish_period = float(self.get_parameter("republish_period_s").value)
        if republish_period > 0.0:
            self.create_timer(republish_period, self._publish_all)

    def _publish_all(self) -> None:
        frame_id = str(self.get_parameter("frame_id").value)
        stamp = self.get_clock().now().to_msg()

        # 1) Float32MultiArray
        self.pub_path_xy.publish(np_to_f32multi(self._path_xy))

        # 2) nav_msgs/Path
        msg_path = Path()
        msg_path.header.frame_id = frame_id
        msg_path.header.stamp = stamp

        poses: list[PoseStamped] = []
        for x, y in self._path_xy:
            ps = PoseStamped()
            ps.header.frame_id = frame_id
            ps.header.stamp = stamp
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            poses.append(ps)

        msg_path.poses = poses
        self.pub_path_rviz.publish(msg_path)

        # 3) OccupancyGrid (from PNG)
        map_path = str(self.get_parameter("map_path").value)
        world_width_m = float(self.get_parameter("world_width_m").value)
        occupied_threshold = int(self.get_parameter("occupied_threshold").value)
        invert = bool(self.get_parameter("invert_map").value)

        msg_map = png_to_occupancygrid(
            map_path=map_path,
            world_width_m=world_width_m,
            occupied_threshold=occupied_threshold,
            invert=invert,
            frame_id=frame_id,
            stamp=stamp,
        )
        self.pub_map.publish(msg_map)


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