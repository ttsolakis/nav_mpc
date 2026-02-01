# nav_mpc_ros/nodes/nav_mpc_node.py
from __future__ import annotations

import os
import time
import numpy as np
import osqp

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from utils.debug_dump import get_default_debug_dir, dump_npz

from nav_mpc_ros.ros_paths import add_nav_mpc_repo_to_syspath
add_nav_mpc_repo_to_syspath()

from nav_mpc_ros.ros_conversions import np_to_f32multi, f32multi_to_np

# nav_mpc imports
from core.mpc2qp import build_qp, make_workspace, update_qp, solve_qp
from simulation.path_following import make_reference_builder


class NavMpcNode(Node):
    """
    MPC controller node.

    Subscribes:
      /nav_mpc/state        Float32MultiArray
      /nav_mpc/obstacles_xy Float32MultiArray  (flattened x,y)
      /nav_mpc/path_xy      Float32MultiArray  (flattened x,y)  (latched)
      /nav_mpc/map          nav_msgs/OccupancyGrid (latched) (for bbox only)

    Publishes:
      /nav_mpc/cmd          Float32MultiArray

      /nav_mpc/pred_markers MarkerArray
        - line strip of predicted horizon (sampled)
        - sphere list of predicted points (sampled)

      /nav_mpc/col_poly_markers MarkerArray
        - line strips of halfspace intersection polygons (sampled along horizon)
    """

    def __init__(self) -> None:
        super().__init__("nav_mpc_controller_node")

        # ---------------- Params ----------------
        self.declare_parameter("dt_mpc", 0.1)
        self.declare_parameter("N", 25)
        self.declare_parameter("embedded", True)
        self.declare_parameter("debugging", False)

        self.declare_parameter("x_goal", [2.0, 2.0, 0.0, 0.0, 0.0])
        self.declare_parameter("velocity_ref", 0.5)

        # Debug dump
        self.step_idx = 0
        repo_root = os.environ.get("NAV_MPC_ROOT", os.path.expanduser("~/dev_ws/src/nav_mpc"))
        self.declare_parameter("debug_dump_dir", get_default_debug_dir(repo_root))

        # Visualization params
        self.declare_parameter("frame_id", "map")

        self.declare_parameter("publish_pred_markers", True)
        self.declare_parameter("pred_num_samples", 5)
        self.declare_parameter("pred_markers_topic", "/nav_mpc/pred_markers")

        self.declare_parameter("publish_col_poly_markers", True)
        self.declare_parameter("col_poly_num_samples", 5)
        self.declare_parameter("col_poly_topic", "/nav_mpc/col_poly_markers")

        # ---------------- QoS / pubs ----------------
        self.pub_cmd = self.create_publisher(Float32MultiArray, "/nav_mpc/cmd", 10)
        self.pub_pred = self.create_publisher(
            MarkerArray, str(self.get_parameter("pred_markers_topic").value), 10
        )
        self.pub_col = self.create_publisher(
            MarkerArray, str(self.get_parameter("col_poly_topic").value), 10
        )

        # ---------------- subs ----------------
        self.sub_state = self.create_subscription(Float32MultiArray, "/nav_mpc/state", self._state_cb, 10)
        self.sub_obstacles = self.create_subscription(Float32MultiArray, "/nav_mpc/obstacles_xy", self._obstacles_cb, 10)
        self.sub_path = self.create_subscription(Float32MultiArray, "/nav_mpc/path_xy", self._path_cb, 1)
        self.sub_map = self.create_subscription(OccupancyGrid, "/nav_mpc/map", self._map_cb, 1)

        # caches
        self.x_latest: np.ndarray | None = None
        self.obstacles_xy_latest: np.ndarray | None = None
        self.path_xy: np.ndarray | None = None

        self._bbox_map: tuple[float, float, float, float] | None = None  # (xmin, xmax, ymin, ymax)

        # collision polytope caches (from update_qp)
        self._last_A_xy: np.ndarray | None = None
        self._last_b_xy: np.ndarray | None = None

        # wait-logging throttling
        self._last_wait_log_s = 0.0
        self._wait_log_period_s = 2.0

        # ---------------- Setup MPC (once) ----------------
        dt = float(self.get_parameter("dt_mpc").value)
        N = int(self.get_parameter("N").value)

        from core.problem_setup import setup_path_tracking_unicycle
        self.problem_name, self.system, self.objective, self.constraints, self.collision, _ = (
            setup_path_tracking_unicycle.setup_problem()
        )

        self.get_logger().info("Setting up NavMpcNode...")

        self.qp = build_qp(
            system=self.system,
            objective=self.objective,
            constraints=self.constraints,
            N=N,
            dt=dt,
            collision=self.collision,
        )

        self.nx = self.system.state_dim
        self.nu = self.system.input_dim
        self.nc_sys = self.qp.nc_sys

        # OSQP
        self.prob = osqp.OSQP()
        self.prob.setup(
            self.qp.P_init, self.qp.q_init, self.qp.A_init, self.qp.l_init, self.qp.u_init,
            warm_starting=True, verbose=False
        )
        self.ws = make_workspace(
            N=N, nx=self.nx, nu=self.nu,
            nc_sys=self.nc_sys, nc_col=self.qp.nc_col,
            A_data=self.qp.A_init.data,
            l_init=self.qp.l_init, u_init=self.qp.u_init,
            P_data=self.qp.P_init.data,
            q_init=self.qp.q_init,
        )

        # warm-start trajectories
        self.X = np.zeros((N + 1, self.nx), dtype=float)
        self.U = np.zeros((N, self.nu), dtype=float)

        # reference builder
        x_goal = np.array(self.get_parameter("x_goal").value, dtype=float)
        v_ref = float(self.get_parameter("velocity_ref").value)
        self.ref_builder = make_reference_builder(
            pos_idx=(0, 1), phi_idx=2, v_idx=3,
            x_goal=x_goal, v_ref=v_ref,
            goal_indices=[4],
            window=40, max_lookahead_points=N,
            stop_radius=0.25, stop_ramp=0.50,
        )

        self.embedded = bool(self.get_parameter("embedded").value)
        self.debugging = bool(self.get_parameter("debugging").value)

        self.timer = self.create_timer(dt, self._tick)
        self.get_logger().info(f"NavMpcNode started, running at {1.0/dt:.1f} Hz in embedded={self.embedded} mode.")

    # ---------------- Callbacks ----------------
    def _state_cb(self, msg: Float32MultiArray) -> None:
        x = f32multi_to_np(msg, dtype=np.float64)
        if x.size > 0:
            self.x_latest = x

    def _obstacles_cb(self, msg: Float32MultiArray) -> None:
        flat = f32multi_to_np(msg, dtype=np.float64)
        if flat.size == 0:
            self.obstacles_xy_latest = flat.reshape(0, 2)
        else:
            self.obstacles_xy_latest = flat.reshape(-1, 2)

    def _path_cb(self, msg: Float32MultiArray) -> None:
        flat = f32multi_to_np(msg, dtype=np.float64)
        if flat.size == 0:
            self.path_xy = None
        else:
            self.path_xy = flat.reshape(-1, 2)

    def _map_cb(self, msg: OccupancyGrid) -> None:
        # bbox in map frame (for halfspace intersection clipping)
        res = float(msg.info.resolution)
        w = int(msg.info.width)
        h = int(msg.info.height)
        ox = float(msg.info.origin.position.x)
        oy = float(msg.info.origin.position.y)

        xmin = ox
        ymin = oy
        xmax = ox + w * res
        ymax = oy + h * res
        self._bbox_map = (xmin, xmax, ymin, ymax)

    # ---------------- Main loop ----------------
    def _tick(self) -> None:
        if self.x_latest is None or self.obstacles_xy_latest is None or self.path_xy is None:
            now = time.monotonic()
            if now - self._last_wait_log_s >= self._wait_log_period_s:
                self._last_wait_log_s = now
                self.get_logger().info(
                    "Waiting for inputs: "
                    f"state={'OK' if self.x_latest is not None else '---'}, "
                    f"obstacles={'OK' if self.obstacles_xy_latest is not None else '---'}, "
                    f"path={'OK' if self.path_xy is not None else '---'}"
                )
            return

        x = self.x_latest
        obstacles_xy = self.obstacles_xy_latest
        global_path = self.path_xy

        N = int(self.get_parameter("N").value)
        dt = float(self.get_parameter("dt_mpc").value)

        # Ensure warm-start is consistent with current x on first tick
        if self.step_idx == 0:
            self.X = np.tile(x.reshape(1, -1), (N + 1, 1))
            self.U = np.zeros((N, self.nu), dtype=float)

        # Build reference
        Xref_seq = self.ref_builder(global_path=global_path, x=x, N=N)

        # Optional debug dump on first iter
        if self.step_idx == 0 and self.debugging:
            dump_npz(dump_dir=str(self.get_parameter("debug_dump_dir").value), tag="ros", step_idx=self.step_idx, dt=dt, N=N, x=x, X=self.X, U=self.U, Xref_seq=Xref_seq, obstacles_xy=obstacles_xy, global_path=global_path)

        # QP update
        t0 = time.perf_counter()
        A_xy, b_xy = update_qp(self.prob, x, self.X, self.U, self.qp, self.ws, Xref_seq, obstacles_xy=obstacles_xy)
        t1 = time.perf_counter()

        self._last_A_xy = None if A_xy is None else np.asarray(A_xy, dtype=float)
        self._last_b_xy = None if b_xy is None else np.asarray(b_xy, dtype=float)

        # Solve with time limit
        time_limit = dt - (t1 - t0)
        if self.embedded and time_limit <= 1e-6:
            # deadline miss: publish previous u0 (or zeros)
            u0 = self.U[0].copy() if self.step_idx > 0 else np.zeros(self.nu)
            self.pub_cmd.publish(np_to_f32multi(u0))
        else:
            X, U, u0 = solve_qp(self.prob, self.nx, self.nu, N, self.embedded, time_limit, self.step_idx, debugging=self.debugging, x=x)
            self.X, self.U = X, U
            self.pub_cmd.publish(np_to_f32multi(u0))

        # Publish visuals every cycle (uses latest self.X and last A/b)
        self._publish_predicted_markers()
        self._publish_collision_poly_markers()

        self.step_idx += 1

    # ---------------- Visualization publishers ----------------
    def _publish_predicted_markers(self) -> None:
        if not bool(self.get_parameter("publish_pred_markers").value):
            return
        if self.X is None or self.X.size == 0:
            return

        frame_id = str(self.get_parameter("frame_id").value)
        stamp = self.get_clock().now().to_msg()

        N = int(self.get_parameter("N").value)
        K = int(self.get_parameter("pred_num_samples").value)
        K = max(2, min(K, N + 1))

        idx = np.linspace(0, N, K).round().astype(int)
        xy = self.X[idx, :2].copy()

        arr = MarkerArray()

        # Clear previous markers
        m_clear = Marker()
        m_clear.header.frame_id = frame_id
        m_clear.header.stamp = stamp
        m_clear.ns = "pred"
        m_clear.id = 0
        m_clear.action = Marker.DELETEALL
        arr.markers.append(m_clear)

        # Line strip
        m_line = Marker()
        m_line.header.frame_id = frame_id
        m_line.header.stamp = stamp
        m_line.ns = "pred"
        m_line.id = 1
        m_line.type = Marker.LINE_STRIP
        m_line.action = Marker.ADD
        m_line.pose.orientation.w = 1.0
        m_line.scale.x = 0.03
        m_line.color.r = 0.1
        m_line.color.g = 0.6
        m_line.color.b = 1.0
        m_line.color.a = 0.9
        m_line.points = self._points_from_xy(xy)
        arr.markers.append(m_line)

        # Sample points
        m_pts = Marker()
        m_pts.header.frame_id = frame_id
        m_pts.header.stamp = stamp
        m_pts.ns = "pred"
        m_pts.id = 2
        m_pts.type = Marker.SPHERE_LIST
        m_pts.action = Marker.ADD
        m_pts.pose.orientation.w = 1.0
        m_pts.scale.x = 0.08
        m_pts.scale.y = 0.08
        m_pts.scale.z = 0.08
        m_pts.color.r = 1.0
        m_pts.color.g = 0.7
        m_pts.color.b = 0.1
        m_pts.color.a = 0.95
        m_pts.points = self._points_from_xy(xy)
        arr.markers.append(m_pts)

        self.pub_pred.publish(arr)

    def _publish_collision_poly_markers(self) -> None:
        if not bool(self.get_parameter("publish_col_poly_markers").value):
            return
        if self._last_A_xy is None or self._last_b_xy is None:
            return
        if self._bbox_map is None:
            return

        Axy = np.asarray(self._last_A_xy, dtype=float)
        Bxy = np.asarray(self._last_b_xy, dtype=float)

        # Expect (Nh, M, 2) and (Nh, M)
        if Axy.ndim != 3 or Axy.shape[2] != 2:
            return
        if Bxy.ndim != 2 or Bxy.shape[0] != Axy.shape[0] or Bxy.shape[1] != Axy.shape[1]:
            return

        frame_id = str(self.get_parameter("frame_id").value)
        stamp = self.get_clock().now().to_msg()

        Nh = Axy.shape[0]
        if Nh <= 0:
            return

        K = int(self.get_parameter("col_poly_num_samples").value)
        K = max(1, min(K, Nh))
        idx = np.linspace(0, Nh - 1, K).round().astype(int)

        arr = MarkerArray()

        # Clear previous markers
        m_clear = Marker()
        m_clear.header.frame_id = frame_id
        m_clear.header.stamp = stamp
        m_clear.ns = "col_poly"
        m_clear.id = 0
        m_clear.action = Marker.DELETEALL
        arr.markers.append(m_clear)

        for k, ii in enumerate(idx):
            Aii = Axy[ii, :, :]  # (M,2)
            bii = Bxy[ii, :]     # (M,)
            verts = self._halfspace_intersection_polygon_Axb(Aii, bii, bbox=self._bbox_map)

            m = Marker()
            m.header.frame_id = frame_id
            m.header.stamp = stamp
            m.ns = "col_poly"
            m.id = int(k + 1)  # 0 used by DELETEALL
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = 0.03

            # fade alpha with horizon sample index
            a = 0.85 * (1.0 - 0.7 * (k / max(1, (len(idx) - 1))))
            m.color.r = 0.0
            m.color.g = 1.0
            m.color.b = 0.2
            m.color.a = float(a)

            if verts.shape[0] >= 3:
                closed = np.vstack([verts, verts[0:1, :]])
                m.points = self._points_from_xy(closed)
            else:
                m.points = []

            arr.markers.append(m)

        self.pub_col.publish(arr)

    # ---------------- Geometry helpers ----------------
    @staticmethod
    def _points_from_xy(xy: np.ndarray) -> list[Point]:
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        pts: list[Point] = []
        for x, y in xy:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.0
            pts.append(p)
        return pts

    @staticmethod
    def _halfspace_intersection_polygon_Axb(
        A: np.ndarray, b: np.ndarray, bbox: tuple[float, float, float, float]
    ) -> np.ndarray:
        """
        Compute vertices of intersection {p | A p <= b} clipped by bbox.

        A: (M,2), b: (M,)
        bbox = (xmin, xmax, ymin, ymax)
        Returns vertices (K,2) CCW, or empty (0,2).
        """
        xmin, xmax, ymin, ymax = map(float, bbox)

        A = np.asarray(A, dtype=float).reshape(-1, 2)
        b = np.asarray(b, dtype=float).reshape(-1)
        if A.shape[0] != b.size:
            return np.empty((0, 2), dtype=float)

        # drop degenerate rows
        row_norm = np.linalg.norm(A, axis=1)
        keep = row_norm > 1e-12
        A = A[keep]
        b = b[keep]
        if A.shape[0] == 0:
            return np.empty((0, 2), dtype=float)

        # bbox halfspaces
        Ab = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)
        bb = np.array([xmax, -xmin, ymax, -ymin], dtype=float)

        Aall = np.vstack([A, Ab])
        ball = np.concatenate([b, bb])

        pts: list[np.ndarray] = []
        M = Aall.shape[0]

        for i in range(M):
            ai = Aall[i]
            bi = ball[i]
            for j in range(i + 1, M):
                aj = Aall[j]
                bj = ball[j]

                Mat = np.array([ai, aj], dtype=float)
                det = Mat[0, 0] * Mat[1, 1] - Mat[0, 1] * Mat[1, 0]
                if abs(det) < 1e-12:
                    continue

                p = np.linalg.solve(Mat, np.array([bi, bj], dtype=float))
                if np.all(Aall @ p <= ball + 1e-9):
                    pts.append(p)

        if not pts:
            return np.empty((0, 2), dtype=float)

        P = np.vstack(pts)

        # sort CCW
        c = P.mean(axis=0)
        ang = np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0])
        order = np.argsort(ang)
        P = P[order]

        # remove near-duplicates
        dedup = [0]
        for k in range(1, P.shape[0]):
            if np.linalg.norm(P[k] - P[dedup[-1]]) > 1e-6:
                dedup.append(k)
        P = P[dedup]

        if P.shape[0] < 3:
            return np.empty((0, 2), dtype=float)

        return P


def main(args=None):
    rclpy.init(args=args)
    node = NavMpcNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
