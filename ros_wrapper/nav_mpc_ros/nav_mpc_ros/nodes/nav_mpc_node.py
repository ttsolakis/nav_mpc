# nav_mpc_ros/nodes/nav_mpc_node.py
from __future__ import annotations

import time
import numpy as np
import osqp

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from nav_mpc_ros.ros_paths import add_nav_mpc_repo_to_syspath
add_nav_mpc_repo_to_syspath()

from nav_mpc_ros.ros_conversions import np_to_f32multi, f32multi_to_np

# nav_mpc imports (same as main.py)
from core.mpc2qp import build_qp, make_workspace, update_qp, solve_qp
from simulation.path_following import make_reference_builder

class NavMpcNode(Node):
    def __init__(self) -> None:
        super().__init__("nav_mpc_controller_node")

        # ---------------- Params ----------------
        self.declare_parameter("dt_mpc", 0.1)
        self.declare_parameter("N", 25)
        self.declare_parameter("embedded", False)
        self.declare_parameter("debugging", True)

        # Needed for ref_builder
        self.declare_parameter("x_goal", [2.0, 2.0, 0.0, 0.0, 0.0])
        self.declare_parameter("velocity_ref", 0.5)

        # ---------------- pubs/subs ----------------
        self.pub_cmd = self.create_publisher(Float32MultiArray, "/nav_mpc/cmd", 10)

        self.sub_state = self.create_subscription(
            Float32MultiArray, "/nav_mpc/state", self._state_cb, 10
        )
        self.sub_obstacles = self.create_subscription(
            Float32MultiArray, "/nav_mpc/obstacles_xy", self._obstacles_cb, 10
        )
        self.sub_path = self.create_subscription(
            Float32MultiArray, "/nav_mpc/path_xy", self._path_cb, 1
        )

        # caches
        self.x_latest: np.ndarray | None = None
        self.obstacles_xy_latest: np.ndarray | None = None
        self.path_xy: np.ndarray | None = None

        # ---------------- Setup MPC (once) ----------------
        dt = float(self.get_parameter("dt_mpc").value)
        N = int(self.get_parameter("N").value)

        from core.problem_setup import setup_path_tracking_unicycle
        self.problem_name, self.system, self.objective, self.constraints, self.collision, _ = (
            setup_path_tracking_unicycle.setup_problem()
        )
        
        self.get_logger().info(f"Setting up nav_mpc...")

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

        # ref builder
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

        self.get_logger().info(f"nav_mpc setup complete.")

    def _state_cb(self, msg: Float32MultiArray) -> None:
        x = f32multi_to_np(msg, dtype=np.float64)
        if x.size > 0:
            self.x_latest = x

    def _obstacles_cb(self, msg: Float32MultiArray) -> None:
        # flattened [x0,y0,x1,y1,...]
        flat = f32multi_to_np(msg, dtype=np.float64)
        if flat.size == 0:
            self.obstacles_xy_latest = flat.reshape(0, 2)
            return
        self.obstacles_xy_latest = flat.reshape(-1, 2)

    def _path_cb(self, msg: Float32MultiArray) -> None:
        flat = f32multi_to_np(msg, dtype=np.float64)
        if flat.size == 0:
            self.path_xy = None
            return
        self.path_xy = flat.reshape(-1, 2)

    def _tick(self) -> None:
        if self.x_latest is None or self.obstacles_xy_latest is None or self.path_xy is None:
            return  # wait until we have all inputs

        self.get_logger().info(f"Solving QP...")

        x = self.x_latest
        obstacles_xy = self.obstacles_xy_latest
        global_path = self.path_xy

        N = int(self.get_parameter("N").value)
        dt = float(self.get_parameter("dt_mpc").value)

        # Build ref from latest state
        Xref_seq = self.ref_builder(global_path=global_path, x=x, N=N)

        # QP update
        obstacles_xy = None
        t0 = time.perf_counter()
        A_xy, b_xy = update_qp(self.prob, x, self.X, self.U, self.qp, self.ws, Xref_seq, obstacles_xy=obstacles_xy)
        t1 = time.perf_counter()

        # Solve with time limit
        self.get_logger().info(f"self.embedded: {self.embedded}")
        self.get_logger().info(f"obstacles_xy: {obstacles_xy}")
        time_limit = max(0.0, dt - (t1 - t0))
        X, U, u0 = solve_qp(
            self.prob, self.nx, self.nu, N,
            self.embedded, time_limit,
            0,  # iteration index not needed here
            debugging=self.debugging,
            x=x,
        )
        self.X, self.U = X, U

        # publish cmd
        self.pub_cmd.publish(np_to_f32multi(u0))

def main(args=None):
    import rclpy
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
