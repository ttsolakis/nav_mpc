# nav_mpc_ros/nodes/sim_node.py
from __future__ import annotations

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import Float32MultiArray

from nav_mpc_ros.ros_paths import add_nav_mpc_repo_to_syspath
add_nav_mpc_repo_to_syspath()

from nav_mpc_ros.ros_conversions import np_to_f32multi, f32multi_to_np

from simulation.simulator import ContinuousSimulator, SimulatorConfig


class SimNode(Node):
    """
    High-rate plant simulation node.

    Subscribes:
      /nav_mpc/cmd   Float32MultiArray (nu)

    Publishes:
      /nav_mpc/state Float32MultiArray (nx)
    """

    def __init__(self) -> None:
        super().__init__("nav_mpc_sim_node")

        # ---------------- Params ----------------
        self.declare_parameter("dt_sim", 0.002)   # 500 Hz
        self.declare_parameter("substeps", 1)
        self.declare_parameter("x_init", [-1.0, -2.0, float(np.pi / 2), 0.0, 0.0])

        dt_sim = float(self.get_parameter("dt_sim").value)
        substeps = int(self.get_parameter("substeps").value)

        # ---------------- QoS ----------------
        qos = QoSProfile(depth=10)

        # ---------------- pubs/subs ----------------
        self.pub_state = self.create_publisher(Float32MultiArray, "/nav_mpc/state", qos)
        self.sub_cmd = self.create_subscription(Float32MultiArray, "/nav_mpc/cmd", self._cmd_cb, qos)

        # ---------------- system + sim ----------------
        # IMPORTANT: do NOT import setup_path_tracking_unicycle here (pulls collision/numba).
        from core.models.unicycle_kinematic_model import UnicycleKinematicModel
        system = UnicycleKinematicModel()

        sim_cfg = SimulatorConfig(dt=dt_sim, method="rk4", substeps=max(1, substeps))
        self.sim = ContinuousSimulator(system, sim_cfg)

        self.x = np.array(self.get_parameter("x_init").value, dtype=float)
        self.u_latest = np.zeros(system.input_dim, dtype=float)

        # Publish initial state immediately (helps controller boot)
        self.pub_state.publish(np_to_f32multi(self.x))

        self.timer = self.create_timer(dt_sim, self._tick)
        self.get_logger().info(f"SimNode started, running at {1.0/dt_sim:.1f} Hz")

    def _cmd_cb(self, msg: Float32MultiArray) -> None:
        u = f32multi_to_np(msg, dtype=np.float64)
        if u.size > 0:
            self.u_latest = u.copy()

    def _tick(self) -> None:
        self.x = self.sim.step(self.x, self.u_latest)
        self.pub_state.publish(np_to_f32multi(self.x))


def main(args=None):
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
