# nav_mpc_ros/nodes/sim_node.py
from __future__ import annotations

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import Float32MultiArray

from geometry_msgs.msg import PoseStamped, TwistStamped, TransformStamped
from tf2_ros import TransformBroadcaster

from nav_mpc_ros.ros_paths import add_nav_mpc_repo_to_syspath
add_nav_mpc_repo_to_syspath()

from nav_mpc_ros.ros_conversions import np_to_f32multi, f32multi_to_np

from simulation.simulator import ContinuousSimulator, SimulatorConfig


def yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    """Quaternion for pure yaw rotation (Z axis)."""
    half = 0.5 * yaw
    return (0.0, 0.0, float(np.sin(half)), float(np.cos(half)))


class SimNode(Node):
    """
    High-rate plant simulation node.

    Subscribes:
      /nav_mpc/cmd   Float32MultiArray (nu)

    Publishes:
      /nav_mpc/state Float32MultiArray (nx)  [raw state]
      /nav_mpc/pose  geometry_msgs/PoseStamped
      /nav_mpc/twist geometry_msgs/TwistStamped
      TF: frame_id -> child_frame_id (optional)
    """

    def __init__(self) -> None:
        super().__init__("nav_mpc_sim_node")

        # ---------------- Params ----------------
        self.declare_parameter("dt_sim", 0.002)   # 500 Hz
        self.declare_parameter("substeps", 1)
        self.declare_parameter("x_init", [-1.0, -2.0, float(np.pi / 2), 0.0, 0.0])

        # RViz / frames
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("child_frame_id", "base_link")
        self.declare_parameter("publish_pose", True)
        self.declare_parameter("publish_twist", True)
        self.declare_parameter("publish_tf", True)

        # Map state -> pose
        # Default matches your unicycle state: [x, y, yaw, v, w]
        self.declare_parameter("pose_xy_idx", [0, 1])   # [x_idx, y_idx]
        self.declare_parameter("pose_z_value", 0.0)     # constant z
        self.declare_parameter("yaw_idx", 2)            # yaw in state

        # Map state -> twist
        self.declare_parameter("twist_vx_idx", 3)       # linear velocity
        self.declare_parameter("twist_wz_idx", 4)       # yaw rate

        dt_sim = float(self.get_parameter("dt_sim").value)
        substeps = int(self.get_parameter("substeps").value)

        # ---------------- QoS ----------------
        qos = QoSProfile(depth=10)

        # ---------------- pubs/subs ----------------
        self.pub_state = self.create_publisher(Float32MultiArray, "/nav_mpc/state", qos)
        self.pub_pose = self.create_publisher(PoseStamped, "/nav_mpc/pose", qos)
        self.pub_twist = self.create_publisher(TwistStamped, "/nav_mpc/twist", qos)

        self.sub_cmd = self.create_subscription(Float32MultiArray, "/nav_mpc/cmd", self._cmd_cb, qos)

        # TF broadcaster (optional)
        self.tf_broadcaster = TransformBroadcaster(self)

        # ---------------- system + sim ----------------
        from core.models.unicycle_kinematic_model import UnicycleKinematicModel
        system = UnicycleKinematicModel()

        sim_cfg = SimulatorConfig(dt=dt_sim, method="rk4", substeps=max(1, substeps))
        self.sim = ContinuousSimulator(system, sim_cfg)

        self.x = np.array(self.get_parameter("x_init").value, dtype=float)
        self.u_latest = np.zeros(system.input_dim, dtype=float)

        # Publish initial messages immediately (helps controller + RViz boot)
        self._publish_all()

        self.timer = self.create_timer(dt_sim, self._tick)
        self.get_logger().info(f"SimNode started, running at {1.0/dt_sim:.1f} Hz")

    def _cmd_cb(self, msg: Float32MultiArray) -> None:
        u = f32multi_to_np(msg, dtype=np.float64)
        if u.size > 0:
            self.u_latest = u.copy()

    def _tick(self) -> None:
        self.x = self.sim.step(self.x, self.u_latest)
        self._publish_all()

    def _publish_all(self) -> None:
        # Always publish raw state
        self.pub_state.publish(np_to_f32multi(self.x))

        frame_id = str(self.get_parameter("frame_id").value)
        child_frame_id = str(self.get_parameter("child_frame_id").value)
        stamp = self.get_clock().now().to_msg()

        publish_pose = bool(self.get_parameter("publish_pose").value)
        publish_twist = bool(self.get_parameter("publish_twist").value)
        publish_tf = bool(self.get_parameter("publish_tf").value)

        # --- PoseStamped ---
        if publish_pose or publish_tf:
            pose_xy_idx = list(self.get_parameter("pose_xy_idx").value)
            x_idx, y_idx = int(pose_xy_idx[0]), int(pose_xy_idx[1])
            z_val = float(self.get_parameter("pose_z_value").value)
            yaw_idx = int(self.get_parameter("yaw_idx").value)

            px = float(self.x[x_idx])
            py = float(self.x[y_idx])
            yaw = float(self.x[yaw_idx])
            qx, qy, qz, qw = yaw_to_quat(yaw)

            if publish_pose:
                msg = PoseStamped()
                msg.header.stamp = stamp
                msg.header.frame_id = frame_id
                msg.pose.position.x = px
                msg.pose.position.y = py
                msg.pose.position.z = z_val
                msg.pose.orientation.x = qx
                msg.pose.orientation.y = qy
                msg.pose.orientation.z = qz
                msg.pose.orientation.w = qw
                self.pub_pose.publish(msg)

            # --- TF (map -> base_link) ---
            if publish_tf:
                tfm = TransformStamped()
                tfm.header.stamp = stamp
                tfm.header.frame_id = frame_id
                tfm.child_frame_id = child_frame_id
                tfm.transform.translation.x = px
                tfm.transform.translation.y = py
                tfm.transform.translation.z = z_val
                tfm.transform.rotation.x = qx
                tfm.transform.rotation.y = qy
                tfm.transform.rotation.z = qz
                tfm.transform.rotation.w = qw
                self.tf_broadcaster.sendTransform(tfm)

        # --- TwistStamped ---
        if publish_twist:
            vx_idx = int(self.get_parameter("twist_vx_idx").value)
            wz_idx = int(self.get_parameter("twist_wz_idx").value)

            vx = float(self.x[vx_idx])
            wz = float(self.x[wz_idx])

            msg = TwistStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = child_frame_id  # velocities in robot frame
            msg.twist.linear.x = vx
            msg.twist.linear.y = 0.0
            msg.twist.linear.z = 0.0
            msg.twist.angular.x = 0.0
            msg.twist.angular.y = 0.0
            msg.twist.angular.z = wz
            self.pub_twist.publish(msg)


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
