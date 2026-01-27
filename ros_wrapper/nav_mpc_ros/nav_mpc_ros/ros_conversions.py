# nav_mpc_ros/ros_conversions.py
from __future__ import annotations

import numpy as np
from std_msgs.msg import Float32MultiArray

def np_to_f32multi(arr: np.ndarray) -> Float32MultiArray:
    msg = Float32MultiArray()
    msg.data = np.asarray(arr, dtype=np.float32).ravel().tolist()
    return msg

def f32multi_to_np(msg: Float32MultiArray, dtype=np.float64) -> np.ndarray:
    return np.asarray(msg.data, dtype=dtype)
