# nav_mpc/utils/compare_npz.py
from __future__ import annotations

import argparse
import numpy as np

"""
Usage:
python -m utils.compare_npz \
  --main ~/dev_ws/src/nav_mpc/results/debug/nav_mpc_main_step0000.npz \
  --ros  ~/dev_ws/src/nav_mpc/results/debug/nav_mpc_ros_step0000.npz
"""

def compare_npz(main_path: str, ros_path: str, atol: float = 0.0, rtol: float = 0.0) -> int:
    a = np.load(main_path)
    b = np.load(ros_path)

    def compare_key(k: str) -> str:
        aa = a[k]
        bb = b[k]
        if aa.shape != bb.shape:
            return f"{k}: shape {aa.shape} != {bb.shape}"
        if aa.size == 0:
            return f"{k}: empty"
        ok = np.allclose(aa, bb, atol=atol, rtol=rtol, equal_nan=True)
        mad = float(np.nanmax(np.abs(aa - bb)))
        return f"{k}: same={ok}, max|diff|={mad:.6g}"

    keys = sorted(set(a.files) | set(b.files))
    all_ok = True
    for k in keys:
        if k not in a.files:
            print(f"{k}: only in ROS")
            all_ok = False
            continue
        if k not in b.files:
            print(f"{k}: only in main")
            all_ok = False
            continue
        line = compare_key(k)
        print(line)
        if "same=True" not in line and "empty" not in line:
            all_ok = False

    return 0 if all_ok else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--main", required=True, help="Path to nav_mpc_main_stepXXXX.npz")
    p.add_argument("--ros", required=True, help="Path to nav_mpc_ros_stepXXXX.npz")
    p.add_argument("--atol", type=float, default=0.0)
    p.add_argument("--rtol", type=float, default=0.0)
    args = p.parse_args()
    raise SystemExit(compare_npz(args.main, args.ros, atol=args.atol, rtol=args.rtol))


if __name__ == "__main__":
    main()


