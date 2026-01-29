"""
Compare two nav_mpc debug .npz dumps (e.g. main vs ROS) and print diffs.

Usage:

  python -m utils.debug_compare_npz \
  /home/tasos/dev_ws/src/results/debug/nav_mpc_main_step0000.npz \
  /home/tasos/dev_ws/src/nav_mpc/results/debug/nav_mpc_ros_step0000.npz

"""

from __future__ import annotations

import argparse
import numpy as np


def _fmt_shape(a: np.ndarray) -> str:
    return "x".join(map(str, a.shape)) if hasattr(a, "shape") else "?"


def compare_npz(a_path: str, b_path: str, atol: float = 0.0, rtol: float = 0.0) -> int:
    a = np.load(a_path, allow_pickle=False)
    b = np.load(b_path, allow_pickle=False)

    keys = sorted(set(a.files) | set(b.files))

    exit_code = 0
    for k in keys:
        if k not in a.files:
            print(f"{k}: only in B")
            exit_code = 1
            continue
        if k not in b.files:
            print(f"{k}: only in A")
            exit_code = 1
            continue

        aa = a[k]
        bb = b[k]

        if aa.shape != bb.shape:
            print(f"{k}: shape A={aa.shape} != B={bb.shape}")
            exit_code = 1
            continue

        if aa.size == 0:
            print(f"{k}: empty")
            continue

        # Handle non-numeric arrays gracefully
        if not (np.issubdtype(aa.dtype, np.number) and np.issubdtype(bb.dtype, np.number)):
            same = np.array_equal(aa, bb)
            print(f"{k}: non-numeric, equal={same}")
            if not same:
                exit_code = 1
            continue

        same = np.allclose(aa, bb, atol=atol, rtol=rtol, equal_nan=True)
        max_abs = float(np.nanmax(np.abs(aa - bb)))
        print(f"{k}: same={same}, max|diff|={max_abs:.6g}, shape={aa.shape}, dtype={aa.dtype}")
        if not same:
            exit_code = 1

    return exit_code


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("a", help="First .npz (e.g. main)")
    p.add_argument("b", help="Second .npz (e.g. ros)")
    p.add_argument("--atol", type=float, default=0.0)
    p.add_argument("--rtol", type=float, default=0.0)
    args = p.parse_args()

    raise SystemExit(compare_npz(args.a, args.b, atol=args.atol, rtol=args.rtol))


if __name__ == "__main__":
    main()
