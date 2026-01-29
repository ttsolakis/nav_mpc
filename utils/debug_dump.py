# nav_mpc/utils/debug_dump.py
from __future__ import annotations

import os
import hashlib
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


def get_repo_root_from_file(file_path: str) -> str:
    # file_path is typically __file__ from nav_mpc/main.py or any module inside nav_mpc
    return os.path.dirname(os.path.dirname(os.path.abspath(file_path)))


def get_default_debug_dir(repo_root: str) -> str:
    return os.path.join(repo_root, "results", "debug")


def _sha1(a: np.ndarray) -> str:
    b = np.ascontiguousarray(a)
    return hashlib.sha1(b.view(np.uint8)).hexdigest()[:12]


def array_stats(name: str, a: Optional[np.ndarray]) -> str:
    if a is None:
        return f"{name}: None"
    a = np.asarray(a)
    if a.size == 0:
        return f"{name}: shape={a.shape}, empty"
    return (
        f"{name}: shape={a.shape}, dtype={a.dtype}, "
        f"min={np.nanmin(a):.3g}, max={np.nanmax(a):.3g}, "
        f"nan={np.isnan(a).any()}, inf={np.isinf(a).any()}, sha1={_sha1(a)}"
    )


def array_head(name: str, a: Optional[np.ndarray], k: int = 8) -> str:
    if a is None:
        return f"{name}: None"
    v = np.asarray(a).ravel()
    return f"{name} head: {v[:k]}"


def dump_npz(
    *,
    dump_dir: str,
    tag: str,              # e.g. "main" or "ros"
    step_idx: int,
    dt: float,
    N: int,
    x: np.ndarray,
    X: np.ndarray,
    U: np.ndarray,
    Xref_seq: np.ndarray,
    obstacles_xy: Optional[np.ndarray],
    global_path: np.ndarray,
) -> str:
    os.makedirs(dump_dir, exist_ok=True)
    path = os.path.join(dump_dir, f"nav_mpc_{tag}_step{step_idx:04d}.npz")

    np.savez(
        path,
        dt=np.array([dt], dtype=float),
        N=np.array([N], dtype=int),
        x=np.asarray(x),
        X=np.asarray(X),
        U=np.asarray(U),
        Xref_seq=np.asarray(Xref_seq),
        obstacles_xy=np.asarray(obstacles_xy) if obstacles_xy is not None else np.zeros((0, 2)),
        global_path=np.asarray(global_path),
    )
    return path


def format_first_iter_summary(
    *,
    step_idx: int,
    dt: float,
    N: int,
    embedded: bool,
    debugging: bool,
    x: np.ndarray,
    X: np.ndarray,
    U: np.ndarray,
    Xref_seq: np.ndarray,
    obstacles_xy: Optional[np.ndarray],
    global_path: np.ndarray,
) -> list[str]:
    lines: list[str] = []
    lines.append("==== nav_mpc DEBUG: first tick inputs ====")
    lines.append(f"step_idx={step_idx}, dt={dt}, N={N}, embedded={embedded}, debugging={debugging}")
    lines.append(array_stats("x", x))
    lines.append(array_stats("X (warm)", X))
    lines.append(array_stats("U (warm)", U))
    lines.append(array_stats("Xref_seq", Xref_seq))
    lines.append(array_stats("obstacles_xy", obstacles_xy))
    lines.append(array_stats("global_path", global_path))
    lines.append(array_head("x", x))
    lines.append(array_head("Xref_seq", Xref_seq))
    lines.append(array_head("obstacles_xy", obstacles_xy))
    lines.append(array_head("global_path", global_path))
    return lines
