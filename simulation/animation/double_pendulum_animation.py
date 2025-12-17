# nav_mpc/simulation/animation/double_pendulum_animation.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from models.double_pendulum_model import DoublePendulumModel


def _resolve_results_dir(save_path: str | Path | None) -> Tuple[Path, Path]:
    """
    Returns (results_dir, mp4_path).

    If save_path is None:
        <project_root>/results/double_pendulum_animation.mp4
    If save_path is a directory:
        <save_path>/double_pendulum_animation.mp4
    If save_path is a file:
        that exact file path (and results_dir = parent)
    """
    if save_path is None:
        project_root = Path(__file__).resolve().parents[2]
        results_dir = project_root / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir, results_dir / "double_pendulum_animation.mp4"

    save_path = Path(save_path)
    if save_path.is_dir():
        results_dir = save_path
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir, results_dir / "double_pendulum_animation.mp4"

    results_dir = save_path.parent
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, save_path


def _infer_input_bounds(system: DoublePendulumModel, constraints: object | None) -> Tuple[np.ndarray, np.ndarray]:
    u_min = None
    u_max = None

    if constraints is not None and hasattr(constraints, "get_bounds"):
        bounds = constraints.get_bounds()
        if len(bounds) == 4:
            _, _, u_min_c, u_max_c = bounds
            u_min = np.asarray(u_min_c)
            u_max = np.asarray(u_max_c)

    if u_min is None or u_max is None:
        u_min_attr = getattr(system, "u_min", None) or getattr(system, "umin", None)
        u_max_attr = getattr(system, "u_max", None) or getattr(system, "umax", None)
        if u_min_attr is not None and u_max_attr is not None:
            u_min = np.asarray(u_min_attr)
            u_max = np.asarray(u_max_attr)

    if u_min is None or u_max is None:
        raise ValueError(
            "Could not infer input bounds. Provide constraints.get_bounds() -> "
            "(x_min,x_max,u_min,u_max) or set system.u_min/u_max."
        )

    return u_min, u_max


def _extract_pred_list(kwargs: dict) -> list[np.ndarray] | None:
    """
    Try multiple common names so main() can pass predictions without changing files.

    Expected:
      - list: each element array shaped (N+1, nx) for that control cycle
      - or np array shaped (T-1, N+1, nx)
    """
    for key in ("X_pred_traj", "X_pred_list", "X_preds", "X_horizon", "X_hist"):
        if key in kwargs and kwargs[key] is not None:
            pred = kwargs[key]
            if isinstance(pred, list):
                return pred
            arr = np.asarray(pred, dtype=float)
            if arr.ndim == 3:
                return [arr[i] for i in range(arr.shape[0])]
    return None


def animate_double_pendulum(
    system: DoublePendulumModel,
    constraints: object | None,
    dt: float,
    x_traj: list[np.ndarray] | np.ndarray,
    u_traj: list[np.ndarray] | np.ndarray,
    save_path: str | Path | None = None,
    show: bool = False,
    save_gif: bool = False,
    **kwargs,
):
    """
    Animate double pendulum motion + torque bar (τ on joint 1),
    plus optional prediction horizon overlay (5 poses, orange fading).

    Optional kwargs
    --------------
    X_pred_traj / X_pred_list / X_preds / X_horizon / X_hist:
        Predicted state sequences per control cycle.
        Each element should be shaped (N+1, nx). Only theta1/theta2 are used.
    """
    X_pred_list = _extract_pred_list(kwargs)

    # ---------------------------
    #  Export config
    # ---------------------------
    TARGET_ANIM_FPS = 20.0
    VIDEO_DPI = 100
    GIF_DPI = 80

    # Prediction overlay config
    MAX_PRED_TRAILS = 5
    PRED_ALPHA_MAX = 1.0
    PRED_ALPHA_MIN = 0.10

    # ---------------------------
    #  Stack trajectories
    # ---------------------------
    x_arr = np.vstack(x_traj) if isinstance(x_traj, list) else np.asarray(x_traj, dtype=float)
    u_arr = np.vstack(u_traj) if isinstance(u_traj, list) else np.asarray(u_traj, dtype=float)

    if x_arr.ndim != 2 or x_arr.shape[1] != 4:
        raise ValueError(f"Expected x_traj shape (T,4), got {x_arr.shape}")

    # allow (T-1,) or (T-1,1) or (T-1,nu)
    if u_arr.ndim == 1:
        u_arr = u_arr.reshape(-1, 1)
    elif u_arr.ndim != 2:
        raise ValueError(f"Expected u_traj shape (T-1,nu) or (T-1,), got {u_arr.shape}")

    T = x_arr.shape[0]
    if u_arr.shape[0] != T - 1:
        raise ValueError(f"Expected u_traj length T-1={T-1}, got {u_arr.shape[0]}")

    # build time
    t = dt * np.arange(T, dtype=float)

    l1 = float(system.l1)
    l2 = float(system.l2)

    # -------- Infer u_min, u_max --------
    u_min, u_max = _infer_input_bounds(system, constraints)

    umax = float(
        np.max(
            np.abs(
                np.concatenate([np.atleast_1d(u_min).astype(float), np.atleast_1d(u_max).astype(float)])
            )
        )
    )
    if umax <= 0.0 or not np.isfinite(umax):
        umax = 1.0

    # ---------------------------
    #  Downsample for animation
    # ---------------------------
    sim_fps = 1.0 / float(dt)
    frame_stride = max(1, int(round(sim_fps / TARGET_ANIM_FPS)))

    frame_indices = np.arange(0, T, frame_stride)
    if frame_indices[-1] != T - 1:
        frame_indices = np.append(frame_indices, T - 1)

    t_anim = t[frame_indices]
    x_anim = x_arr[frame_indices]
    u_anim = u_arr[np.clip(frame_indices[:-1], 0, T - 2), 0].astype(float)  # visualize first input
    T_anim = x_anim.shape[0]

    # ---------------------------
    #  Precompute positions (actual trajectory)
    # ---------------------------
    theta1 = x_anim[:, 0]
    theta2 = x_anim[:, 1]

    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)

    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)

    # ---------------------------
    #  Figure / axes
    # ---------------------------
    fig, (ax_pend, ax_torque) = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"wspace": 0.45})
    fig.suptitle("Double Pendulum MPC simulation")

    # --- Pendulum axis ---
    ax_pend.set_aspect("equal", "box")
    max_len = 1.2 * (l1 + l2)
    ax_pend.set_xlim(-max_len, max_len)
    ax_pend.set_ylim(-max_len, max_len)
    ax_pend.set_xlabel("x [m]")
    ax_pend.set_ylabel("y [m]")
    ax_pend.grid(True, alpha=0.25)
    ax_pend.plot(0, 0, "ko")  # pivot

    # actual (blue)
    (rod1,) = ax_pend.plot([], [], "o-", lw=2, color="tab:blue")
    (rod2,) = ax_pend.plot([], [], "o-", lw=2, color="tab:blue")
    time_text = ax_pend.text(0.05, 0.9, "", transform=ax_pend.transAxes)

    # predicted (orange, fading) — each pose has two rods
    pred_rod1: list[Line2D] = []
    pred_rod2: list[Line2D] = []
    if X_pred_list is not None:
        alphas = np.linspace(PRED_ALPHA_MAX, PRED_ALPHA_MIN, MAX_PRED_TRAILS)
        for a in alphas:
            (p1,) = ax_pend.plot([], [], "o-", lw=1.6, color="tab:orange", alpha=float(a))
            (p2,) = ax_pend.plot([], [], "o-", lw=1.6, color="tab:orange", alpha=float(a))
            pred_rod1.append(p1)
            pred_rod2.append(p2)

    # --- Torque bar axis ---
    ax_torque.set_xlim(0, 1)
    ax_torque.set_ylim(-umax, umax)
    ax_torque.set_xticks([])
    ax_torque.set_ylabel("Torque u [Nm]", labelpad=8)
    ax_torque.grid(True, axis="y", alpha=0.25)
    ax_torque.axhline(0.0, color="k", linewidth=0.8)

    if np.isfinite(u_min).all():
        ax_torque.axhline(float(np.atleast_1d(u_min).min()), linestyle="--", linewidth=1.0, color="k", alpha=0.3)
    if np.isfinite(u_max).all():
        ax_torque.axhline(float(np.atleast_1d(u_max).max()), linestyle="--", linewidth=1.0, color="k", alpha=0.3)

    bar_width = 0.6
    torque_rect = Rectangle((0.2, 0.0), bar_width, 0.0)
    ax_torque.add_patch(torque_rect)

    def _set_pred_poses(step_k: int) -> None:
        """
        Draw 5 predicted double-pendulum poses for the *current* horizon:
          - equally spaced along horizon indices
          - exclude index 0 (current state), so it doesn't coincide with blue rods
          - same orange color, fading alpha (set per artist)
        """
        if X_pred_list is None:
            return

        last_cycle = min(max(step_k, 0), len(X_pred_list) - 1)
        pred = np.asarray(X_pred_list[last_cycle], dtype=float)

        # expected shape (N+1, nx) with theta1 at col 0, theta2 at col 1
        if pred.ndim != 2 or pred.shape[0] < 2 or pred.shape[1] < 2:
            for p1, p2 in zip(pred_rod1, pred_rod2):
                p1.set_data([], [])
                p2.set_data([], [])
            return

        Np1 = pred.shape[0]
        if Np1 <= 2:
            for p1, p2 in zip(pred_rod1, pred_rod2):
                p1.set_data([], [])
                p2.set_data([], [])
            return

        # future-only indices: 1..N
        idx = np.linspace(1, Np1 - 1, MAX_PRED_TRAILS).round().astype(int)
        idx = np.clip(idx, 1, Np1 - 1)

        th1 = pred[idx, 0]
        th2 = pred[idx, 1]

        x1p = l1 * np.sin(th1)
        y1p = -l1 * np.cos(th1)

        x2p = x1p + l2 * np.sin(th2)
        y2p = y1p - l2 * np.cos(th2)

        for j in range(MAX_PRED_TRAILS):
            x1b, y1b = float(x1p[j]), float(y1p[j])
            x2b, y2b = float(x2p[j]), float(y2p[j])

            pred_rod1[j].set_data([0.0, x1b], [0.0, y1b])
            pred_rod2[j].set_data([x1b, x2b], [y1b, y2b])

    def init():
        rod1.set_data([], [])
        rod2.set_data([], [])
        torque_rect.set_height(0.0)
        torque_rect.set_y(0.0)
        time_text.set_text("")

        if X_pred_list is not None:
            for p1, p2 in zip(pred_rod1, pred_rod2):
                p1.set_data([], [])
                p2.set_data([], [])

        artists = [rod1, rod2, torque_rect, time_text]
        artists.extend(pred_rod1)
        artists.extend(pred_rod2)
        return tuple(artists)

    def update(frame_idx: int):
        # current step index in original trajectory
        k = int(frame_indices[frame_idx])

        x1b, y1b = float(x1[frame_idx]), float(y1[frame_idx])
        x2b, y2b = float(x2[frame_idx]), float(y2[frame_idx])

        rod1.set_data([0.0, x1b], [0.0, y1b])
        rod2.set_data([x1b, x2b], [y1b, y2b])

        if X_pred_list is not None:
            _set_pred_poses(step_k=k)

        # torque for this frame (between samples)
        uu = float(u_anim[frame_idx]) if frame_idx < T_anim - 1 else 0.0

        if uu >= 0.0:
            torque_rect.set_y(0.0)
            torque_rect.set_height(min(uu, umax))
        else:
            torque_rect.set_y(max(-umax, uu))
            torque_rect.set_height(-uu)

        time_text.set_text(f"t = {t_anim[frame_idx]:.2f} s")

        artists = [rod1, rod2, torque_rect, time_text]
        artists.extend(pred_rod1)
        artists.extend(pred_rod2)
        return tuple(artists)

    ani = FuncAnimation(
        fig,
        update,
        frames=T_anim,
        init_func=init,
        blit=True,
        interval=1000.0 / TARGET_ANIM_FPS,
    )

    # ---------------------------
    #  Save outputs
    # ---------------------------
    _, mp4_path = _resolve_results_dir(save_path)
    video_fps = int(TARGET_ANIM_FPS)

    ani.save(mp4_path, fps=video_fps, dpi=VIDEO_DPI)
    print(f"[animator] Saved animation to {mp4_path}")

    if save_gif:
        gif_path = mp4_path.with_suffix(".gif")
        writer = PillowWriter(fps=video_fps)
        ani.save(gif_path, writer=writer, dpi=GIF_DPI)
        print(f"[animator] Saved GIF animation to {gif_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return ani
