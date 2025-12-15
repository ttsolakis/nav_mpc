# nav_mpc/simulation/animation/double_pendulum_animation.py

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

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


def animate_double_pendulum(
    system: DoublePendulumModel,
    constraints: object | None,
    dt: float,
    x_traj: list[np.ndarray] | np.ndarray,
    u_traj: list[np.ndarray] | np.ndarray,
    save_path: str | Path | None = None,
    show: bool = False,
    save_gif: bool = False,
):
    """
    Animate double pendulum motion + torque bar (τ on joint 1).

    This version accepts dt + trajectories and builds time vector internally.

    Parameters
    ----------
    system : DoublePendulumModel
        Provides lengths (system.l1, system.l2).
    constraints : object or None
        If not None and has get_bounds(), used to infer input bounds (u_min, u_max).
    dt : float
        Sampling time [s]. Used to build t.
    x_traj : list of (4,) or (T,4)
        State trajectory: [theta1, theta2, theta1_dot, theta2_dot].
    u_traj : list of (nu,) or (T-1,nu)
        Inputs applied between samples. Only u[:,0] is visualized as torque bar.
    save_path : str | Path, optional
        None -> <project_root>/results/double_pendulum_animation.mp4
        dir  -> <dir>/double_pendulum_animation.mp4
        file -> saved exactly there
    show : bool
        Whether to show the animation.
    save_gif : bool
        If True, also saves a GIF next to the MP4.
    """
    # ---------------------------
    #  Lightweight export config
    # ---------------------------
    TARGET_ANIM_FPS = 20.0
    VIDEO_DPI = 100
    GIF_DPI = 80

    # ---------------------------
    #  Stack trajectories
    # ---------------------------
    x_arr = np.vstack(x_traj) if isinstance(x_traj, list) else np.asarray(x_traj)
    u_arr = np.vstack(u_traj) if isinstance(u_traj, list) else np.asarray(u_traj)

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

    # choose scale for bar
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
    # inputs are between samples; pick u[k] aligned with frame k (except last)
    u_anim = u_arr[np.clip(frame_indices[:-1], 0, T - 2), 0].astype(float)  # visualize first input
    T_anim = x_anim.shape[0]

    # ---------------------------
    #  Precompute positions
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
    fig, (ax_pend, ax_torque) = plt.subplots(1, 2, figsize=(7, 3.2), gridspec_kw={"wspace": 0.45})
    fig.suptitle("Double Pendulum MPC simulation")

    # --- Pendulum axis ---
    ax_pend.set_aspect("equal", "box")
    max_len = 1.2 * (l1 + l2)
    ax_pend.set_xlim(-max_len, max_len)
    ax_pend.set_ylim(-max_len, max_len)
    ax_pend.set_xlabel("x [m]")
    ax_pend.set_ylabel("y [m]")
    ax_pend.plot(0, 0, "ko")  # pivot

    rod1, = ax_pend.plot([], [], "o-", lw=2)
    rod2, = ax_pend.plot([], [], "o-", lw=2)
    time_text = ax_pend.text(0.05, 0.9, "", transform=ax_pend.transAxes)

    # --- Torque bar axis ---
    ax_torque.set_xlim(0, 1)
    ax_torque.set_ylim(-umax, umax)
    ax_torque.set_xticks([])
    ax_torque.set_ylabel("Torque u [Nm]", labelpad=8)
    ax_torque.axhline(0.0, color="k", linewidth=0.8)

    if np.isfinite(u_min).all():
        ax_torque.axhline(float(np.atleast_1d(u_min).min()), linestyle="--", linewidth=1.0, color="k", alpha=0.3)
    if np.isfinite(u_max).all():
        ax_torque.axhline(float(np.atleast_1d(u_max).max()), linestyle="--", linewidth=1.0, color="k", alpha=0.3)

    bar_width = 0.6
    torque_rect = Rectangle((0.2, 0.0), bar_width, 0.0)
    ax_torque.add_patch(torque_rect)

    def init():
        rod1.set_data([], [])
        rod2.set_data([], [])
        torque_rect.set_height(0.0)
        torque_rect.set_y(0.0)
        time_text.set_text("")
        return rod1, rod2, torque_rect, time_text

    def update(frame_idx: int):
        x1b, y1b = float(x1[frame_idx]), float(y1[frame_idx])
        x2b, y2b = float(x2[frame_idx]), float(y2[frame_idx])

        rod1.set_data([0.0, x1b], [0.0, y1b])
        rod2.set_data([x1b, x2b], [y1b, y2b])

        # torque for this frame (between samples)
        uu = float(u_anim[frame_idx]) if frame_idx < T_anim - 1 else 0.0

        if uu >= 0.0:
            torque_rect.set_y(0.0)
            torque_rect.set_height(min(uu, umax))
        else:
            torque_rect.set_y(max(-umax, uu))
            torque_rect.set_height(-uu)

        time_text.set_text(f"t = {t_anim[frame_idx]:.2f} s")
        return rod1, rod2, torque_rect, time_text

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
    results_dir, mp4_path = _resolve_results_dir(save_path)
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
