# nav_mpc/simulation/animation/rover_animation.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon

from models.simple_rover_model import SimpleRoverModel


def _resolve_results_dir(save_path: str | Path | None) -> Tuple[Path, Path]:
    """
    Returns (results_dir, mp4_path).

    If save_path is None:
        <project_root>/results/rover_animation.mp4
    If save_path is a directory:
        <save_path>/rover_animation.mp4
    If save_path is a file:
        that exact file path (and results_dir = parent)
    """
    if save_path is None:
        project_root = Path(__file__).resolve().parents[2]
        results_dir = project_root / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir, results_dir / "rover_animation.mp4"

    save_path = Path(save_path)
    if save_path.is_dir():
        results_dir = save_path
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir, results_dir / "rover_animation.mp4"

    results_dir = save_path.parent
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, save_path


def _infer_wheel_speed_bounds(system: SimpleRoverModel, constraints: object | None) -> Tuple[float, float]:
    """
    Infer bounds for wheel speeds omega_l, omega_r from state bounds:
      - we expect the augmented model: x = [px,py,phi,omega_l,omega_r]
      - and constraints to expose x_min/x_max (or get_bounds()).
    """
    # Preferred: constraints.get_bounds()
    if constraints is not None and hasattr(constraints, "get_bounds"):
        bounds = constraints.get_bounds()
        if len(bounds) == 4:
            x_min, x_max, _, _ = bounds
            x_min = np.asarray(x_min, dtype=float).reshape(-1)
            x_max = np.asarray(x_max, dtype=float).reshape(-1)
            if x_min.size >= 5 and x_max.size >= 5:
                wmin = float(min(x_min[3], x_min[4]))
                wmax = float(max(x_max[3], x_max[4]))
                return wmin, wmax

    # Next: direct attributes on constraints
    x_min_attr = getattr(constraints, "x_min", None) if constraints is not None else None
    x_max_attr = getattr(constraints, "x_max", None) if constraints is not None else None
    if x_min_attr is not None and x_max_attr is not None:
        x_min = np.asarray(x_min_attr, dtype=float).reshape(-1)
        x_max = np.asarray(x_max_attr, dtype=float).reshape(-1)
        if x_min.size >= 5 and x_max.size >= 5:
            wmin = float(min(x_min[3], x_min[4]))
            wmax = float(max(x_max[3], x_max[4]))
            return wmin, wmax

    # Fallback: use typical limits (matches your earlier u bounds)
    return -15.0, 15.0


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _square_body_vertices(px: float, py: float, phi: float, half_side: float) -> np.ndarray:
    """
    Square centered at (px,py), side = 2*half_side, rotated by phi.
    Returns (4,2) vertices in CCW order.
    """
    pts = np.array(
        [
            [-half_side, -half_side],
            [+half_side, -half_side],
            [+half_side, +half_side],
            [-half_side, +half_side],
        ],
        dtype=float,
    )

    c = float(np.cos(phi))
    s = float(np.sin(phi))
    R = np.array([[c, -s], [s, c]], dtype=float)

    return (pts @ R.T) + np.array([px, py], dtype=float)


def animate_rover(
    system: SimpleRoverModel,
    constraints: object | None,
    dt: float,
    x_traj: list[np.ndarray] | np.ndarray,
    u_traj: list[np.ndarray] | np.ndarray,
    x_goal: np.ndarray | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
    save_gif: bool = False,
):
    """
    Animation for the augmented rover model where Δu is the input.

    Expected model convention:
      State x = [px, py, phi, omega_l, omega_r]  (shape (T,5))
      Input u = [alpha_l, alpha_r]              (shape (T-1,2))

    This animation shows:
      - Left: XY trajectory with a square rover body + heading line
      - Right: wheel speeds (omega_l, omega_r) as *growing lines* over time

    Notes
    -----
    We do NOT plot accelerations here (even though they are the true inputs).
    Wheel speeds come from the state: x[:,3], x[:,4].
    """
    # ---------------------------
    #  Export config
    # ---------------------------
    TARGET_ANIM_FPS = 20.0
    VIDEO_DPI = 110
    GIF_DPI = 80

    # ---------------------------
    #  Stack trajectories
    # ---------------------------
    x_arr = np.vstack(x_traj) if isinstance(x_traj, list) else np.asarray(x_traj, dtype=float)
    u_arr = np.vstack(u_traj) if isinstance(u_traj, list) else np.asarray(u_traj, dtype=float)

    if x_arr.ndim != 2 or x_arr.shape[1] != 5:
        raise ValueError(f"Expected x_traj shape (T,5), got {x_arr.shape}")
    if u_arr.ndim != 2 or u_arr.shape[1] != 2:
        raise ValueError(f"Expected u_traj shape (T-1,2), got {u_arr.shape}")

    T = x_arr.shape[0]
    if u_arr.shape[0] != T - 1:
        raise ValueError(f"Expected u_traj length T-1={T-1}, got {u_arr.shape[0]}")

    # time base
    t = dt * np.arange(T, dtype=float)

    # pose
    px = x_arr[:, 0].astype(float)
    py = x_arr[:, 1].astype(float)
    phi = _wrap_to_pi(x_arr[:, 2].astype(float))

    # wheel speeds from state
    omega_l = x_arr[:, 3].astype(float)
    omega_r = x_arr[:, 4].astype(float)

    # goal parsing
    goal_xy = None
    if x_goal is not None:
        x_goal = np.asarray(x_goal, dtype=float).reshape(-1)
        if x_goal.size >= 2:
            goal_xy = x_goal[:2].copy()

    # body size: side = 2*L (L is half-track)
    L = float(getattr(system, "L"))
    half_side = float(L)

    # wheel speed bounds for scaling
    wmin, wmax = _infer_wheel_speed_bounds(system, constraints)
    wabs = max(abs(wmin), abs(wmax), 1e-6)

    # ---------------------------
    #  Downsample for animation
    # ---------------------------
    sim_fps = 1.0 / float(dt)
    frame_stride = max(1, int(round(sim_fps / TARGET_ANIM_FPS)))

    frame_indices = np.arange(0, T, frame_stride)
    if frame_indices[-1] != T - 1:
        frame_indices = np.append(frame_indices, T - 1)

    t_anim = t[frame_indices]
    T_anim = frame_indices.size

    # ---------------------------
    #  Figure / axes
    # ---------------------------
    fig, (ax_xy, ax_w) = plt.subplots(
        1, 2, figsize=(8.4, 3.6), gridspec_kw={"wspace": 0.35}
    )
    fig.suptitle("Simple rover simulation")

    # -------- XY axis --------
    ax_xy.set_aspect("equal", "box")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.grid(True)

    margin = 3.0 * half_side
    xmin, xmax = float(px.min() - margin), float(px.max() + margin)
    ymin, ymax = float(py.min() - margin), float(py.max() + margin)
    if goal_xy is not None:
        xmin = min(xmin, float(goal_xy[0]) - margin)
        xmax = max(xmax, float(goal_xy[0]) + margin)
        ymin = min(ymin, float(goal_xy[1]) - margin)
        ymax = max(ymax, float(goal_xy[1]) + margin)

    ax_xy.set_xlim(xmin, xmax)
    ax_xy.set_ylim(ymin, ymax)

    ax_xy.plot(px[0], py[0], marker=".", markersize=2, linestyle="None", label="start")
    if goal_xy is not None:
        ax_xy.plot(goal_xy[0], goal_xy[1], marker="*", markersize=8, linestyle="None", label="goal")

    (trail_line,) = ax_xy.plot([], [], linestyle="--", linewidth=1.5, label="trajectory")

    body_poly = Polygon(
        _square_body_vertices(float(px[0]), float(py[0]), float(phi[0]), half_side),
        closed=True,
        alpha=0.6,
    )
    ax_xy.add_patch(body_poly)

    (heading_line,) = ax_xy.plot([], [], linewidth=2)
    time_text = ax_xy.text(0.02, 0.95, "", transform=ax_xy.transAxes)
    ax_xy.legend(loc="best")

    # -------- wheel speed axis --------
    ax_w.set_title("Wheel speeds")
    ax_w.set_xlabel("time [s]")
    ax_w.set_ylabel(r"$\omega$ [rad/s]")
    ax_w.grid(True)

    ax_w.set_xlim(float(t[0]), float(t[-1]))
    ax_w.set_ylim(-1.1 * wabs, 1.1 * wabs)

    # bounds
    ax_w.axhline(0.0, color="k", linewidth=0.8, alpha=0.8)
    ax_w.axhline(float(wmin), linestyle="--", linewidth=1.0, color="k", alpha=0.3)
    ax_w.axhline(float(wmax), linestyle="--", linewidth=1.0, color="k", alpha=0.3)

    # evolving lines (start empty)
    (w_l_line,) = ax_w.plot([], [], linewidth=2, label=r"$\omega_l$")
    (w_r_line,) = ax_w.plot([], [], linewidth=2, label=r"$\omega_r$")
    ax_w.legend(loc="best")

    def _set_body(px_k: float, py_k: float, phi_k: float) -> None:
        verts = _square_body_vertices(px_k, py_k, phi_k, half_side)
        body_poly.set_xy(verts)

        hlen = 1.2 * half_side
        xh = px_k + hlen * float(np.cos(phi_k))
        yh = py_k + hlen * float(np.sin(phi_k))
        heading_line.set_data([px_k, xh], [py_k, yh])

    def init():
        trail_line.set_data([], [])
        _set_body(float(px[0]), float(py[0]), float(phi[0]))
        time_text.set_text("")
        w_l_line.set_data([], [])
        w_r_line.set_data([], [])
        return trail_line, body_poly, heading_line, time_text, w_l_line, w_r_line

    def update(frame_idx: int):
        k = int(frame_indices[frame_idx])

        px_k = float(px[k])
        py_k = float(py[k])
        phi_k = float(phi[k])

        # XY trail
        trail_line.set_data(px[: k + 1], py[: k + 1])

        # Rover pose
        _set_body(px_k, py_k, phi_k)

        # Wheel-speed evolving lines
        w_l_line.set_data(t[: k + 1], omega_l[: k + 1])
        w_r_line.set_data(t[: k + 1], omega_r[: k + 1])

        time_text.set_text(f"t = {t_anim[frame_idx]:.2f} s")
        return trail_line, body_poly, heading_line, time_text, w_l_line, w_r_line

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
