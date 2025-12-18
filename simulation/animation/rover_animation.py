# nav_mpc/simulation/animation/rover_animation.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D

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

    x_min_attr = getattr(constraints, "x_min", None) if constraints is not None else None
    x_max_attr = getattr(constraints, "x_max", None) if constraints is not None else None
    if x_min_attr is not None and x_max_attr is not None:
        x_min = np.asarray(x_min_attr, dtype=float).reshape(-1)
        x_max = np.asarray(x_max_attr, dtype=float).reshape(-1)
        if x_min.size >= 5 and x_max.size >= 5:
            wmin = float(min(x_min[3], x_min[4]))
            wmax = float(max(x_max[3], x_max[4]))
            return wmin, wmax

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


def _extract_pred_list(kwargs: dict) -> list[np.ndarray] | None:
    """
    Try multiple common names so main() can pass predictions without changing this file.
    Expected each element: array of shape (N+1, nx) for that control cycle.
    """
    for key in ("X_pred_traj", "X_pred_list", "X_preds", "X_horizon", "X_hist"):
        if key in kwargs and kwargs[key] is not None:
            pred = kwargs[key]
            if isinstance(pred, list):
                return pred
            arr = np.asarray(pred, dtype=float)
            # allow (T, N+1, nx) as a numpy array
            if arr.ndim == 3:
                return [arr[i] for i in range(arr.shape[0])]
    return None


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
    **kwargs,
):
    """
    Animation for the augmented rover model where Δu is the input.

    Expected model convention:
      State x = [px, py, phi, omega_l, omega_r]  (shape (T,5))
      Input u = [alpha_l, alpha_r]              (shape (T-1,2))

    Optional prediction overlay:
      Pass a list of predicted state trajectories via one of:
        - X_pred_traj, X_pred_list, X_preds, X_horizon, X_hist
      Each element must be shaped (N+1, nx) and corresponds to one control cycle.

    What you will see at each timestep:
      1) Current robot body (opaque)
      2) Current predicted XY plan (orange polyline)
      3) 5 ghost robot bodies equally spaced along the horizon, fading to alpha=0.1
    """
    X_pred_list = _extract_pred_list(kwargs)

    # ---------------------------
    #  Export config
    # ---------------------------
    TARGET_ANIM_FPS = 20.0
    VIDEO_DPI = 110
    GIF_DPI = 80

    # Prediction overlay config
    N_GHOSTS = 5
    GHOST_ALPHA_MIN = 0.10
    PLAN_LINE_COLOR = "orange"
    PLAN_LINE_WIDTH = 2.0

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
    fig, (ax_xy, ax_w) = plt.subplots(1, 2, figsize=(8.4, 3.6), gridspec_kw={"wspace": 0.35})
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

    ax_xy.plot(px[0], py[0], marker=".", markersize=3, linestyle="None", label="start")
    if goal_xy is not None:
        ax_xy.plot(goal_xy[0], goal_xy[1], marker="*", markersize=8, linestyle="None", label="goal")

    (trail_line,) = ax_xy.plot([], [], linestyle="--", linewidth=1.5, label="trajectory")

    # Current predicted plan polyline (orange)
    (plan_line,) = ax_xy.plot([], [], linewidth=PLAN_LINE_WIDTH, linestyle="-", color=PLAN_LINE_COLOR, label="plan (current)")

    # Current body (opaque)
    body_poly = Polygon(
        _square_body_vertices(float(px[0]), float(py[0]), float(phi[0]), half_side),
        closed=True,
        alpha=1.0,  # opaque
    )
    ax_xy.add_patch(body_poly)

    # Ghost bodies (5) fading down to alpha=0.1
    ghost_polys: list[Polygon] = []
    if X_pred_list is not None:
        ghost_alphas = np.linspace(1.0, GHOST_ALPHA_MIN, N_GHOSTS + 1)[1:]  # exclude 1.0 (current)
        for a in ghost_alphas:
            gp = Polygon(
                _square_body_vertices(float(px[0]), float(py[0]), float(phi[0]), half_side),
                closed=True,
                alpha=float(a),
            )
            ax_xy.add_patch(gp)
            ghost_polys.append(gp)

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

    ax_w.axhline(0.0, color="k", linewidth=0.8, alpha=0.8)
    ax_w.axhline(float(wmin), linestyle="--", linewidth=1.0, color="k", alpha=0.3)
    ax_w.axhline(float(wmax), linestyle="--", linewidth=1.0, color="k", alpha=0.3)

    (w_l_line,) = ax_w.plot([], [], linewidth=2, label=r"$\omega_l$")
    (w_r_line,) = ax_w.plot([], [], linewidth=2, label=r"$\omega_r$")
    ax_w.legend(loc="best")

    def _set_body(poly: Polygon, px_k: float, py_k: float, phi_k: float) -> None:
        verts = _square_body_vertices(px_k, py_k, phi_k, half_side)
        poly.set_xy(verts)

    def _set_heading(px_k: float, py_k: float, phi_k: float) -> None:
        hlen = 1.2 * half_side
        xh = px_k + hlen * float(np.cos(phi_k))
        yh = py_k + hlen * float(np.sin(phi_k))
        heading_line.set_data([px_k, xh], [py_k, yh])

    def _set_plan_and_ghosts(step_k: int) -> None:
        """
        For the current control cycle:
          - draw current plan XY as orange line
          - draw 5 ghost bodies equally spaced along horizon
        """
        if X_pred_list is None:
            plan_line.set_data([], [])
            for gp in ghost_polys:
                gp.set_xy(_square_body_vertices(float(px[step_k]), float(py[step_k]), float(phi[step_k]), half_side))
                gp.set_visible(False)
            return

        last_cycle = min(max(step_k, 0), len(X_pred_list) - 1)
        pred = np.asarray(X_pred_list[last_cycle], dtype=float)

        if pred.ndim != 2 or pred.shape[1] < 3:
            plan_line.set_data([], [])
            for gp in ghost_polys:
                gp.set_visible(False)
            return

        # current plan line (orange)
        xs = pred[:, 0]
        ys = pred[:, 1]
        plan_line.set_data(xs, ys)

        # equally spaced indices (including 0 and last)
        Nh = pred.shape[0] - 1
        if Nh <= 0:
            for gp in ghost_polys:
                gp.set_visible(False)
            return

        idxs = np.linspace(0, Nh, N_GHOSTS + 1).round().astype(int)  # includes 0
        # ghosts are the 1..N_GHOSTS samples (exclude 0 which is current state)
        ghost_idxs = idxs[1:]

        for gp, ii in zip(ghost_polys, ghost_idxs, strict=False):
            pxg = float(pred[ii, 0])
            pyg = float(pred[ii, 1])
            phig = float(_wrap_to_pi(np.array([pred[ii, 2]])).item())
            _set_body(gp, pxg, pyg, phig)
            gp.set_visible(True)

    def init():
        trail_line.set_data([], [])
        plan_line.set_data([], [])

        _set_body(body_poly, float(px[0]), float(py[0]), float(phi[0]))
        _set_heading(float(px[0]), float(py[0]), float(phi[0]))
        time_text.set_text("")

        w_l_line.set_data([], [])
        w_r_line.set_data([], [])

        for gp in ghost_polys:
            gp.set_visible(False)

        artists = [trail_line, plan_line, body_poly, heading_line, time_text, w_l_line, w_r_line]
        artists.extend(ghost_polys)
        return tuple(artists)

    def update(frame_idx: int):
        k = int(frame_indices[frame_idx])

        # XY trail
        trail_line.set_data(px[: k + 1], py[: k + 1])

        # Current rover pose (opaque)
        _set_body(body_poly, float(px[k]), float(py[k]), float(phi[k]))
        _set_heading(float(px[k]), float(py[k]), float(phi[k]))

        # Plan + ghost bodies from current prediction horizon
        _set_plan_and_ghosts(step_k=k)

        # Wheel-speed evolving lines
        w_l_line.set_data(t[: k + 1], omega_l[: k + 1])
        w_r_line.set_data(t[: k + 1], omega_r[: k + 1])

        time_text.set_text(f"t = {t_anim[frame_idx]:.2f} s")

        artists = [trail_line, plan_line, body_poly, heading_line, time_text, w_l_line, w_r_line]
        artists.extend(ghost_polys)
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
