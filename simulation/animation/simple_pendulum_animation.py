# nav_mpc/simulation/animation/simple_pendulum_animation.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
from pathlib import Path

from models.simple_pendulum_model import SimplePendulumModel


def animate_pendulum(
    system: SimplePendulumModel,
    constraints: object | None,
    t: np.ndarray,
    x_traj: np.ndarray,
    u_traj: np.ndarray,
    save_path: str | Path | None = None,
    show: bool = False,
    save_gif: bool = False,
):
    """
    Animate simple pendulum motion + torque bar.

    Parameters
    ----------
    system : SimplePendulumModel
        Provides pendulum length (system.l).
    constraints : object or None
        If not None and has get_bounds(), used to infer input bounds
        (u_min, u_max) for scaling the torque bar.
    t : (T,)
        Time stamps for x_traj (assumed uniform).
    x_traj : (T, 2)
        State trajectory, x[:,0] = theta, x[:,1] = theta_dot.
    u_traj : (T-1, 1) or (T-1,)
        Control inputs applied between samples.
    save_path : str or Path, optional
        If None, saves to '<project_root>/results/pendulum_animation.mp4'.
    show : bool
        Whether to plt.show() the animation.
    save_gif : bool
        If True, also saves a GIF next to the MP4.
    """
    x_traj = np.asarray(x_traj)
    u_traj = np.asarray(u_traj).reshape(-1)
    t      = np.asarray(t)

    T = x_traj.shape[0]
    assert x_traj.shape[1] == 2, "SimplePendulumModel assumed to have 2 states."
    assert u_traj.shape[0] == T - 1, "Expect u_traj length = len(x_traj)-1."

    l = system.l

    # -------- Infer u_min, u_max from constraints/system --------
    u_min = None
    u_max = None

    # 1) constraints.get_bounds() if available
    if constraints is not None and hasattr(constraints, "get_bounds"):
        bounds = constraints.get_bounds()
        # expected: (x_min, x_max, u_min, u_max)
        if len(bounds) == 4:
            _, _, u_min_c, u_max_c = bounds
            u_min = np.asarray(u_min_c)
            u_max = np.asarray(u_max_c)

    # 2) fall back to system attributes if still None
    if u_min is None or u_max is None:
        u_min_attr = getattr(system, "u_min", None) or getattr(system, "umin", None)
        u_max_attr = getattr(system, "u_max", None) or getattr(system, "umax", None)
        if u_min_attr is not None and u_max_attr is not None:
            u_min = np.asarray(u_min_attr)
            u_max = np.asarray(u_max_attr)

    if u_min is None or u_max is None:
        raise ValueError(
            "Could not infer input bounds: constraints.get_bounds() did not "
            "return u_min/u_max and system has no u_min/u_max attributes."
        )

    # We assume scalar input for the pendulum; if vector, take the worst-case
    umax = float(np.max(np.abs(np.concatenate(
        [np.atleast_1d(u_min), np.atleast_1d(u_max)]
    ))))

    # Precompute bob positions
    theta = x_traj[:, 0]
    x_bob = l * np.sin(theta)
    y_bob = -l * np.cos(theta)

    # Setup figure: left = pendulum, right = torque bar
    fig, (ax_pend, ax_torque) = plt.subplots(
        1, 2, figsize=(8, 4), gridspec_kw={"wspace": 0.4}
    )
    fig.suptitle("Pendulum MPC simulation")

    # --- Pendulum axis ---
    ax_pend.set_aspect("equal", "box")
    max_len = 1.2 * l
    ax_pend.set_xlim(-max_len, max_len)
    ax_pend.set_ylim(-max_len, max_len)
    ax_pend.set_xlabel("x [m]")
    ax_pend.set_ylabel("y [m]")

    ax_pend.plot(0, 0, "ko")  # pivot

    line, = ax_pend.plot([], [], "o-", lw=2)
    time_text = ax_pend.text(0.05, 0.9, "", transform=ax_pend.transAxes)

    # --- Torque bar axis ---
    ax_torque.set_xlim(0, 1)
    ax_torque.set_ylim(-umax, umax)
    ax_torque.set_xticks([])
    ax_torque.set_ylabel("Torque u [Nm]", labelpad=8)
    ax_torque.axhline(0.0, color="k", linewidth=0.8)

    # (Optional) draw u_min / u_max as dashed lines for reference
    if np.isfinite(u_min).all():
        ax_torque.axhline(float(np.atleast_1d(u_min).min()),
                          linestyle="--", linewidth=1.0, color="k", alpha=0.3)
    if np.isfinite(u_max).all():
        ax_torque.axhline(float(np.atleast_1d(u_max).max()),
                          linestyle="--", linewidth=1.0, color="k", alpha=0.3)

    bar_width = 0.6
    torque_rect = Rectangle(
        (0.2, 0.0),  # bottom-left
        bar_width,
        0.0,         # height updated in animation
    )
    ax_torque.add_patch(torque_rect)

    def init():
        line.set_data([], [])
        torque_rect.set_height(0.0)
        torque_rect.set_y(0.0)
        time_text.set_text("")
        return line, torque_rect, time_text

    def update(frame):
        xb = x_bob[frame]
        yb = y_bob[frame]
        line.set_data([0, xb], [0, yb])

        if frame < T - 1:
            u = u_traj[frame]
        else:
            u = 0.0

        if u >= 0:
            torque_rect.set_y(0.0)
            torque_rect.set_height(min(u, umax))
        else:
            torque_rect.set_y(max(-umax, u))
            torque_rect.set_height(-u)

        time_text.set_text(f"t = {t[frame]:.2f} s")

        return line, torque_rect, time_text

    ani = FuncAnimation(
        fig,
        update,
        frames=T,
        init_func=init,
        blit=True,
        interval=1000 * (t[1] - t[0]),
    )

    # ---------- Auto-save to <project_root>/results ----------
    if save_path is None:
        project_root = Path(__file__).resolve().parents[2]
        results_dir = project_root / "results"
        results_dir.mkdir(exist_ok=True)
        save_path = results_dir / "pendulum_animation.mp4"

    save_path = Path(save_path)
    fps = int(1.0 / (t[1] - t[0]))

    # Save MP4
    ani.save(save_path, fps=fps)
    print(f"[animator] Saved animation to {save_path}")

    # Optionally also save GIF
    if save_gif:
        gif_path = save_path.with_suffix(".gif")
        writer = PillowWriter(fps=fps)
        ani.save(gif_path, writer=writer)
        print(f"[animator] Saved GIF animation to {gif_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return ani
