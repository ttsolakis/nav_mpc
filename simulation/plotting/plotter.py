# nav_mpc/simulation/plotting/plotter.py

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from models.dynamics import SystemModel


def plot_state_input_trajectories(
    system: SystemModel,
    t: np.ndarray,
    x_traj: np.ndarray,
    u_traj: np.ndarray,
    x_ref: np.ndarray | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """
    Generic plotting for state and input trajectories.

    Parameters
    ----------
    system : SystemModel
        Provides state_dim and input_dim.
    t : (T,)
        Time stamps for state trajectory x_traj.
        Inputs are assumed defined on intervals [t_k, t_{k+1}),
        so u_traj must have shape (T-1, nu).
    x_traj : (T, nx)
        Closed-loop state trajectory.
    u_traj : (T-1, nu)
        Closed-loop input trajectory.
    x_ref : (nx,), optional
        Reference state to plot as horizontal lines (if given).
    save_path : str or Path, optional
        If None, figures are saved to
        '<project_root>/results/state_trajectories.png' and
        '<project_root>/results/input_trajectories.png'.
        If given and is a directory, files are saved inside that directory
        with the same filenames.
    show : bool
        Whether to show the figures with plt.show().
    """
    nx = system.state_dim
    nu = system.input_dim

    t      = np.asarray(t)
    x_traj = np.asarray(x_traj)
    u_traj = np.asarray(u_traj)

    assert x_traj.shape[0] == t.shape[0], "x_traj and t must have same length"
    assert u_traj.shape[0] == t.shape[0] - 1, "u_traj must have length len(t)-1"

    # Determine results directory
    if save_path is None:
        # plotter.py is nav_mpc/simulation/plotting/plotter.py
        project_root = Path(__file__).resolve().parents[2]
        results_dir = project_root / "results"
    else:
        save_path = Path(save_path)
        if save_path.is_dir():
            results_dir = save_path
        else:
            # if a file path is passed, use its parent directory
            results_dir = save_path.parent

    results_dir.mkdir(exist_ok=True)

    # =========================
    #  Figure 1: state plots
    # =========================
    fig_x, axes_x = plt.subplots(nx, 1, figsize=(8, 2.5 * nx), sharex=True)
    if nx == 1:
        axes_x = [axes_x]  # make it iterable

    fig_x.suptitle("State trajectories")

    for i in range(nx):
        ax = axes_x[i]
        ax.plot(t, x_traj[:, i], label=f"x{i}")
        if x_ref is not None:
            ax.axhline(x_ref[i], linestyle="--", alpha=0.5, label="x_ref")
        ax.set_ylabel(f"x{i}")
        ax.grid(True)
        ax.legend(loc="best")

    axes_x[-1].set_xlabel("Time [s]")

    fig_x.tight_layout(rect=[0, 0.03, 1, 0.95])

    state_path = results_dir / "state_trajectories.png"
    fig_x.savefig(state_path, dpi=150, bbox_inches="tight")
    print(f"[plotter] Saved state figure to {state_path}")

    # =========================
    #  Figure 2: input plots
    # =========================
    t_u = t[:-1]

    fig_u, axes_u = plt.subplots(nu, 1, figsize=(8, 2.5 * nu), sharex=True)
    if nu == 1:
        axes_u = [axes_u]

    fig_u.suptitle("Input trajectories")

    for j in range(nu):
        ax = axes_u[j]
        ax.step(t_u, u_traj[:, j], where="post", label=f"u{j}")
        ax.set_ylabel(f"u{j}")
        ax.grid(True)
        ax.legend(loc="best")

    axes_u[-1].set_xlabel("Time [s]")

    fig_u.tight_layout(rect=[0, 0.03, 1, 0.95])

    input_path = results_dir / "input_trajectories.png"
    fig_u.savefig(input_path, dpi=150, bbox_inches="tight")
    print(f"[plotter] Saved input figure to {input_path}")

    if show:
        plt.show()
    else:
        plt.close(fig_x)
        plt.close(fig_u)
