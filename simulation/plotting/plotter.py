# nav_mpc/simulation/plotting/plotter.py

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from models.dynamics import SystemModel


def plot_state_input_trajectories(
    system: SystemModel,
    constraints: object | None,
    t: np.ndarray,
    x_traj: np.ndarray,
    u_traj: np.ndarray,
    x_ref: np.ndarray | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
    x_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    u_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    """
    Generic plotting for state and input trajectories.

    Parameters
    ----------
    system : SystemModel
        Provides state_dim and input_dim (and optionally bounds).
    constraints : object or None
        If not None and has method get_bounds(), it should return
        (x_min, x_max, u_min, u_max). These are used as default
        state/input bounds unless x_bounds/u_bounds are explicitly given.
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
    x_bounds : (x_min, x_max), optional
        Each is (nx,) array. Overrides bounds from `constraints`/`system`
        if provided.
    u_bounds : (u_min, u_max), optional
        Each is (nu,) array. Overrides bounds from `constraints`/`system`
        if provided.
    """
    nx = system.state_dim
    nu = system.input_dim

    t      = np.asarray(t)
    x_traj = np.asarray(x_traj)
    u_traj = np.asarray(u_traj)

    assert x_traj.shape[0] == t.shape[0], "x_traj and t must have same length"
    assert u_traj.shape[0] == t.shape[0] - 1, "u_traj must have length len(t)-1"

    # ------------ Resolve bounds from constraints/system ------------
    # 1) constraints.get_bounds() if available
    if (x_bounds is None or u_bounds is None) and constraints is not None:
        if hasattr(constraints, "get_bounds"):
            bounds = constraints.get_bounds()
            if len(bounds) == 4:
                x_min_c, x_max_c, u_min_c, u_max_c = map(np.asarray, bounds)
                if x_bounds is None:
                    x_bounds = (x_min_c, x_max_c)
                if u_bounds is None:
                    u_bounds = (u_min_c, u_max_c)

    # 2) fall back to system attributes if still None
    if x_bounds is None:
        x_min = getattr(system, "x_min", None) or getattr(system, "xmin", None)
        x_max = getattr(system, "x_max", None) or getattr(system, "xmax", None)
        if x_min is not None and x_max is not None:
            x_bounds = (np.asarray(x_min), np.asarray(x_max))

    if u_bounds is None:
        u_min = getattr(system, "u_min", None) or getattr(system, "umin", None)
        u_max = getattr(system, "u_max", None) or getattr(system, "umax", None)
        if u_min is not None and u_max is not None:
            u_bounds = (np.asarray(u_min), np.asarray(u_max))

    # ------------ Determine results directory ------------
    if save_path is None:
        project_root = Path(__file__).resolve().parents[2]
        results_dir = project_root / "results"
    else:
        save_path = Path(save_path)
        if save_path.is_dir():
            results_dir = save_path
        else:
            results_dir = save_path.parent

    results_dir.mkdir(exist_ok=True)

    # =========================
    #  Figure 1: state plots
    # =========================
    fig_x, axes_x = plt.subplots(nx, 1, figsize=(8, 2.5 * nx), sharex=True)
    if nx == 1:
        axes_x = [axes_x]

    fig_x.suptitle("State trajectories")

    for i in range(nx):
        ax = axes_x[i]

        # state trajectory with subscript
        ax.plot(t, x_traj[:, i], label=fr"$x_{i}$")

        # reference (if provided) with ref subscript
        if x_ref is not None:
            ax.axhline(
                x_ref[i],
                linestyle="--",
                alpha=0.7,
                label=fr"$x_{{\mathrm{{ref}},\,{i}}}$",
            )

        # state bounds (if provided and finite)
        if x_bounds is not None:
            x_min_i = x_bounds[0][i]
            x_max_i = x_bounds[1][i]

            if np.isfinite(x_min_i):
                ax.axhline(
                    x_min_i,
                    linestyle="--",
                    linewidth=1.5,
                    color="k",
                    alpha=0.3,
                )
            if np.isfinite(x_max_i):
                ax.axhline(
                    x_max_i,
                    linestyle="--",
                    linewidth=1.5,
                    color="k",
                    alpha=0.3,
                )

        ax.set_ylabel(fr"$x_{i}$")
        ax.grid(True)
        ax.legend(loc="best")

    axes_x[-1].set_xlabel(r"Time $t$ [s]")

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

        ax.step(t_u, u_traj[:, j], where="post", label=fr"$u_{j}$")

        # input bounds (if provided and finite)
        if u_bounds is not None:
            u_min_j = u_bounds[0][j]
            u_max_j = u_bounds[1][j]

            if np.isfinite(u_min_j):
                ax.axhline(
                    u_min_j,
                    linestyle="--",
                    linewidth=1.5,
                    color="k",
                    alpha=0.3,
                )
            if np.isfinite(u_max_j):
                ax.axhline(
                    u_max_j,
                    linestyle="--",
                    linewidth=1.5,
                    color="k",
                    alpha=0.3,
                )

        ax.set_ylabel(fr"$u_{j}$")
        ax.grid(True)
        ax.legend(loc="best")

    axes_u[-1].set_xlabel(r"Time $t$ [s]")

    fig_u.tight_layout(rect=[0, 0.03, 1, 0.95])

    input_path = results_dir / "input_trajectories.png"
    fig_u.savefig(input_path, dpi=150, bbox_inches="tight")
    print(f"[plotter] Saved input figure to {input_path}")

    if show:
        plt.show()
    else:
        plt.close(fig_x)
        plt.close(fig_u)
