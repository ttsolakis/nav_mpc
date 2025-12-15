# nav_mpc/simulation/plotting/plotter.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from models.dynamics import SystemModel


def _resolve_results_dir(save_path: str | Path | None) -> Path:
    """
    Resolve where to save plots.

    If save_path is None:
        <project_root>/results
    If save_path is a directory:
        that directory
    If save_path is a file path:
        its parent directory
    """
    if save_path is None:
        project_root = Path(__file__).resolve().parents[2]
        results_dir = project_root / "results"
    else:
        save_path = Path(save_path)
        results_dir = save_path if save_path.is_dir() else save_path.parent

    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def _resolve_bounds(
    system: SystemModel,
    constraints: object | None,
    x_bounds: Optional[tuple[np.ndarray, np.ndarray]],
    u_bounds: Optional[tuple[np.ndarray, np.ndarray]],
) -> tuple[Optional[tuple[np.ndarray, np.ndarray]], Optional[tuple[np.ndarray, np.ndarray]]]:
    """
    Resolve (x_min, x_max) and (u_min, u_max) bounds.

    Priority:
      1) explicit x_bounds / u_bounds arguments (if provided)
      2) constraints.get_bounds() if available
      3) system.{x_min,x_max} or system.{xmin,xmax} (same for u)
      4) None (no bounds)
    """
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

    return x_bounds, u_bounds


def plot_state_input_trajectories(
    system: SystemModel,
    constraints: object | None,
    dt: float,
    x_traj: list[np.ndarray] | np.ndarray,
    u_traj: list[np.ndarray] | np.ndarray,
    x_ref: np.ndarray | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
    x_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    u_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    """
    Generic plotting for state and input trajectories.

    This version accepts dt + trajectories and builds:
      - x_traj/u_traj as stacked arrays
      - time vector t automatically

    Parameters
    ----------
    system : SystemModel
        Provides state_dim and input_dim (and optionally bounds).
    constraints : object or None
        If not None and has method get_bounds(), it should return
        (x_min, x_max, u_min, u_max). Used unless x_bounds/u_bounds override.
    dt : float
        Sampling time [s]. Used to build the time array.
    x_traj : list of (nx,) or (T, nx)
        Closed-loop state trajectory. If list, will be vstacked.
    u_traj : list of (nu,) or (T-1, nu)
        Closed-loop input trajectory. If list, will be vstacked.
    x_ref : (nx,), optional
        Reference state to plot as horizontal lines (if given).
    save_path : str or Path, optional
        If None, figures are saved to '<project_root>/results/...'.
        If a directory, figures are saved there.
        If a file, figures are saved in its parent directory.
    show : bool
        Whether to show the figures with plt.show().
    x_bounds : (x_min, x_max), optional
        Each is (nx,) array. Overrides bounds from constraints/system.
    u_bounds : (u_min, u_max), optional
        Each is (nu,) array. Overrides bounds from constraints/system.
    """
    nx = system.state_dim
    nu = system.input_dim

    # -------------------------
    # Stack trajectories
    # -------------------------
    x_arr = np.vstack(x_traj) if isinstance(x_traj, list) else np.asarray(x_traj)
    u_arr = np.vstack(u_traj) if isinstance(u_traj, list) else np.asarray(u_traj)

    if x_arr.ndim != 2 or x_arr.shape[1] != nx:
        raise ValueError(f"x_traj must have shape (T,{nx}); got {x_arr.shape}")

    if u_arr.ndim != 2 or u_arr.shape[1] != nu:
        raise ValueError(f"u_traj must have shape (T-1,{nu}); got {u_arr.shape}")

    if u_arr.shape[0] != x_arr.shape[0] - 1:
        raise ValueError(
            f"u_traj must have length T-1. Got T={x_arr.shape[0]} and u_len={u_arr.shape[0]}"
        )

    # -------------------------
    # Build time vector
    # -------------------------
    t = dt * np.arange(x_arr.shape[0], dtype=float)
    t_u = t[:-1]

    # -------------------------
    # Bounds + save dir
    # -------------------------
    x_bounds, u_bounds = _resolve_bounds(system, constraints, x_bounds, u_bounds)
    results_dir = _resolve_results_dir(save_path)

    # =========================
    # Figure 1: states
    # =========================
    fig_x, axes_x = plt.subplots(nx, 1, figsize=(8, 2.5 * nx), sharex=True)
    if nx == 1:
        axes_x = [axes_x]

    fig_x.suptitle("State trajectories")

    for i in range(nx):
        ax = axes_x[i]
        ax.plot(t, x_arr[:, i], label=fr"$x_{i}$")

        if x_ref is not None:
            ax.axhline(
                float(x_ref[i]),
                linestyle="--",
                alpha=0.7,
                label=fr"$x_{{\mathrm{{ref}},\,{i}}}$",
            )

        if x_bounds is not None:
            x_min_i = float(x_bounds[0][i])
            x_max_i = float(x_bounds[1][i])

            if np.isfinite(x_min_i):
                ax.axhline(x_min_i, linestyle="--", linewidth=1.5, color="k", alpha=0.3)
            if np.isfinite(x_max_i):
                ax.axhline(x_max_i, linestyle="--", linewidth=1.5, color="k", alpha=0.3)

        ax.set_ylabel(fr"$x_{i}$")
        ax.grid(True)
        ax.legend(loc="best")

    axes_x[-1].set_xlabel(r"Time $t$ [s]")
    fig_x.tight_layout(rect=[0, 0.03, 1, 0.95])

    state_path = results_dir / "state_trajectories.png"
    fig_x.savefig(state_path, dpi=150, bbox_inches="tight")
    print(f"[plotter] Saved state figure to {state_path}")

    # =========================
    # Figure 2: inputs
    # =========================
    fig_u, axes_u = plt.subplots(nu, 1, figsize=(8, 2.5 * nu), sharex=True)
    if nu == 1:
        axes_u = [axes_u]

    fig_u.suptitle("Input trajectories")

    for j in range(nu):
        ax = axes_u[j]
        ax.step(t_u, u_arr[:, j], where="post", label=fr"$u_{j}$")

        if u_bounds is not None:
            u_min_j = float(u_bounds[0][j])
            u_max_j = float(u_bounds[1][j])

            if np.isfinite(u_min_j):
                ax.axhline(u_min_j, linestyle="--", linewidth=1.5, color="k", alpha=0.3)
            if np.isfinite(u_max_j):
                ax.axhline(u_max_j, linestyle="--", linewidth=1.5, color="k", alpha=0.3)

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
