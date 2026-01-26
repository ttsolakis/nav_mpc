from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numba import njit
from core.constraints.collision_constraints.halfspace_corridor import compute_collision_halfspaces_horizon_inplace


# ============================================================
# Workspace (preallocated buffers to avoid per-step allocations)
# ============================================================

@dataclass(slots=True)
class QPWorkspace:
    # Constraint updates (CSC data + bounds)
    A_data_new: np.ndarray   # == Ax in OSQP update
    l_new: np.ndarray
    u_new: np.ndarray

    # Objective updates (upper-tri CSC data + linear term)
    P_data_new: np.ndarray   # == Px in OSQP update (upper-tri only)
    q_new: np.ndarray

    # Linearization sequences
    Xbar: np.ndarray
    Ubar: np.ndarray

    # Reference sequence
    Xref: np.ndarray   # (N+1, nx)

    # Stage-wise evaluated dynamics
    Ad_all: np.ndarray
    Bd_all: np.ndarray
    cd_all: np.ndarray

    # Stage-wise evaluated inequality linearization
    Gx_all: np.ndarray
    Gu_all: np.ndarray
    rhs_all: np.ndarray

    # Collision buffers (optional, allocated only if nc_col > 0)
    Acol_xy: np.ndarray | None   # (N, nc_col, 2)
    bcol: np.ndarray | None      # (N, nc_col)
    empty_obstacles_xy: np.ndarray  # shape (0,2)



def make_workspace(
    N: int,
    nx: int,
    nu: int,
    nc_sys: int,
    nc_col: int,
    A_data: np.ndarray,     # qp.A_init.data
    l_init: np.ndarray,     # qp.l_init
    u_init: np.ndarray,     # qp.u_init
    P_data: np.ndarray,     # qp.P_init.data
    q_init: np.ndarray,     # qp.q_init
) -> QPWorkspace:
    """
    Allocate all scratch buffers once.

    Notes:
    - A_data should be qp.A_init.data (CSC data array). We keep a private copy for updates.
    - P_data should be qp.P_init.data (upper-tri CSC data array). We keep a private copy for updates.
    - l_init/u_init and q_init are copied to private arrays that we overwrite each step.
    """
    
    Acol_xy = None
    bcol = None
    if nc_col > 0:
        Acol_xy = np.zeros((N, nc_col, 2), dtype=float)
        bcol = np.full((N, nc_col), 0.0, dtype=float)

    empty_obstacles_xy = np.empty((0,2), dtype=float)

    return QPWorkspace(
        A_data_new=A_data.copy(),
        l_new=l_init.copy(),
        u_new=u_init.copy(),
        P_data_new=P_data.copy(),
        q_new=q_init.copy(),
        Xbar=np.empty((N + 1, nx), dtype=float),
        Ubar=np.empty((N, nu), dtype=float),
        Xref=np.empty((N + 1, nx), dtype=float),
        Ad_all=np.empty((N, nx, nx), dtype=float),
        Bd_all=np.empty((N, nx, nu), dtype=float),
        cd_all=np.empty((N, nx), dtype=float),
        Gx_all=np.empty((N, nc_sys, nx), dtype=float),
        Gu_all=np.empty((N, nc_sys, nu), dtype=float),
        rhs_all=np.empty((N, nc_sys), dtype=float),
        Acol_xy=Acol_xy,
        bcol=bcol,
        empty_obstacles_xy=empty_obstacles_xy,
    )


# ==========================
# Small utility functionality
# ==========================

def shift_state_sequence_inplace(X: np.ndarray, Xbar_out: np.ndarray) -> None:
    """
    X: (N+1, nx)  -> Xbar_out: (N+1, nx)
    Xbar[k] = X[k+1] for k=0..N-1
    Xbar[N] = 2*X[N] - X[N-1]
    """
    Xbar_out[:-1, :] = X[1:, :]
    Xbar_out[-1, :] = 2.0 * X[-1, :] - X[-2, :]


def shift_input_sequence_inplace(U: np.ndarray, Ubar_out: np.ndarray) -> None:
    """
    U: (N, nu) -> Ubar_out: (N, nu)
    Ubar[k] = U[k+1] for k=0..N-2
    Ubar[N-1] = U[N-1]
    """
    Ubar_out[:-1, :] = U[1:, :]
    Ubar_out[-1, :] = U[-1, :]


def extract_solution(res, nx: int, nu: int, N: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    OSQP solution z = [x0..xN, u0..u(N-1)].
    Returns X: (N+1, nx), U: (N, nu)
    """
    z = res.x
    X_flat = z[0 : (N + 1) * nx]
    U_flat = z[(N + 1) * nx :]
    X = X_flat.reshape(N + 1, nx)
    U = U_flat.reshape(N, nu)
    return X, U


# ==========================
# Numba fast fill (A,l,u)
# ==========================

@njit(cache=True)
def _fill_qp_arrays_inplace(
    A_data_new: np.ndarray,
    l_new: np.ndarray,
    u_new: np.ndarray,
    A_template: np.ndarray,
    l_template: np.ndarray,
    u_template: np.ndarray,
    idx_Ad: np.ndarray,   # (N, nx, nx)
    idx_Bd: np.ndarray,   # (N, nx, nu)
    idx_Gx: np.ndarray,   # (N, nc_sys, nx)
    idx_Gu: np.ndarray,   # (N, nc_sys, nu)
    x0: np.ndarray,       # (nx,)
    cd_all: np.ndarray,   # (N, nx)
    Ad_all: np.ndarray,   # (N, nx, nx)
    Bd_all: np.ndarray,   # (N, nx, nu)
    Gx_all: np.ndarray,   # (N, nc_sys, nx)
    Gu_all: np.ndarray,   # (N, nc_sys, nu)
    rhs_all: np.ndarray,  # (N, nc_sys)
    nx: int,
    nu: int,
    nc_sys: int,
    nc_col: int,
    N: int,
    n_eq: int,
) -> None:
    """
    Fill A_data_new, l_new, u_new by copying templates and overwriting only the
    time-varying entries (dynamics and inequality linearization).
    """
    A_data_new[:] = A_template
    l_new[:] = l_template
    u_new[:] = u_template

    # Initial condition: x0 = current x
    for i in range(nx):
        l_new[i] = x0[i]
        u_new[i] = x0[i]

    for k in range(N):
        row0 = (k + 1) * nx

        # Equality RHS = cd_k
        for i in range(nx):
            l_new[row0 + i] = cd_all[k, i]
            u_new[row0 + i] = cd_all[k, i]

        # -Ad_k
        for i in range(nx):
            for j in range(nx):
                A_data_new[idx_Ad[k, i, j]] = -Ad_all[k, i, j]

        # -Bd_k
        for i in range(nx):
            for j in range(nu):
                A_data_new[idx_Bd[k, i, j]] = -Bd_all[k, i, j]

        # Inequalities
        nc = nc_sys + nc_col
        ineq_row0 = n_eq + k * nc

        # Gx_k
        for i in range(nc_sys):
            for j in range(nx):
                A_data_new[idx_Gx[k, i, j]] = Gx_all[k, i, j]

        # Gu_k
        for i in range(nc_sys):
            for j in range(nu):
                A_data_new[idx_Gu[k, i, j]] = Gu_all[k, i, j]

        # upper bound = rhs_k
        for i in range(nc_sys):
            u_new[ineq_row0 + i] = rhs_all[k, i]

@njit(cache=True)
def _write_collision_csc_inplace(A_data_new, u_new, idx_Cxy, idx_Ucol, A_xy, b):
    N = A_xy.shape[0]
    M = A_xy.shape[1]
    for k in range(N):
        for j in range(M):
            A_data_new[idx_Cxy[k, j, 0]] = A_xy[k, j, 0]
            A_data_new[idx_Cxy[k, j, 1]] = A_xy[k, j, 1]
            u_new[idx_Ucol[k, j]] = b[k, j]


# ==========================
# Objective update (P,q)
# ==========================

def _fill_objective_inplace(
    P_data_new: np.ndarray,
    q_new: np.ndarray,
    P_template: np.ndarray,
    q_template: np.ndarray,
    idx_Px: np.ndarray,   # (N+1, nx, nx) upper-tri entries valid for j>=i
    idx_Pu: np.ndarray,   # (N,   nu, nu) upper-tri entries valid for j>=i
    Xbar: np.ndarray,     # (N+1, nx)
    Xref: np.ndarray,     # (N+1, nx)
    N: int,
    nx: int,
    nu: int,
    Q: np.ndarray,        # (nx,nx) stage weight on error
    QN: np.ndarray,       # (nx,nx) terminal weight on error
    R: np.ndarray,        # (nu,nu) input weight
    e_fun,
    Ex_fun,
) -> None:
    """
    Build the LTV quadratic objective around Xbar:

      e(x) ≈ e0 + E (x - xbar) = E x + (e0 - E xbar) = E x + b

      0.5 * (E x + b)^T Q (E x + b)
        => P = E^T Q E
           q = E^T Q b

    Terminal uses QN.

    Input blocks are constant: 0.5*(u-u_ref)^T R (u-u_ref) gives Pu=R and qu=-R u_ref
    (qu is already in q_template if set offline).
    """
    # reset from templates
    P_data_new[:] = P_template
    q_new[:] = q_template

    # stage costs for k=0..N-1
    for k in range(N):
        xk = Xbar[k, :]
        rk = Xref[k, :]

        e0 = e_fun(xk, rk)  
        E  = Ex_fun(xk, rk) 

        b = e0 - E @ xk     # (nx,)
        Px = E.T @ Q @ E    # (nx,nx)
        qx = E.T @ Q @ b    # (nx,)

        # write Px upper-tri
        for i in range(nx):
            for j in range(i, nx):
                P_data_new[idx_Px[k, i, j]] = Px[i, j]

        # write qx
        q_new[k * nx : (k + 1) * nx] = qx

    # terminal k=N
    xN = Xbar[N, :]
    rN = Xref[N,:]

    e0 = e_fun(xN, rN)
    E = Ex_fun(xN, rN)

    b = e0 - E @ xN
    Px = E.T @ QN @ E
    qx = E.T @ QN @ b

    for i in range(nx):
        for j in range(i, nx):
            P_data_new[idx_Px[N, i, j]] = Px[i, j]

    q_new[N * nx : (N + 1) * nx] = qx

    # constant input Hessian blocks Pu=R (upper-tri)
    for k in range(N):
        for i in range(nu):
            for j in range(i, nu):
                P_data_new[idx_Pu[k, i, j]] = R[i, j]


# ==========================
# Public fast update function
# ==========================

def update_qp(
    prob,
    x: np.ndarray,
    X: np.ndarray,
    U: np.ndarray,
    qp,
    ws: QPWorkspace,
    Xref_seq: np.ndarray,
    obstacles_xy: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    
    """
    Fast QP update:
      - shifts (X,U) to get (Xbar,Ubar)
      - evaluates stage kernels (Ad,Bd,cd,Gx,Gu,g)
      - computes rhs for inequality linearization
      - fills A,l,u in-place (numba)
      - fills P,q in-place
      - pushes updates to OSQP
    """

    # Normalize obstacles
    obstacles_xy = ws.empty_obstacles_xy if obstacles_xy is None else obstacles_xy
    collisions_enabled = (qp.nc_col > 0)

    N = U.shape[0]
    nx = X.shape[1]
    nu = U.shape[1]
    nc_sys = qp.nc_sys
    nc_col = qp.nc_col
    n_eq = (N + 1) * nx

    # 0) Set reference sequence
    ws.Xref[:, :] = Xref_seq

    # 1) Shift linearization sequences
    shift_state_sequence_inplace(X, ws.Xbar)
    shift_input_sequence_inplace(U, ws.Ubar)

    # 2) Stage-wise evaluation (compiled kernels) + rhs computation from (Gx,Gu,g)
    for k in range(N):
        xk = ws.Xbar[k, :]
        uk = ws.Ubar[k, :]

        ws.Ad_all[k, :, :] = qp.Ad_fun(xk, uk)
        ws.Bd_all[k, :, :] = qp.Bd_fun(xk, uk)
        ws.cd_all[k, :] = qp.cd_fun(xk, uk)

        if nc_sys > 0:
            ws.Gx_all[k, :, :] = qp.Gx_fun(xk, uk)
            ws.Gu_all[k, :, :] = qp.Gu_fun(xk, uk)

            g0 = qp.g_fun(xk, uk)  # (nc_sys,)
            ws.rhs_all[k, :] = -(g0 - ws.Gx_all[k] @ xk - ws.Gu_all[k] @ uk)

    # 3) Fill A,l,u in-place (numba)
    _fill_qp_arrays_inplace(
        A_data_new=ws.A_data_new,
        l_new=ws.l_new,
        u_new=ws.u_new,
        A_template=qp.A_template,
        l_template=qp.l_template,
        u_template=qp.u_template,
        idx_Ad=qp.idx_Ad,
        idx_Bd=qp.idx_Bd,
        idx_Gx=qp.idx_Gx,
        idx_Gu=qp.idx_Gu,
        x0=x,
        cd_all=ws.cd_all,
        Ad_all=ws.Ad_all,
        Bd_all=ws.Bd_all,
        Gx_all=ws.Gx_all,
        Gu_all=ws.Gu_all,
        rhs_all=ws.rhs_all,
        nx=nx,
        nu=nu,
        nc_sys=nc_sys,
        nc_col=nc_col,
        N=N,
        n_eq=n_eq,
    )

    # -------------------------------------------------
    # Collision constraints (halfspaces) in-place update
    # -------------------------------------------------
    Axy_out = None
    bcol_out = None

    if collisions_enabled:
        if qp.pos_idx is None:
            raise ValueError("nc_col > 0 but qp.pos_idx is None.")
        if qp.idx_Cxy is None:
            raise ValueError("nc_col > 0 but qp.idx_Cxy not built offline.")
        if qp.idx_Ucol is None:
            raise ValueError("nc_col > 0 but qp.idx_Ucol not built offline.")
        if ws.Acol_xy is None or ws.bcol is None:
            raise ValueError("Workspace collision buffers not allocated but nc_col > 0.")

        ix, iy = qp.pos_idx

        # (recommended) heading index provided by collision config/offline qp
        if getattr(qp, "psi_idx", None) is None:
            raise ValueError("nc_col > 0 but qp.psi_idx is None (needed for angular binning).")
        psi_idx = qp.psi_idx

        # If you want to hardcode instead, replace the 3 lines above with:
        # psi_idx = 2

        # stage k constrains x_{k+1}; since Xbar[k] = X[k+1], use Xbar[:N]
        centers_xy = ws.Xbar[:N, [ix, iy]]      # (N,2)
        headings   = ws.Xbar[:N, psi_idx]       # (N,)

        A_xy = ws.Acol_xy
        b = ws.bcol

        compute_collision_halfspaces_horizon_inplace(
            obstacles_xy=obstacles_xy,
            centers_xy=centers_xy,
            headings=headings,
            A_xy_out=A_xy,
            b_out=b,
            M=nc_col,
            rho=qp.r_safe,
            roi=qp.roi,
            b_loose=qp.b_loose,
            eps_norm=qp.eps_norm,
            pick="angular_bins",
        )

        _write_collision_csc_inplace(
            ws.A_data_new,
            ws.u_new,
            qp.idx_Cxy,
            qp.idx_Ucol,
            A_xy,
            b,
        )

        Axy_out = A_xy
        bcol_out = b



    # 4) Fill P,q (Python, small dense ops per stage)
    _fill_objective_inplace(
        P_data_new=ws.P_data_new,
        q_new=ws.q_new,
        P_template=qp.P_template,
        q_template=qp.q_template,
        idx_Px=qp.idx_Px,
        idx_Pu=qp.idx_Pu,
        Xbar=ws.Xbar,
        Xref=ws.Xref,
        N=N,
        nx=nx,
        nu=nu,
        Q=qp.Q,
        QN=qp.QN,
        R=qp.R,
        e_fun=qp.e_fun,
        Ex_fun=qp.Ex_fun,
    )

    # 5) Push updates to OSQP
    prob.update(Px=ws.P_data_new, q=ws.q_new, Ax=ws.A_data_new, l=ws.l_new, u=ws.u_new)

    return (Axy_out, bcol_out)


def solve_qp(
    prob,
    nx: int,
    nu: int,
    N: int,
    embedded: bool,
    time_limit: float,
    step_idx: int,
    debugging: bool = False,
    x: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the already-updated OSQP problem and extract (X, U, u0).

    If embedded=True, applies a time_limit based on dt minus the provided evaluation time.
    """
    if embedded:
        prob.update_settings(time_limit=max(1e-5, time_limit))

    res = prob.solve()

    if res.info.status not in ("solved", "solved inaccurate"):
        raise ValueError(f"OSQP did not solve the problem at step {step_idx}! Status: {res.info.status}")

    X, U = extract_solution(res, nx, nu, N) 
    u0 = U[0]

    if debugging:
        print_solution(step_idx, x, u0, X, U)

    return X, U, u0

def print_solution(
    i: int,
    x: np.ndarray,
    u0: np.ndarray,
    X: np.ndarray,
    U: np.ndarray,
    *,
    precision: int = 3,
    max_rows: int | None = None,
    header: bool = True,
) -> None:
    """
    Pretty-print the MPC solution in the terminal.

    Layout: each line corresponds to an index k.
      - For k = 0..N: prints X[k]
      - For k = 0..N-1: prints U[k] on the same row
      - For k = N: prints U as '-' (no input at terminal state)

    Also prints current sim step i, current state x, and applied input u0.

    Args:
        i: simulation step index
        x: current state (nx,)
        u0: applied input (nu,)
        X: predicted state trajectory (N+1, nx)
        U: predicted input trajectory (N, nu)
        precision: numeric formatting precision
        max_rows: optionally truncate printing to first/last rows if long
        header: print a header block
    """
    x = np.asarray(x).reshape(-1)
    u0 = np.asarray(u0).reshape(-1)
    X = np.asarray(X)
    U = np.asarray(U)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D (N+1, nx), got shape {X.shape}")
    if U.ndim != 2:
        raise ValueError(f"U must be 2D (N, nu), got shape {U.shape}")
    if X.shape[0] != U.shape[0] + 1:
        raise ValueError(f"Expected X rows = U rows + 1, got X{X.shape}, U{U.shape}")

    N = U.shape[0]
    nx = X.shape[1]
    nu = U.shape[1]

    # Determine how to show indices if truncating
    def _row_indices() -> list[int]:
        if max_rows is None or (N + 1) <= max_rows:
            return list(range(N + 1))
        # show first half and last half
        first = max_rows // 2
        last = max_rows - first
        return list(range(first)) + [-1] + list(range((N + 1) - last, N + 1))

    rows = _row_indices()

    # Column labels
    x_cols = [f"x{j}" for j in range(nx)]
    u_cols = [f"u{j}" for j in range(nu)]

    # Compute widths
    k_w = max(2, len(str(N)))
    col_w = max(8, precision + 6)  # enough for sign + decimals

    def fmt_num(v: float) -> str:
        return f"{v: {col_w}.{precision}f}"

    def fmt_dash() -> str:
        return " " * (col_w - 1) + "-"

    if header:
        print("\n" + "=" * 80)
        print(f"MPC solution @ sim step i={i}")
        print(f"current x: {np.array2string(x, precision=precision, floatmode='fixed')}")
        print(f"applied u: {np.array2string(u0, precision=precision, floatmode='fixed')}")
        print("-" * 80)

        # Table header
        k_hdr = "k".rjust(k_w)
        hdr = (
            f"{k_hdr} | "
            + "  ".join(name.rjust(col_w) for name in x_cols)
            + " || "
            + "  ".join(name.rjust(col_w) for name in u_cols)
        )
        print(hdr)
        print("-" * len(hdr))

    for k in rows:
        if k == -1:
            # ellipsis row
            dots_x = "  ".join((" " * (col_w - 3) + "...") for _ in range(nx))
            dots_u = "  ".join((" " * (col_w - 3) + "...") for _ in range(nu))
            print(f"{'..'.rjust(k_w)} | {dots_x} || {dots_u}")
            continue

        xk = X[k]
        x_str = "  ".join(fmt_num(float(v)) for v in xk)

        if k < N:
            uk = U[k]
            u_str = "  ".join(fmt_num(float(v)) for v in uk)
        else:
            u_str = "  ".join(fmt_dash() for _ in range(nu))

        print(f"{str(k).rjust(k_w)} | {x_str} || {u_str}")

    if header:
        print("=" * 80 + "\n")
