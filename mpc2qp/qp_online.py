from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numba import njit
from constraints.collision_constraints.halfspace_corridor import compute_collision_halfspaces_horizon

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


def make_workspace(
    N: int,
    nx: int,
    nu: int,
    nc_sys: int,
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
) -> np.ndarray | None:
    """
    Fast QP update:
      - shifts (X,U) to get (Xbar,Ubar)
      - evaluates stage kernels (Ad,Bd,cd,Gx,Gu,g)
      - computes rhs for inequality linearization
      - fills A,l,u in-place (numba)
      - fills P,q in-place
      - pushes updates to OSQP
    """
    N = U.shape[0]
    nx = X.shape[1]
    nu = U.shape[1]
    nc_sys = qp.nc_sys
    nc_col = qp.nc_col
    nc_total = nc_sys + nc_col
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

    Axy_out = None
    bcol_out = None

    if nc_col > 0:
        if obstacles_xy is None:
            raise ValueError("nc_col > 0 but obstacles_xy is None.")
        if qp.pos_idx is None:
            raise ValueError("nc_col > 0 but qp.pos_idx is None.")
        if qp.idx_Cx is None or qp.idx_Cy is None:
            raise ValueError("nc_col > 0 but qp.idx_Cx/idx_Cy not built offline.")

        ix, iy = qp.pos_idx

        # stage k constrains x_{k+1}; since Xbar[k] = X[k+1], use Xbar[:N]
        centers_xy = ws.Xbar[:N, [ix, iy]]  # (N,2) view is fine

        A_xy, b = compute_collision_halfspaces_horizon(
            obstacles_xy=obstacles_xy,
            centers_xy=centers_xy,
            M=nc_col,     
            rho=qp.r_safe,
            roi=qp.roi,
            b_loose=qp.b_loose,
            pick="closest",
        )

        # Write collision A coefficients into CSC data
        for k in range(N):
            for j in range(nc_col):
                ws.A_data_new[qp.idx_Cx[k, j]] = A_xy[k, j, 0]
                ws.A_data_new[qp.idx_Cy[k, j]] = A_xy[k, j, 1]

            # Write collision bounds into u
            coll_row0 = n_eq + k * nc_total + nc_sys
            ws.u_new[coll_row0 : coll_row0 + nc_col] = b[k, :]

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
