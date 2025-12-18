from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numba import njit


# ============================================================
# Workspace (preallocated buffers to avoid per-step allocations)
# ============================================================

@dataclass(slots=True)
class QPWorkspace:
    Ax_new: np.ndarray
    l_new: np.ndarray
    u_new: np.ndarray

    # NEW: objective buffers
    P_new: np.ndarray   # OSQP expects upper-triangular CSC data array for P
    q_new: np.ndarray

    Xbar: np.ndarray
    Ubar: np.ndarray

    Ad_all: np.ndarray
    Bd_all: np.ndarray
    cd_all: np.ndarray

    Gx_all: np.ndarray
    Gu_all: np.ndarray
    rhs_all: np.ndarray


def make_workspace(
    N: int,
    nx: int,
    nu: int,
    nc: int,
    A_data: np.ndarray,
    l0: np.ndarray,
    u0: np.ndarray,
    P_data: np.ndarray,
    q0: np.ndarray,
) -> QPWorkspace:
    """
    Allocate all scratch buffers once.

    Notes:
    - A_data should be qp.A.data (CSC data array). We keep a private copy for updates.
    - P_data should be qp.P0.data (upper-tri CSC data array). We keep a private copy for updates.
    - l0/u0 and q0 are copied to private arrays that we overwrite each step.
    """
    return QPWorkspace(
        Ax_new=A_data.copy(),
        l_new=l0.copy(),
        u_new=u0.copy(),
        P_new=P_data.copy(),
        q_new=q0.copy(),
        Xbar=np.empty((N + 1, nx), dtype=float),
        Ubar=np.empty((N, nu), dtype=float),
        Ad_all=np.empty((N, nx, nx), dtype=float),
        Bd_all=np.empty((N, nx, nu), dtype=float),
        cd_all=np.empty((N, nx), dtype=float),
        Gx_all=np.empty((N, nc, nx), dtype=float),
        Gu_all=np.empty((N, nc, nu), dtype=float),
        rhs_all=np.empty((N, nc), dtype=float),
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
    Ax_new: np.ndarray,
    l_new: np.ndarray,
    u_new: np.ndarray,
    Ax_template: np.ndarray,
    l_template: np.ndarray,
    u_template: np.ndarray,
    idx_Ad: np.ndarray,   # (N, nx, nx)
    idx_Bd: np.ndarray,   # (N, nx, nu)
    idx_Gx: np.ndarray,   # (N, nc, nx)
    idx_Gu: np.ndarray,   # (N, nc, nu)
    x0: np.ndarray,       # (nx,)
    cd_all: np.ndarray,   # (N, nx)
    Ad_all: np.ndarray,   # (N, nx, nx)
    Bd_all: np.ndarray,   # (N, nx, nu)
    Gx_all: np.ndarray,   # (N, nc, nx)
    Gu_all: np.ndarray,   # (N, nc, nu)
    rhs_all: np.ndarray,  # (N, nc)
    nx: int,
    nu: int,
    nc: int,
    N: int,
    n_eq: int,
) -> None:

    Ax_new[:] = Ax_template
    l_new[:] = l_template
    u_new[:] = u_template

    # Initial condition RHS: x0 = current x
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
                Ax_new[idx_Ad[k, i, j]] = -Ad_all[k, i, j]

        # -Bd_k
        for i in range(nx):
            for j in range(nu):
                Ax_new[idx_Bd[k, i, j]] = -Bd_all[k, i, j]

        # Inequalities
        ineq_row0 = n_eq + k * nc

        for i in range(nc):
            for j in range(nx):
                Ax_new[idx_Gx[k, i, j]] = Gx_all[k, i, j]

        for i in range(nc):
            for j in range(nu):
                Ax_new[idx_Gu[k, i, j]] = Gu_all[k, i, j]

        for i in range(nc):
            u_new[ineq_row0 + i] = rhs_all[k, i]


# ==========================
# Objective update (P,q)
# ==========================

def _fill_objective_inplace(
    P_new: np.ndarray,
    q_new: np.ndarray,
    P_template: np.ndarray,
    q_template: np.ndarray,
    idx_Px: np.ndarray,   # (N+1, nx, nx) upper-tri entries valid for j>=i
    idx_Pu: np.ndarray,   # (N,   nu, nu) upper-tri entries valid for j>=i
    Xbar: np.ndarray,     # (N+1, nx)
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

      e(x) ≈ e0 + E (x - xbar) = (E x) + (e0 - E xbar)

      0.5 * (E x + b)^T Q (E x + b)
        => P = E^T Q E
           q = E^T Q b     where b = (e0 - E xbar)

    Terminal uses QN.

    Input blocks are constant: 0.5*(u-u_ref)^T R (u-u_ref) gives Pu=R and qu=-R u_ref.
    (qu is already in q_template if you set it offline.)
    """

    # reset from templates
    P_new[:] = P_template
    q_new[:] = q_template

    # stage costs for k=0..N-1
    for k in range(N):
        xk = Xbar[k, :]

        e0 = e_fun(xk)          # (nx,)
        E  = Ex_fun(xk)         # (nx,nx)

        b  = e0 - E @ xk        # (nx,)
        Px = E.T @ Q @ E        # (nx,nx)
        qx = E.T @ Q @ b        # (nx,)

        # write Px upper-tri
        for i in range(nx):
            for j in range(i, nx):
                P_new[idx_Px[k, i, j]] = Px[i, j]

        # write qx
        q_new[k * nx : (k + 1) * nx] = qx

    # terminal k=N
    xN = Xbar[N, :]

    e0 = e_fun(xN)
    E  = Ex_fun(xN)

    b  = e0 - E @ xN
    Px = E.T @ QN @ E
    qx = E.T @ QN @ b

    for i in range(nx):
        for j in range(i, nx):
            P_new[idx_Px[N, i, j]] = Px[i, j]

    q_new[N * nx : (N + 1) * nx] = qx

    # constant input Hessian blocks Pu=R (upper-tri)
    for k in range(N):
        for i in range(nu):
            for j in range(i, nu):
                P_new[idx_Pu[k, i, j]] = R[i, j]


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
) -> None:
    """
    Fast QP update.

    Also updates objective:
      prob.update(Px=..., q=...)
    """

    N = U.shape[0]
    nx = X.shape[1]
    nu = U.shape[1]
    nc = qp.idx_Gx.shape[1]
    n_eq = (N + 1) * nx

    # 1) Shift linearization sequences
    shift_state_sequence_inplace(X, ws.Xbar)
    shift_input_sequence_inplace(U, ws.Ubar)

    # 2) Stage-wise evaluation (compiled kernels)
    for k in range(N):
        xk = ws.Xbar[k, :]
        uk = ws.Ubar[k, :]

        ws.Ad_all[k, :, :] = qp.Ad_fun(xk, uk)
        ws.Bd_all[k, :, :] = qp.Bd_fun(xk, uk)
        ws.cd_all[k, :]    = qp.cd_fun(xk, uk)

        ws.Gx_all[k, :, :] = qp.Gx_fun(xk, uk)
        ws.Gu_all[k, :, :] = qp.Gu_fun(xk, uk)
        ws.rhs_all[k, :]   = qp.rhs_fun(xk, uk)

    # 3) Fill A,l,u in-place (numba)
    _fill_qp_arrays_inplace(
        Ax_new=ws.Ax_new,
        l_new=ws.l_new,
        u_new=ws.u_new,
        Ax_template=qp.Ax_template,
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
        nc=nc,
        N=N,
        n_eq=n_eq,
    )

    # 4) Fill P,q (Python, small dense ops per stage)
    _fill_objective_inplace(
        P_new=ws.P_new,
        q_new=ws.q_new,
        P_template=qp.P_template,
        q_template=qp.q_template,
        idx_Px=qp.idx_Px,
        idx_Pu=qp.idx_Pu,
        Xbar=ws.Xbar,
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
    prob.update(Px=ws.P_new, q=ws.q_new, Ax=ws.Ax_new, l=ws.l_new, u=ws.u_new)
