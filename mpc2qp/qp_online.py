# nav_mpc/mpc2qp/qp_online.py

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
) -> QPWorkspace:
    """
    Allocate all scratch buffers once.

    Notes:
    - A_data should be qp.A.data (CSC data array). We keep a private copy for updates.
    - l0/u0 are copied to private arrays that we overwrite each step.
    """
    return QPWorkspace(
        Ax_new=A_data.copy(),
        l_new=l0.copy(),
        u_new=u0.copy(),
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
# Numba fast fill (in-place)
# ==========================

@njit(cache=True)
def _fill_qp_arrays_inplace(
    Ax_new: np.ndarray,
    l_new: np.ndarray,
    u_new: np.ndarray,
    Ax_template: np.ndarray,
    l_template: np.ndarray,
    u_template: np.ndarray,
    # maps for dynamics:
    idx_Ad: np.ndarray,   # (N, nx, nx)
    idx_Bd: np.ndarray,   # (N, nx, nu)
    # maps for inequalities:
    idx_Gx: np.ndarray,   # (N, nc, nx)
    idx_Gu: np.ndarray,   # (N, nc, nu)
    # rhs data:
    x0: np.ndarray,       # (nx,)
    cd_all: np.ndarray,   # (N, nx)
    Ad_all: np.ndarray,   # (N, nx, nx)
    Bd_all: np.ndarray,   # (N, nx, nu)
    Gx_all: np.ndarray,   # (N, nc, nx)
    Gu_all: np.ndarray,   # (N, nc, nu)
    rhs_all: np.ndarray,  # (N, nc)
    # dims:
    nx: int,
    nu: int,
    nc: int,
    N: int,
    n_eq: int,
) -> None:
    """
    Fill Ax_new, l_new, u_new in-place:
      - start from templates
      - overwrite equality RHS (x0 and cd)
      - overwrite Ax entries for -Ad and -Bd blocks
      - overwrite Ax entries for Gx and Gu inequality blocks
      - overwrite inequality upper bounds u (rhs)
    """

    # Copy templates
    Ax_new[:] = Ax_template
    l_new[:] = l_template
    u_new[:] = u_template

    # Initial condition RHS: x0 = current x
    for i in range(nx):
        l_new[i] = x0[i]
        u_new[i] = x0[i]

    # Stage-wise dynamics and inequalities
    for k in range(N):
        row0 = (k + 1) * nx

        # Equality RHS = cd_k
        for i in range(nx):
            l_new[row0 + i] = cd_all[k, i]
            u_new[row0 + i] = cd_all[k, i]

        # Equality matrix: -Ad_k on x_k
        for i in range(nx):
            for j in range(nx):
                Ax_new[idx_Ad[k, i, j]] = -Ad_all[k, i, j]

        # Equality matrix: -Bd_k on u_k
        for i in range(nx):
            for j in range(nu):
                Ax_new[idx_Bd[k, i, j]] = -Bd_all[k, i, j]

        # Inequalities: stage rows start at n_eq + k*nc
        ineq_row0 = n_eq + k * nc

        # Fill Gx_k
        for i in range(nc):
            for j in range(nx):
                Ax_new[idx_Gx[k, i, j]] = Gx_all[k, i, j]

        # Fill Gu_k
        for i in range(nc):
            for j in range(nu):
                Ax_new[idx_Gu[k, i, j]] = Gu_all[k, i, j]

        # Bounds: l=-inf already, u=rhs_k
        for i in range(nc):
            u_new[ineq_row0 + i] = rhs_all[k, i]


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
    Fast QP update, minimal signature.

    Inputs:
      prob : osqp.OSQP instance
      x    : current state (nx,)
      X,U  : previous optimal solution sequences to shift (X: (N+1,nx), U: (N,nu))
      qp   : QPStructures dataclass from qp_offline (must expose templates, idx maps, and kernels)
      ws   : QPWorkspace with preallocated buffers

    Steps:
      1) shift (X,U) -> (Xbar,Ubar)
      2) compute Ad,Bd,cd per stage (compiled kernels)
      3) compute Gx,Gu,rhs per stage (compiled kernels)
      4) numba-fill Ax,l,u using index maps
      5) prob.update(Ax=..., l=..., u=...)
    """

    # Dimensions (trust shapes from arrays / qp maps)
    N = U.shape[0]
    nx = X.shape[1]
    nu = U.shape[1]
    nc = qp.idx_Gx.shape[1]
    n_eq = (N + 1) * nx

    # 1) Shift linearization sequences
    shift_state_sequence_inplace(X, ws.Xbar)
    shift_input_sequence_inplace(U, ws.Ubar)

    # 2) Stage-wise evaluation (Python loop; kernels are compiled)
    for k in range(N):
        xk = ws.Xbar[k, :]
        uk = ws.Ubar[k, :]

        ws.Ad_all[k, :, :] = qp.Ad_fun(xk, uk)
        ws.Bd_all[k, :, :] = qp.Bd_fun(xk, uk)
        ws.cd_all[k, :] = qp.cd_fun(xk, uk)

        ws.Gx_all[k, :, :] = qp.Gx_fun(xk, uk)
        ws.Gu_all[k, :, :] = qp.Gu_fun(xk, uk)
        ws.rhs_all[k, :] = qp.rhs_fun(xk, uk)

    # 3) Fill arrays in-place (numba)
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

    # 4) Push updates to OSQP
    prob.update(Ax=ws.Ax_new, l=ws.l_new, u=ws.u_new)
