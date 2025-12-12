# nav_mpc/mpc2qp/qp_online_fast.py

from __future__ import annotations

import numpy as np
from numba import njit
from typing import Tuple
import time


# from qp_online import shift_state_sequence, shift_input_sequence

def shift_state_sequence_inplace(X: np.ndarray, Xbar_out: np.ndarray) -> None:
    Xbar_out[:-1, :] = X[1:, :]
    Xbar_out[-1, :]  = 2.0 * X[-1, :] - X[-2, :]

def shift_input_sequence_inplace(U: np.ndarray, Ubar_out: np.ndarray) -> None:
    Ubar_out[:-1, :] = U[1:, :]
    Ubar_out[-1, :]  = U[-1, :]

def extract_solution(res, nx: int, nu: int, N: int) -> Tuple[np.ndarray, np.ndarray]:
    z = res.x
    X_flat = z[0:(N + 1) * nx]
    U_flat = z[(N + 1) * nx:]
    X = X_flat.reshape(N + 1, nx)
    U = U_flat.reshape(N,     nu)
    return X, U

def shift_state_sequence(X: np.ndarray) -> np.ndarray:
    """
    Given previous optimal state sequence X of shape (N+1, nx),
    build a shifted linearization sequence X_bar of shape (N+1, nx):

        X_bar[k] = X[k+1]          for k = 0..N-1   (shift forward)
        X_bar[N] = 2*X[N] - X[N-1] (linear extrapolation for the last one)
    """
    X = np.asarray(X)
    X_bar = np.empty_like(X)

    # shift all but last
    X_bar[:-1, :] = X[1:, :]

    # extrapolate terminal
    X_bar[-1, :] = 2.0 * X[-1, :] - X[-2, :]

    return X_bar


def shift_input_sequence(U: np.ndarray) -> np.ndarray:
    """
    Given previous optimal input sequence U of shape (N, nu),
    build shifted U_bar of shape (N, nu):

        U_bar[k]   = U[k+1]  for k = 0..N-2
        U_bar[N-1] = U[N-1]  (hold last input)
    """
    U = np.asarray(U)
    if U.shape[0] == 0:
        return U

    U_bar = np.empty_like(U)
    U_bar[:-1, :] = U[1:, :]
    U_bar[-1, :] = U[-1, :]
    return U_bar


@njit(cache=True)
def _fill_qp_arrays_inplace(
    Ax_new: np.ndarray,
    l_new: np.ndarray,
    u_new: np.ndarray,
    Ax_template: np.ndarray,
    l_template: np.ndarray,
    u_template: np.ndarray,
    idx_Ad: np.ndarray,
    idx_Bd: np.ndarray,
    x0: np.ndarray,
    cd_all: np.ndarray,
    Ad_all: np.ndarray,
    Bd_all: np.ndarray,
    nx: int,
    N: int,
    n_eq: int,
):
    """
    Fill Ax_new, l_new, u_new in-place:
      - start from templates
      - overwrite equality RHS (x0 and cd)
      - overwrite Ax entries for -Ad and -Bd blocks
    """

    # Copy templates
    Ax_new[:] = Ax_template
    l_new[:] = l_template
    u_new[:] = u_template

    # Initial condition RHS: first nx equality rows
    for i in range(nx):
        l_new[i] = x0[i]
        u_new[i] = x0[i]

    # Dynamics RHS and matrix blocks
    # Equality rows for dynamics start at row = nx (but our equations start at row (k+1)*nx)
    for k in range(N):
        row0 = (k + 1) * nx

        # RHS = cd_k
        for i in range(nx):
            l_new[row0 + i] = cd_all[k, i]
            u_new[row0 + i] = cd_all[k, i]

        # Matrix entries: -Ad_k on x_k
        for i in range(nx):
            for j in range(nx):
                Ax_new[idx_Ad[k, i, j]] = -Ad_all[k, i, j]

        # Matrix entries: -Bd_k on u_k
        # nu assumed small; in your case nu=1
        for i in range(nx):
            Ax_new[idx_Bd[k, i, 0]] = -Bd_all[k, i, 0]


def update_qp_fast(
    prob,
    x: np.ndarray,
    X: np.ndarray,
    U: np.ndarray,
    # preallocated outputs:
    Ax_new: np.ndarray,
    l_new: np.ndarray,
    u_new: np.ndarray,
    Xbar: np.ndarray,
    Ubar: np.ndarray,
    Ad_all: np.ndarray,
    Bd_all: np.ndarray,
    cd_all: np.ndarray,
    # templates + maps:
    Ax_template: np.ndarray,
    l_template: np.ndarray,
    u_template: np.ndarray,
    idx_Ad: np.ndarray,
    idx_Bd: np.ndarray,
    # stage kernels:
    Ad_fun,
    Bd_fun,
    cd_fun,
    N: int,
    nx: int,
    nu: int,
    # optional timing:
    timing: dict | None = None,
):
    """
    Fast QP update:
      - shift (X,U) -> (Xbar,Ubar)
      - compute Ad,Bd,cd per stage (small)
      - numba-fill Ax,l,u
      - prob.update(Ax=..., l=..., u=...)
    """

    t0 = time.perf_counter()

    # Shift linearization sequences
    shift_state_sequence_inplace(X, Xbar)  # (N+1, nx)
    shift_input_sequence_inplace(U, Ubar)  # (N, nu)

    t1 = time.perf_counter()

    for k in range(N):
        xk = Xbar[k, :]
        uk = Ubar[k, :]
        Ad_all[k, :, :] = Ad_fun(xk, uk)
        Bd_all[k, :, :] = Bd_fun(xk, uk)
        cd_all[k, :] = cd_fun(xk, uk)

    n_eq = (N + 1) * nx

    t2 = time.perf_counter()

    _fill_qp_arrays_inplace(
        Ax_new=Ax_new,
        l_new=l_new,
        u_new=u_new,
        Ax_template=Ax_template,
        l_template=l_template,
        u_template=u_template,
        idx_Ad=idx_Ad,
        idx_Bd=idx_Bd,
        x0=x,
        cd_all=cd_all,
        Ad_all=Ad_all,
        Bd_all=Bd_all,
        nx=nx,
        N=N,
        n_eq=n_eq
    )

    t3 = time.perf_counter()

    # Only update what actually changes
    prob.update(Ax=Ax_new, l=l_new, u=u_new)

    t4 = time.perf_counter()

    if timing is not None:
        timing["shift"]      += (t1 - t0)
        timing["linearize"]  += (t2 - t1)
        timing["fill"]       += (t3 - t2)
        timing["osqp_update"]+= (t4 - t3)
