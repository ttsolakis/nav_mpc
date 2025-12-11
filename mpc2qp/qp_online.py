# nav_mpc/utils/qp_online.py

import numpy as np
from scipy import sparse
from typing import Callable, Tuple


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


# def pack_args(
#     x0: np.ndarray,
#     x_bar_seq: np.ndarray,
#     u_bar_seq: np.ndarray,
#     N: int,
# ) -> list:
#     """
#     Build argument list in the SAME order as used in build_linear_*:

#       [x0_0,...,x0_{nx-1},
#        xbar0_0,...,xbar0_{nx-1}, ubar0_0,...,ubar0_{nu-1},
#        ...,
#        xbar{N-1}_0,...,xbar{N-1}_{nx-1}, ubar{N-1}_0,...,ubar{N-1}_{nu-1}]
#     """
#     nx = x0.shape[0]
#     nu = u_bar_seq.shape[1] if N > 0 else 0

#     args: list[float] = []

#     # x0
#     args.extend(np.asarray(x0).reshape(nx))

#     # stages 0..N-1
#     for k in range(N):
#         args.extend(np.asarray(x_bar_seq[k]).reshape(nx))
#         args.extend(np.asarray(u_bar_seq[k]).reshape(nu))

#     return args

def pack_args(x0: np.ndarray, x_bar_seq: np.ndarray, u_bar_seq: np.ndarray, N: int) -> list:
    nx = x0.shape[0]
    nu = u_bar_seq.shape[1] if N > 0 else 0

    total_len = nx + N * (nx + nu)
    args = [0.0] * total_len

    idx = 0

    # x0
    for j in range(nx):
        args[idx] = float(x0[j])
        idx += 1

    # stages
    for k in range(N):
        # x_bar_seq[k, :]
        for j in range(nx):
            args[idx] = float(x_bar_seq[k, j])
            idx += 1
        # u_bar_seq[k, :]
        for j in range(nu):
            args[idx] = float(u_bar_seq[k, j])
            idx += 1

    return args


def set_qp(
    x0: np.ndarray,
    X_bar: np.ndarray,
    U_bar: np.ndarray,
    N: int,
    A_fun,
    l_fun,
    u_fun,
    P_fun,
    q_fun,
):
    """
    Build full QP matrices (P, q, A, l, u) for a given (x0, X̄, Ū).

    This is used for the first OSQP setup, and can also be reused
    inside update_qp if you want to avoid duplicated code.
    """
    args = pack_args(x0, X_bar, U_bar, N)

    A = np.array(A_fun(*args), dtype=float)
    A = sparse.csc_matrix(A)

    l = np.array(l_fun(*args), dtype=float).reshape(-1)
    u = np.array(u_fun(*args), dtype=float).reshape(-1)

    P = P_fun(*args)             # assumed sparse.csc_matrix already
    q = q_fun(*args).reshape(-1)

    return P, q, A, l, u


def update_qp(
    prob,
    x: np.ndarray,
    X: np.ndarray,
    U: np.ndarray,
    N: int,
    A_fun: Callable,
    l_fun: Callable,
    u_fun: Callable,
    P_fun: Callable,
    q_fun: Callable,
) -> None:
    """
    Re-linearize around the new (x0, x̄, ū), rebuild QP data and update OSQP.

    - prob   : OSQP instance
    - x      : current state (x0 at this step)
    - X, U   : previous optimal trajectories from the last QP solve
    - N      : horizon length
    - A_fun, l_fun, u_fun, P_fun, q_fun : lambdified QP builders
    """
    # Shift linearization sequences
    x_init = x.copy()
    x_bar_seq = shift_state_sequence(X)  # (N+1, nx)
    u_bar_seq = shift_input_sequence(U)  # (N,   nu)

    # Pack args and evaluate numeric QP matrices/vectors
    args = pack_args(x_init, x_bar_seq, u_bar_seq, N)

    A_new = np.array(A_fun(*args), dtype=float)
    A_new = sparse.csc_matrix(A_new)
    l_new = np.array(l_fun(*args), dtype=float).reshape(-1)
    u_new = np.array(u_fun(*args), dtype=float).reshape(-1)
    P_new = P_fun(*args)                     # sparse.csc_matrix
    q_new = q_fun(*args).reshape(-1)

    # OSQP expects only the upper-triangular data for Px
    prob.update(
        Px=sparse.triu(P_new).data,
        Ax=A_new.data,
        q=q_new,
        l=l_new,
        u=u_new,
    )

def extract_solution(res, nx: int, nu: int, N: int) -> Tuple[np.ndarray, np.ndarray]:
    z = res.x
    X_flat = z[0:(N + 1) * nx]
    U_flat = z[(N + 1) * nx:]
    X = X_flat.reshape(N + 1, nx)
    U = U_flat.reshape(N,     nu)
    return X, U



