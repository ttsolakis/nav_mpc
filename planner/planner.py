# nav_mpc/planner.py

import numpy as np
from scipy import sparse


def shift_state_sequence(X: np.ndarray) -> np.ndarray:
    """
    Given previous optimal state sequence X of shape (N+1, nx),
    build a shifted linearization sequence X_bar of shape (N+1, nx):

        X_bar[k]   = X[k+1]          for k = 0..N-1   (shift forward)
        X_bar[N]   = 2*X[N] - X[N-1] (linear extrapolation for the last one)
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

        U_bar[k]   = U[k+1]   for k = 0..N-2
        U_bar[N-1] = U[N-1]   (hold last input)
    """
    U = np.asarray(U)
    U_bar = np.empty_like(U)

    if U.shape[0] == 0:
        return U  # degenerate case

    # shift all but last
    U_bar[:-1, :] = U[1:, :]

    # keep last input
    U_bar[-1, :] = U[-1, :]

    return U_bar


def pack_args(x0: np.ndarray, x_bar_seq: np.ndarray, u_bar_seq: np.ndarray, N: int) -> list:
    """
    Build argument list in the SAME order as used in build_linear_*:

      [x0_0,...,x0_{nx-1},
       xbar0_0,...,xbar0_{nx-1}, ubar0_0,...,ubar0_{nu-1},
       ...,
       xbar{N-1}_0,...,xbar{N-1}_{nx-1}, ubar{N-1}_0,...,ubar{N-1}_{nu-1}]
    """
    nx = x0.shape[0]
    nu = u_bar_seq.shape[1] if N > 0 else 0

    args = []

    # x0
    args.extend(np.asarray(x0).reshape(nx))

    # stages 0..N-1
    for k in range(N):
        args.extend(np.asarray(x_bar_seq[k]).reshape(nx))
        args.extend(np.asarray(u_bar_seq[k]).reshape(nu))

    return args


def evaluate_and_update_qp(
    prob,
    x: np.ndarray,
    X: np.ndarray,
    U: np.ndarray,
    N: int,
    A_fun,
    l_fun,
    u_fun,
    P_fun,
    q_fun,
):
    """
    Use the previous optimal sequences (X, U) and the current state x
    to build new linearization points (x̄, ū), re-evaluate QP data,
    and update the OSQP problem in-place.
    """

    # 1) Build new initial condition + shifted linearization sequences
    x_init = x.copy()
    x_bar_seq = shift_state_sequence(X)   # shape (N+1, nx)
    u_bar_seq = shift_input_sequence(U)   # shape (N,   U.shape[1])

    # 2) Pack args in correct order
    args = pack_args(x_init, x_bar_seq, u_bar_seq, N)

    # 3) Re-evaluate QP matrices/vectors
    A_new = np.array(A_fun(*args), dtype=float)
    A_new = sparse.csc_matrix(A_new)
    l_new = np.array(l_fun(*args), dtype=float).reshape(-1)
    u_new = np.array(u_fun(*args), dtype=float).reshape(-1)
    P_new = P_fun(*args)             # sparse.csc_matrix
    q_new = q_fun(*args).reshape(-1)

    # 4) Update OSQP problem
    prob.update(
        Px=sparse.triu(P_new).data,
        Ax=A_new.data,
        q=q_new,
        l=l_new,
        u=u_new,
    )
