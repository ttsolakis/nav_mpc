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


def pack_args(x0: np.ndarray, x_bar_seq: np.ndarray, u_bar_seq: np.ndarray, N: int) -> np.ndarray:
    """
    Flatten (x0, X_bar, U_bar) into a 1D array in the SAME order used
    in the SymPy constructions:

      [x0_0,...,x0_{nx-1},
       xbar0_0,...,xbar0_{nx-1}, ubar0_0,...,ubar0_{nu-1},
       ...,
       xbar{N-1}_0,...,xbar{N-1}_{nx-1}, ubar{N-1}_0,...,ubar{N-1}_{nu-1}]
    """
    nx = x0.shape[0]
    nu = u_bar_seq.shape[1] if N > 0 else 0

    total_len = nx + N * (nx + nu)
    theta = np.empty(total_len, dtype=float)

    idx = 0

    # x0
    for j in range(nx):
        theta[idx] = float(x0[j])
        idx += 1

    # stages
    for k in range(N):
        # x_bar_seq[k, :]
        for j in range(nx):
            theta[idx] = float(x_bar_seq[k, j])
            idx += 1
        # u_bar_seq[k, :]
        for j in range(nu):
            theta[idx] = float(u_bar_seq[k, j])
            idx += 1

    return theta


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
    Build full QP matrices (P, q, A, l, u) for a given (x0, X̄, Ū),
    and precompute the sparsity patterns of A and P (upper triangle)
    for fast updates later.
    """
    theta = pack_args(x0, X_bar, U_bar, N)

    # ---- A: dense once, sparse CSC for OSQP setup ----
    A_dense = np.array(A_fun(theta), dtype=float)
    A = sparse.csc_matrix(A_dense)

    l = np.array(l_fun(theta), dtype=float).reshape(-1)
    u = np.array(u_fun(theta), dtype=float).reshape(-1)

    # ---- P: first evaluation (P_fun returns sparse) ----
    P_raw0 = P_fun(theta)
    if sparse.isspmatrix(P_raw0):
        P_full0 = P_raw0.tocsc()
    else:
        P_dense0 = np.array(P_raw0, dtype=float)
        P_full0 = sparse.csc_matrix(P_dense0)

    P0 = sparse.triu(P_full0).tocsc()

    q = q_fun(theta).reshape(-1)

    # ---- A sparsity pattern (CSC order) ----
    n_cols_A = A.shape[1]
    A_row_idx = A.indices.copy()
    A_col_idx = np.empty_like(A_row_idx)
    for j in range(n_cols_A):
        start = A.indptr[j]
        end = A.indptr[j + 1]
        A_col_idx[start:end] = j

    # ---- P sparsity pattern (upper triangle) ----
    P0_coo = P0.tocoo()
    P_row_idx = P0_coo.row
    P_col_idx = P0_coo.col

    # Return P0 (upper triangular), plus patterns
    return P0, q, A, l, u, A_row_idx, A_col_idx, P_row_idx, P_col_idx


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
    A_row_idx: np.ndarray,
    A_col_idx: np.ndarray,
    P_row_idx: np.ndarray,
    P_col_idx: np.ndarray,
) -> None:
    """
    Re-linearize around the new (x0, x̄, ū), rebuild QP data values and update OSQP.

    The sparsity patterns of A and P are FIXED (as set in set_qp).
    Here we only recompute the numeric values Ax_new, Px_new, q, l, u.
    """
    # Shift linearization sequences
    x_init    = x.copy()
    x_bar_seq = shift_state_sequence(X)  # (N+1, nx)
    u_bar_seq = shift_input_sequence(U)  # (N,   nu)

    # Pack into theta
    theta = pack_args(x_init, x_bar_seq, u_bar_seq, N)

    # --- A: dense, then sample nonzeros in CSC order ---
    A_dense = np.array(A_fun(theta), dtype=float)
    Ax_new = A_dense[A_row_idx, A_col_idx]

    l_new = np.array(l_fun(theta), dtype=float).reshape(-1)
    u_new = np.array(u_fun(theta), dtype=float).reshape(-1)

    P_raw_new = P_fun(theta)
    if sparse.isspmatrix(P_raw_new):
        P_dense_new = P_raw_new.toarray()
    else:
        P_dense_new = np.array(P_raw_new, dtype=float)

    Px_new = P_dense_new[P_row_idx, P_col_idx]

    q_new = q_fun(theta).reshape(-1)

    prob.update(
        Px=Px_new,
        Ax=Ax_new,
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
