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
    args = pack_args(x0, X_bar, U_bar, N)

    # Dense A, then CSC once
    A_dense = np.array(A_fun(*args), dtype=float)
    A = sparse.csc_matrix(A_dense)

    l = np.array(l_fun(*args), dtype=float).reshape(-1)
    u = np.array(u_fun(*args), dtype=float).reshape(-1)

    P = P_fun(*args)             # assumed sparse.csc_matrix already
    q = q_fun(*args).reshape(-1)

    # --- Store A sparsity pattern (in CSC order) ---
    # In CSC:
    #   - data: values
    #   - indices: row indices of each nonzero
    #   - indptr: [start idx of column j]
    #
    # data is ordered column-by-column, so we can reconstruct
    # the col index for each nonzero:
    n_cols = A.shape[1]
    row_idx = A.indices.copy()
    col_idx = np.empty_like(row_idx)

    for j in range(n_cols):
        start = A.indptr[j]
        end   = A.indptr[j + 1]
        col_idx[start:end] = j

    # For convenience, also build pairs we can reuse if we want:
    # (but row_idx, col_idx with CSC order is enough)
    return P, q, A, l, u, row_idx, col_idx

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
) -> None:
    # Shift linearization sequences
    x_init    = x.copy()
    x_bar_seq = shift_state_sequence(X)  # (N+1, nx)
    u_bar_seq = shift_input_sequence(U)  # (N,   nu)

    # Pack args and evaluate numeric QP pieces
    args = pack_args(x_init, x_bar_seq, u_bar_seq, N)

    # Dense A just for values; no more CSC building
    A_dense = np.array(A_fun(*args), dtype=float)

    # Extract only values at nonzero positions (same order as CSC .data)
    Ax_new = A_dense[A_row_idx, A_col_idx]

    l_new = np.array(l_fun(*args), dtype=float).reshape(-1)
    u_new = np.array(u_fun(*args), dtype=float).reshape(-1)

    # P_new is still built fully for now
    P_new = P_fun(*args)                     # sparse.csc_matrix
    Px_new = sparse.triu(P_new).data
    q_new = q_fun(*args).reshape(-1)

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



