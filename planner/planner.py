# nav_mpc/planner/planner.py

from typing import Tuple

import numpy as np
from scipy import sparse
from scipy.linalg import expm

from models.dynamics import SystemModel
from objectives.objectives import Objective
from constraints.constraints import SystemConstraints


def discretize_affine(A: np.ndarray,
                      B: np.ndarray,
                      c: np.ndarray,
                      dt: float):
    """
    x_dot = A x + B u + c  →  x_{k+1} = Ad x_k + Bd u_k + cd
    using block-matrix exponential.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    c = np.asarray(c, dtype=float).reshape(-1)

    n = A.shape[0]
    m = B.shape[1]

    M = np.zeros((n + m + 1, n + m + 1), dtype=float)
    M[:n, :n] = A
    M[:n, n:n + m] = B
    M[:n, n + m] = c
    M[n + m, n + m] = 1.0

    Md = expm(M * dt)

    Ad = Md[:n, :n]
    Bd = Md[:n, n:n + m]
    cd = Md[:n, n + m]

    return Ad, Bd, cd


def assemble_ltv_mpc_qp(
    system: SystemModel,
    A_fun,
    B_fun,
    c_fun,
    objective: Objective,
    constraints: SystemConstraints,
    N: int,
    dt: float,
    x_init: np.ndarray,
    x_ref: np.ndarray,
    x_bar_seq: np.ndarray,
    u_bar_seq: np.ndarray,
) -> Tuple[sparse.csc_matrix, np.ndarray, sparse.csc_matrix, np.ndarray, np.ndarray]:
    """
    Build (P, q, A, l, u) for the LTV MPC QP with decision variable

        z = [x0, x1, ..., xN, u0, ..., u_{N-1}]  ∈ R^{(N+1)nx + N nu}.

    The linearization is done around the provided (x_bar_seq, u_bar_seq).

    Parameters
    ----------
    system : SystemModel
    A_fun, B_fun, c_fun : callables
        Lambdified continuous-time linearization from qp_formulation.build_linearized_system.
    objective : Objective
        Provides Q, QN, R.
    constraints : SystemConstraints
        Provides simple box bounds on x and u.
    N : int
        Horizon length.
    dt : float
        Discretization step.
    x_init : np.ndarray, shape (nx,)
        Initial state for this QP.
    x_ref : np.ndarray, shape (nx,)
        Reference state used in the cost.
    x_bar_seq : array-like, shape (N+1, nx)
        Linearization points for the state (only first N used for dynamics).
    u_bar_seq : array-like, shape (N, nu)
        Linearization points for the input.

    Returns
    -------
    P : csc_matrix
    q : np.ndarray
    A : csc_matrix
    l : np.ndarray
    u : np.ndarray
    """
    n = system.state_dim
    m = system.input_dim
    nx, nu = n, m

    x_bar_seq = np.asarray(x_bar_seq, dtype=float)
    u_bar_seq = np.asarray(u_bar_seq, dtype=float)
    x_init = np.asarray(x_init, dtype=float).reshape(nx)
    x_ref  = np.asarray(x_ref,  dtype=float).reshape(nx)

    # ----------------------------
    # 1) Discretize along horizon
    # ----------------------------
    Ad_list, Bd_list, cd_list = [], [], []
    for k in range(N):
        x_bar = x_bar_seq[k]
        u_bar = u_bar_seq[k]

        args = list(x_bar) + list(u_bar)

        A_k = np.array(A_fun(*args), dtype=float)
        B_k = np.array(B_fun(*args), dtype=float)
        c_k = np.array(c_fun(*args), dtype=float).reshape(-1)

        Ad_k, Bd_k, cd_k = discretize_affine(A_k, B_k, c_k, dt)

        Ad_list.append(Ad_k)
        Bd_list.append(Bd_k)
        cd_list.append(cd_k)

    # ----------------------------
    # 2) Equality constraints
    # ----------------------------
    Nz = (N + 1) * nx + N * nu
    n_eq = (N + 1) * nx

    Aeq = sparse.lil_matrix((n_eq, Nz), dtype=float)
    leq = np.zeros(n_eq, dtype=float)
    ueq = np.zeros(n_eq, dtype=float)

    # Initial condition x0 = x_init
    Aeq[0:nx, 0:nx] = sparse.eye(nx)
    leq[0:nx] = x_init
    ueq[0:nx] = x_init

    # Dynamics:
    #   x_{k+1} - Ad_k x_k - Bd_k u_k = cd_k
    for k in range(N):
        row_start = (k + 1) * nx
        row_end   = row_start + nx

        col_xk_start   = k * nx
        col_xk_end     = col_xk_start + nx
        col_xkp1_start = (k + 1) * nx
        col_xkp1_end   = col_xkp1_start + nx
        col_uk_start   = (N + 1) * nx + k * nu
        col_uk_end     = col_uk_start + nu

        Ad_k = Ad_list[k]
        Bd_k = Bd_list[k]
        cd_k = cd_list[k]

        Aeq[row_start:row_end, col_xk_start:col_xk_end] = -Ad_k
        Aeq[row_start:row_end, col_xkp1_start:col_xkp1_end] = sparse.eye(nx)
        Aeq[row_start:row_end, col_uk_start:col_uk_end] = -Bd_k

        leq[row_start:row_end] = cd_k
        ueq[row_start:row_end] = cd_k

    Aeq = Aeq.tocsc()

    # ----------------------------
    # 3) Box constraints
    # ----------------------------
    xmin, xmax, umin, umax = constraints.get_bounds()

    Aineq = sparse.eye(Nz, format="csc")

    x_lower_all = np.kron(np.ones(N + 1), xmin)
    x_upper_all = np.kron(np.ones(N + 1), xmax)
    u_lower_all = np.kron(np.ones(N), umin)
    u_upper_all = np.kron(np.ones(N), umax)

    lineq = np.hstack([x_lower_all, u_lower_all])
    uineq = np.hstack([x_upper_all, u_upper_all])

    # Stack equalities + inequalities
    A = sparse.vstack([Aeq, Aineq], format="csc")
    l = np.hstack([leq, lineq])
    u = np.hstack([ueq, uineq])

    # ----------------------------
    # 4) Objective P, q
    # ----------------------------
    Q, QN, R = objective.get_weights()

    Q_block = sparse.kron(sparse.eye(N), sparse.csc_matrix(Q))
    QN_block = sparse.csc_matrix(QN)
    R_block = sparse.kron(sparse.eye(N), sparse.csc_matrix(R))

    P = sparse.block_diag([Q_block, QN_block, R_block], format="csc")

    q_x = np.hstack([
        np.kron(np.ones(N), -Q @ x_ref),
        -QN @ x_ref
    ])
    q_u = np.zeros(N * nu)
    q = np.hstack([q_x, q_u])

    return P, q, A, l, u
