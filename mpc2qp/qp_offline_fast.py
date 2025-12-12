# nav_mpc/mpc2qp/qp_offline_fast.py

from __future__ import annotations

import numpy as np
import sympy as sp
from scipy import sparse
from typing import Callable, Tuple

from sympy.utilities.autowrap import autowrap

from models.dynamics import SystemModel
from constraints.sys_constraints import SystemConstraints
from objectives.objectives import Objective


def _build_stage_discretization_funs(
    system: SystemModel,
    dt: float,
    use_autowrap: bool = True,
) -> Tuple[Callable, Callable, Callable]:
    """
    Build *stage-wise* numeric callables:

      Ad_fun(x, u) -> (nx, nx)
      Bd_fun(x, u) -> (nx, nu)
      cd_fun(x, u) -> (nx,)

    using exact symbolic Jacobians, but only for ONE stage (small matrices),
    not the whole horizon.

    If use_autowrap=True, compile tiny C/Cython wrappers for Ad,Bd,cd evaluation.
    """

    nx = system.state_dim
    nu = system.input_dim

    x_sym = system.state_symbolic()     # (nx, 1)
    u_sym = system.input_symbolic()     # (nu, 1)
    f_sym = system.dynamics_symbolic()  # (nx, 1)

    # Continuous-time linearization
    A_ct = f_sym.jacobian(x_sym)                # (nx, nx)
    B_ct = f_sym.jacobian(u_sym)                # (nx, nu)
    c_ct = f_sym - A_ct * x_sym - B_ct * u_sym  # (nx, 1)

    # 2nd-order Taylor discretization
    I = sp.eye(nx)
    A2 = A_ct * A_ct

    Ad = I + dt * A_ct + (dt**2) * A2 / 2
    Bd = dt * B_ct + (dt**2) * (A_ct * B_ct) / 2
    cd = dt * c_ct + (dt**2) * (A_ct * c_ct) / 2

    # Argument ordering for stage kernels:
    # core(*x, *u)
    x_args = list(x_sym)
    u_args = list(u_sym)
    args = x_args + u_args

    if use_autowrap:
        # Tiny compiled kernels (fast calls)
        Ad_core = autowrap(Ad, args=args, backend="cython")
        Bd_core = autowrap(Bd, args=args, backend="cython")
        cd_core = autowrap(cd, args=args, backend="cython")

        def Ad_fun(x: np.ndarray, u: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float).reshape(-1)
            u = np.asarray(u, dtype=float).reshape(-1)
            return np.asarray(Ad_core(*x, *u), dtype=float)

        def Bd_fun(x: np.ndarray, u: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float).reshape(-1)
            u = np.asarray(u, dtype=float).reshape(-1)
            return np.asarray(Bd_core(*x, *u), dtype=float)

        def cd_fun(x: np.ndarray, u: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float).reshape(-1)
            u = np.asarray(u, dtype=float).reshape(-1)
            return np.asarray(cd_core(*x, *u), dtype=float).reshape(-1)

    else:
        # Pure NumPy lambdify (no compilation, slower runtime)
        # Keep as (x_list, u_list) to avoid scalar-arg explosion
        Ad_core = sp.lambdify([x_args, u_args], Ad, "numpy")
        Bd_core = sp.lambdify([x_args, u_args], Bd, "numpy")
        cd_core = sp.lambdify([x_args, u_args], cd, "numpy")

        def Ad_fun(x: np.ndarray, u: np.ndarray) -> np.ndarray:
            return np.asarray(Ad_core(list(x), list(u)), dtype=float)

        def Bd_fun(x: np.ndarray, u: np.ndarray) -> np.ndarray:
            return np.asarray(Bd_core(list(x), list(u)), dtype=float)

        def cd_fun(x: np.ndarray, u: np.ndarray) -> np.ndarray:
            return np.asarray(cd_core(list(x), list(u)), dtype=float).reshape(-1)

    return Ad_fun, Bd_fun, cd_fun


def _csc_position(A_csc: sparse.csc_matrix, row: int, col: int) -> int:
    """
    Return the index p such that A_csc.data[p] corresponds to entry (row, col).
    Assumes the entry exists in the sparsity pattern.
    """
    start = A_csc.indptr[col]
    end = A_csc.indptr[col + 1]
    rows = A_csc.indices[start:end]
    # linear search in this column (columns are short here)
    for offset, r in enumerate(rows):
        if r == row:
            return start + offset
    raise KeyError(f"Entry ({row},{col}) not found in CSC pattern.")


def build_qp_structures_fast(
    system: SystemModel,
    objective: Objective,
    constraints: SystemConstraints,
    N: int,
    dt: float,
    use_autowrap: bool = True,
):
    """
    Build OSQP matrices for setup plus fast-update templates and index maps.

    Returns
    -------
    P0 : csc upper-triangular (constant)
    q0 : (n_z,) (constant)
    A  : csc (constant sparsity, values are initial)
    l0 : (m,) initial bounds
    u0 : (m,) initial bounds

    Ax_template, l_template, u_template : baseline arrays to copy every step
    idx_Ad : (N, nx, nx) indices into Ax for the -Ad_k block entries
    idx_Bd : (N, nx, nu) indices into Ax for the -Bd_k block entries

    Ad_fun, Bd_fun, cd_fun : stage functions to compute Ad,Bd,cd at (x̄_k,ū_k)
    """

    nx = system.state_dim
    nu = system.input_dim
    nc = constraints.constraints_dim

    n_z = (N + 1) * nx + N * nu
    n_eq = (N + 1) * nx
    n_ineq = N * nc
    m = n_eq + n_ineq

    # -----------------------------
    # Build constant objective (P,q)
    # -----------------------------
    Q = objective.Q
    QN = objective.QN
    R = objective.R

    x_ref = objective.x_ref.reshape(nx)
    u_ref = objective.u_ref.reshape(nu)

    if N > 0:
        P_state = sparse.kron(sparse.eye(N), Q)
        P_x = sparse.block_diag([P_state, QN], format="csc")
        P_u = sparse.kron(sparse.eye(N), R)
        P_full = sparse.block_diag([P_x, P_u], format="csc")
    else:
        P_full = sparse.csc_matrix(QN)

    P0 = sparse.triu(P_full).tocsc()

    q_x_stage = -Q @ x_ref
    q_x_all = np.kron(np.ones(N), q_x_stage) if N > 0 else np.zeros(0)
    q_x_term = -QN @ x_ref
    q_u_stage = -R @ u_ref
    q_u_all = np.kron(np.ones(N), q_u_stage) if N > 0 else np.zeros(0)
    q0 = np.hstack([q_x_all, q_x_term, q_u_all]).reshape(-1)

    # ---------------------------------------------
    # Build sparse A with FULL pattern (incl -Ad/-Bd)
    # ---------------------------------------------
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    # Equality: initial condition x0 = current x (A has I on first nx rows and cols)
    for i in range(nx):
        rows.append(i)
        cols.append(i)
        data.append(1.0)

    # Equality: dynamics rows for k=0..N-1:
    # x_{k+1} - Ad_k x_k - Bd_k u_k = cd_k
    for k in range(N):
        row0 = (k + 1) * nx
        col_xk = k * nx
        col_xkp1 = (k + 1) * nx
        col_uk = (N + 1) * nx + k * nu

        # +I on x_{k+1}
        for i in range(nx):
            rows.append(row0 + i)
            cols.append(col_xkp1 + i)
            data.append(1.0)

        # -Ad_k on x_k (store zeros now, but in pattern)
        for i in range(nx):
            for j in range(nx):
                rows.append(row0 + i)
                cols.append(col_xk + j)
                data.append(0.0)

        # -Bd_k on u_k (store zeros now)
        for i in range(nx):
            for j in range(nu):
                rows.append(row0 + i)
                cols.append(col_uk + j)
                data.append(0.0)

    # Inequalities: torque bounds only (nc==2)
    assert nc == 2, "This fast builder assumes nc=2 torque bounds like your current constraints."

    u_min = float(constraints.u_min[0])
    u_max = float(constraints.u_max[0])

    for k in range(N):
        ineq_row0 = n_eq + k * nc
        col_uk = (N + 1) * nx + k * nu

        # u_k <= u_max
        rows.append(ineq_row0 + 0)
        cols.append(col_uk + 0)
        data.append(1.0)

        # -u_k <= -u_min   (equivalent to u_k >= u_min)
        rows.append(ineq_row0 + 1)
        cols.append(col_uk + 0)
        data.append(-1.0)

    A_coo = sparse.coo_matrix((data, (rows, cols)), shape=(m, n_z))
    A = A_coo.tocsc()  # keep stored zeros!

    # ---------------------------------------
    # Build l/u templates (constant structure)
    # ---------------------------------------
    l_template = np.empty(m, dtype=float)
    u_template = np.empty(m, dtype=float)

    # equality bounds: l=u= [x0; cd...], filled online
    l_template[:n_eq] = 0.0
    u_template[:n_eq] = 0.0

    for k in range(N):
        r0 = n_eq + k * nc
        l_template[r0 + 0] = -np.inf
        u_template[r0 + 0] = u_max

        l_template[r0 + 1] = -np.inf
        u_template[r0 + 1] = -u_min

    Ax_template = A.data.copy()

    # ---------------------------------------
    # Precompute CSC indices for Ad/Bd blocks
    # ---------------------------------------
    idx_Ad = np.empty((N, nx, nx), dtype=np.int64)
    idx_Bd = np.empty((N, nx, nu), dtype=np.int64)

    for k in range(N):
        row0 = (k + 1) * nx
        col_xk = k * nx
        col_uk = (N + 1) * nx + k * nu

        for i in range(nx):
            for j in range(nx):
                idx_Ad[k, i, j] = _csc_position(A, row0 + i, col_xk + j)

        for i in range(nx):
            for j in range(nu):
                idx_Bd[k, i, j] = _csc_position(A, row0 + i, col_uk + j)

    # -------------------------
    # Stage discretization funs
    # -------------------------
    Ad_fun, Bd_fun, cd_fun = _build_stage_discretization_funs(
        system=system,
        dt=dt,
        use_autowrap=use_autowrap,
    )

    # Initial l/u for setup: start with templates (x0/cd updated online)
    l0 = l_template.copy()
    u0 = u_template.copy()

    return (
        P0, q0, A, l0, u0,
        Ax_template, l_template, u_template,
        idx_Ad, idx_Bd,
        Ad_fun, Bd_fun, cd_fun
    )
