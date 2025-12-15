# nav_mpc/mpc2qp/qp_offline.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import sympy as sp
from scipy import sparse
from sympy.utilities.autowrap import autowrap

from models.dynamics import SystemModel
from objectives.objectives import Objective
from constraints.sys_constraints import SystemConstraints

@dataclass(frozen=True, slots=True)
class QPStructuresFast:

    # OSQP base problem
    P0: sparse.csc_matrix
    q0: np.ndarray
    A:  sparse.csc_matrix
    l0: np.ndarray
    u0: np.ndarray

    # Templates copied each MPC step (for fast overwrite)
    Ax_template: np.ndarray
    l_template:  np.ndarray
    u_template:  np.ndarray

    # Index maps into A.data for fast filling
    idx_Ad: np.ndarray   # (N, nx, nx)
    idx_Bd: np.ndarray   # (N, nx, nu)
    idx_Gx: np.ndarray   # (N, nc, nx)
    idx_Gu: np.ndarray   # (N, nc, nu)

    # Compiled stage kernels
    Ad_fun:  Callable[[np.ndarray, np.ndarray], np.ndarray]
    Bd_fun:  Callable[[np.ndarray, np.ndarray], np.ndarray]
    cd_fun:  Callable[[np.ndarray, np.ndarray], np.ndarray]
    Gx_fun:  Callable[[np.ndarray, np.ndarray], np.ndarray]
    Gu_fun:  Callable[[np.ndarray, np.ndarray], np.ndarray]
    rhs_fun: Callable[[np.ndarray, np.ndarray], np.ndarray]

def _build_discretized_linearized_dynamics(system: SystemModel, dt: float) -> Tuple[Callable, Callable, Callable]:
    """
    Build *stage-wise* numeric callables:

      Ad_fun(x, u) -> (nx, nx)
      Bd_fun(x, u) -> (nx, nu)
      cd_fun(x, u) -> (nx, )

    using exact symbolic Jacobians, but only for ONE stage (small matrices), not the whole horizon.

    Compile tiny C/Cython wrappers for Ad,Bd,cd evaluation.

    Mathematically:

    Dynamics:
    ẋ = f(x,u)

    Linearize symbolically:
    ẋ = A_ct x + B_ct u + c_ct
    where:
    A_ct = ∂f/∂x
    B_ct = ∂f/∂u
    c_ct = f - A_ct x - B_ct u
    Discretize with 2nd-order Taylor:
    Ad = I + dt A_ct + (dt^2)/2 A_ct^2
    Bd = dt B_ct + (dt^2)/2 A_ct B_ct
    cd = dt c_ct + (dt^2)/2 A_ct c_ct

    Compile fast callables for Ad, Bd, cd at given (x,u).
    
    """

    nx = system.state_dim
    nu = system.input_dim

    x_sym = system.state_symbolic()     # (nx, 1)
    u_sym = system.input_symbolic()     # (nu, 1)
    f_sym = system.dynamics_symbolic()  # (nx, 1)

    # Continuous-time linearization
    A_ct = f_sym.jacobian(x_sym)                # (nx, nx)
    B_ct = f_sym.jacobian(u_sym)                # (nx, nu)
    c_ct = f_sym - A_ct * x_sym - B_ct * u_sym  # (nx,  1)

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


    return Ad_fun, Bd_fun, cd_fun

def _build_linearized_inequality_kernels(system: SystemModel, constraints: SystemConstraints):
    """
    Build compiled stage kernels for inequality constraints g(x,u)<=0.

    Returns:
      Gx_fun(x,u) -> (nc, nx)
      Gu_fun(x,u) -> (nc, nu)
      rhs_fun(x,u)-> (nc,)  with rhs = -(g - Gx*x - Gu*u) evaluated at (x,u)
    """
    x_sym = system.state_symbolic()    # (nx,1)
    u_sym = system.input_symbolic()    # (nu,1)

    g = constraints.constraints_symbolic()      # (nc,1)
    Gx = g.jacobian(x_sym)                      # (nc,nx)
    Gu = g.jacobian(u_sym)                      # (nc,nu)

    # rhs(x,u) = -(g(x,u) - Gx(x,u)*x - Gu(x,u)*u)
    rhs = -(g - Gx * x_sym - Gu * u_sym)        # (nc,1)

    args = list(x_sym) + list(u_sym)

    Gx_core  = autowrap(Gx,  args=args, backend="cython")
    Gu_core  = autowrap(Gu,  args=args, backend="cython")
    rhs_core = autowrap(rhs, args=args, backend="cython")

    def Gx_fun(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        u = np.asarray(u, dtype=float).reshape(-1)
        return np.asarray(Gx_core(*x, *u), dtype=float)

    def Gu_fun(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        u = np.asarray(u, dtype=float).reshape(-1)
        return np.asarray(Gu_core(*x, *u), dtype=float)

    def rhs_fun(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        u = np.asarray(u, dtype=float).reshape(-1)
        return np.asarray(rhs_core(*x, *u), dtype=float).reshape(-1)

    return Gx_fun, Gu_fun, rhs_fun

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

def build_qp(system: SystemModel, objective: Objective, constraints: SystemConstraints, N: int, dt: float):
    """
    This function is the “offline factory” that builds everything OSQP needs once, 
    plus the “maps” that let you update only the time-varying pieces fast online:
    fast-update templates and index maps.

    QP (OSQP form):

    minimize_z   0.5 * zᵀ P_k(x̄,ū) z + q_k(x̄,ū)ᵀ z
    subject to   l_k(x̄,ū) ≤ A_k(x̄,ū) z ≤ u_k(x̄,ū)

    Decision vector:

    z = [x₀, x₁, …, x_N, u₀, u₁, …, u_{N−1}]ᵀ ∈ ℝ^{(N+1)nx + Nnu}

    1) Objective function:
    
          [ Q_0   0     0    ...    0    |  0     0    ...    0    ]  
          [  0   Q_1    0    ...    0    |  0     0    ...    0    ]
          [  0    0    Q_2   ...    0    |  0     0    ...    0    ]
          [ ...  ...   ...   ...   ...   | ...   ...   ...   ...   ]
    P_k = [  0    0     0    ...   Q_{N} |  0     0    ...    0    ]  ∈ ℝ^{((N+1)×nx + N×nu)×((N+1)×nx + N×nu)}
          [ -----------------------------+------------------------ ]
          [  0    0     0    ...    0    | R_1    0    ...    0    ]
          [  0    0     0    ...    0    |  0    R_2   ...    0    ]
          [ ...  ...   ...   ...   ...   | ...   ...   ...   ...   ]
          [  0    0     0    ...    0    |  0     0    ... R_{N-1} ]

    q_k = [-Q_0*x_r -Q_1*x_r ... -Q_{N-1}*x_r -Q_N*x_r  0 0 ... 0  ]^T ∈ ℝ^{((N+1)×nx + N×nu) × 1}
                
    2)Dynamics (equality constraints):

    For each stage k = 0…N−1 we enforce the linearized/discretized dynamics:

    x_{k+1} − A_k x_k − B_k u_k = c_k

          [  I     0    0     ...     0 |   0     0     ...       0   ]
          [-A_0    I    0     ...     0 | -B_0    0     ...       0   ]
    Aeq = [  0   -A_1   I     ...     0 |   0    B_1    ...       0   ] ∈ ℝ^{((N+1)×nx)×((N+1)×nx + N×nu)}
          [ ...                         |  ...                        ]  
          [  0     0   ...  -A_{N-1}  I |   0     0     ...   -B_{N-1}] 

    leq = ueq = [x0 c_0 c_1 c_2 ... c_{N_1}]^T ∈ ℝ^{((N+1)×nx + N×nu)× 1}

    3) Inequality constraints (general nonlinear g(x,u) ≤ 0, linearized online)

    User provides stage constraints as a symbolic vector:

        g(x_k, u_k) ≤ 0,   g ∈ ℝ^{nc}

    Online (in update_qp_fast), we linearize around (x̄_k, ū_k):

        g(x,u) ≈ g₀ + Gx (x − x̄) + Gu (u − ū) ≤ 0

    Rearrange into an affine inequality in (x,u):

        Gx_k x + Gu_k u ≤ rhs_k

    where

        Gx_k  = ∂g/∂x |_(x̄_k,ū_k)
        Gu_k  = ∂g/∂u |_(x̄_k,ū_k)
        rhs_k = −( g(x̄_k,ū_k) − Gx_k x̄_k − Gu_k ū_k )

    OSQP uses l ≤ A z ≤ u, so we encode each stage inequality as:

        (Gx_k)x_k + (Gu_k)u_k ≤ rhs_k
        lineq = −∞,   uineq = rhs_k

    4) Total constraints:

    A = [ Aeq ]
        [Aineq]

    l = [ leq ]
        [lineq]

    u = [ ueq ]
        [uineq]


    Offline (THIS function) we:
    - insert zeros in A for the full sparsity pattern of the blocks (Gx_k, Gu_k) for every stage k, so A has constant sparsity.
    - build l_template/u_template with l = −∞ and a dummy u placeholder (0.0) which is overwritten online.
    - precompute CSC indices (idx_Gx, idx_Gu) so we can fill A.data fast online.

    Online we:
    - overwrite the stored zeros in A.data at idx_Gx/idx_Gu with the current linearization values
    - overwrite u for the inequality rows with rhs_k

    Returns
    -------
    P0 : CSC upper-triangular objective matrix (currently constant)
    q0 : objective linear term (currently constant)
    A  : CSC constraint matrix with constant sparsity (values initialized)
    l0, u0 : initial bounds

    Ax_template, l_template, u_template :
        Baseline arrays copied each MPC step before overwriting time-varying entries.

    idx_Ad, idx_Bd :
        CSC indices into A.data for the −A_d,k and −B_d,k dynamics blocks.

    idx_Gx, idx_Gu :
        CSC indices into A.data for the inequality linearization blocks (Gx_k, Gu_k).

    Ad_fun, Bd_fun, cd_fun :
        compiled stage dynamics kernels (evaluate A_d, B_d, c_d at a given (x,u)).

    Gx_fun, Gu_fun, rhs_fun :
        compiled stage constraint kernels (evaluate Gx, Gu, rhs at a given (x,u)).
    """

    nx = system.state_dim
    nu = system.input_dim
    nc = constraints.constraints_dim

    n_z = (N + 1) * nx + N * nu
    n_eq = (N + 1) * nx
    n_ineq = N * nc
    m = n_eq + n_ineq

    # --------------------------------------------------------------------------------------------------------
    # Build constant objective (P,q). For now constant later will be updated to time-varying P_k(x̄,ū), q_k(x̄,ū)
    # --------------------------------------------------------------------------------------------------------
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

    # -----------------------------------------------
    # Build sparse A with FULL pattern (incl -Ad/-Bd)
    # -----------------------------------------------
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    # Equality constraints (Aeq, leq, ueq)
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

    # Inequality constraints (Aineq, lineq, uineq)
    # Inequality constraints: g(x_k,u_k) <= 0  linearized online
    for k in range(N):
        ineq_row0 = n_eq + k * nc
        col_xk = k * nx
        col_uk = (N + 1) * nx + k * nu

        # placeholders for Gx_k (nc x nx)
        for i in range(nc):
            for j in range(nx):
                rows.append(ineq_row0 + i)
                cols.append(col_xk + j)
                data.append(0.0)

        # placeholders for Gu_k (nc x nu)
        for i in range(nc):
            for j in range(nu):
                rows.append(ineq_row0 + i)
                cols.append(col_uk + j)
                data.append(0.0)

    A_coo = sparse.coo_matrix((data, (rows, cols)), shape=(m, n_z))  # A sparse matrix in COOrdinate format.
    A = A_coo.tocsc()  # keep stored zeros!

    # ---------------------------------------
    # Build l/u templates (constant structure)
    # ---------------------------------------
    l_template = np.empty(m, dtype=float)
    u_template = np.empty(m, dtype=float)

    # equality bounds: l=u= [x0; cd...], filled online
    l_template[:n_eq] = 0.0
    u_template[:n_eq] = 0.0

    # inequality bounds: filled online
    l_template[n_eq:] = -np.inf
    u_template[n_eq:] = 0.0     # placeholder; overwritten online with rhs

    Ax_template = A.data.copy()

    # -----------------------------------------------
    # Precompute CSC indices for Ad,Bd, Gx, Gu blocks
    # -----------------------------------------------
    idx_Ad = np.empty((N, nx, nx), dtype=np.int64)
    idx_Bd = np.empty((N, nx, nu), dtype=np.int64)
    idx_Gx = np.empty((N, nc, nx), dtype=np.int64)
    idx_Gu = np.empty((N, nc, nu), dtype=np.int64)

    for k in range(N):
        
        col_xk = k * nx
        col_uk = (N + 1) * nx + k * nu
        
        # dynamics blocks
        row0 = (k + 1) * nx
        for i in range(nx):
            for j in range(nx):
                idx_Ad[k, i, j] = _csc_position(A, row0 + i, col_xk + j)

        for i in range(nx):
            for j in range(nu):
                idx_Bd[k, i, j] = _csc_position(A, row0 + i, col_uk + j)

        # inequality blocks
        ineq_row0 = n_eq + k * nc
        for i in range(nc):
            for j in range(nx):
                idx_Gx[k, i, j] = _csc_position(A, ineq_row0 + i, col_xk + j)

        for i in range(nc):
            for j in range(nu):
                idx_Gu[k, i, j] = _csc_position(A, ineq_row0 + i, col_uk + j)


    # -----------------------------------------
    # Linearized/discretized dynamics per stage
    # -----------------------------------------
    Ad_fun, Bd_fun, cd_fun = _build_discretized_linearized_dynamics(system=system, dt=dt)

    # -----------------------------------------
    # Linearized/discretized dynamics per stage
    # -----------------------------------------

    Gx_fun, Gu_fun, rhs_fun = _build_linearized_inequality_kernels(system, constraints)

    # Initial l/u for setup: start with templates (x0/cd updated online)
    l0 = l_template.copy()
    u0 = u_template.copy()

    return QPStructuresFast(
        P0=P0, q0=q0, A=A, l0=l0, u0=u0,
        Ax_template=Ax_template, l_template=l_template, u_template=u_template,
        idx_Ad=idx_Ad, idx_Bd=idx_Bd, idx_Gx=idx_Gx, idx_Gu=idx_Gu,
        Ad_fun=Ad_fun, Bd_fun=Bd_fun, cd_fun=cd_fun,
        Gx_fun=Gx_fun, Gu_fun=Gu_fun, rhs_fun=rhs_fun,
    )