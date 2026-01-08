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
from constraints.collision_constraints.halfspace_corridor import HalfspaceCorridorCollisionConfig

@dataclass(frozen=True, slots=True)
class QPStructures:
    # OSQP initial problem data (constant sparsity)
    P_init: sparse.csc_matrix
    q_init: np.ndarray
    A_init: sparse.csc_matrix
    l_init: np.ndarray
    u_init: np.ndarray

    # Templates copied each MPC step (for fast overwrite)
    A_template: np.ndarray          # == A_init.data baseline
    l_template: np.ndarray
    u_template: np.ndarray
    P_template: np.ndarray          # == P_init.data baseline (upper-tri only)
    q_template: np.ndarray

    # Weights (convenience)
    Q: np.ndarray
    QN: np.ndarray
    R: np.ndarray

    # Index maps into A.data for fast filling
    idx_Ad: np.ndarray   # (N, nx, nx)
    idx_Bd: np.ndarray   # (N, nx, nu)
    idx_Gx: np.ndarray   # (N, nc_sys, nx)
    idx_Gu: np.ndarray   # (N, nc_sys, nu)

    # Index maps into P.data for fast filling (upper-triangular only)
    idx_Px: np.ndarray   # (N+1, nx, nx) valid only for j>=i
    idx_Pu: np.ndarray   # (N,   nu, nu) valid only for j>=i

    # Compiled stage kernels
    Ad_fun: Callable[[np.ndarray, np.ndarray], np.ndarray]
    Bd_fun: Callable[[np.ndarray, np.ndarray], np.ndarray]
    cd_fun: Callable[[np.ndarray, np.ndarray], np.ndarray]
    Gx_fun: Callable[[np.ndarray, np.ndarray], np.ndarray]
    Gu_fun: Callable[[np.ndarray, np.ndarray], np.ndarray]
    g_fun:  Callable[[np.ndarray, np.ndarray], np.ndarray] 

    # Compiled objective-error kernels
    e_fun: Callable[[np.ndarray, np.ndarray], np.ndarray]
    Ex_fun: Callable[[np.ndarray, np.ndarray], np.ndarray]


    # Collision corridor (optional)
    nc_sys: int
    nc_col: int
    pos_idx: tuple[int, int] | None     # (ix,iy)
    b_loose: float
    r_safe: float
    roi: float
    M_col: int                          # == collision.M (kept for convenience)
    idx_Cx: np.ndarray | None           # (N, nc_col) indices into A_init.data
    idx_Cy: np.ndarray | None           # (N, nc_col) indices into A_init.data


def _build_discretized_linearized_dynamics(system: SystemModel, dt: float) -> Tuple[Callable, Callable, Callable]:
    """
    Build *stage-wise* numeric callables:

      Ad_fun(x, u) -> (nx, nx)
      Bd_fun(x, u) -> (nx, nu)
      cd_fun(x, u) -> (nx, )

    using exact symbolic Jacobians, but only for ONE stage (small matrices), not the whole horizon.

    Compile tiny C/Cython wrappers for Ad, Bd, cd evaluation.

    Mathematically:

    Continuous-time dynamics:
      xdot = f(x,u)

    Linearize (symbolic):
      xdot ≈ A_ct(x̄,ū) x + B_ct(x̄,ū) u + c_ct(x̄,ū)
    where:
      A_ct = ∂f/∂x
      B_ct = ∂f/∂u
      c_ct = f - A_ct x - B_ct u

    Discretize with 2nd-order Taylor (about dt):
      Ad = I + dt A_ct + (dt^2)/2 A_ct^2
      Bd = dt B_ct + (dt^2)/2 A_ct B_ct
      cd = dt c_ct + (dt^2)/2 A_ct c_ct
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

    # Argument ordering for stage kernels: core(*x, *u)
    args = list(x_sym) + list(u_sym)

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
    Build compiled stage kernels for inequality constraints g(x,u) <= 0.

    Returns:
      Gx_fun(x,u) -> (nc, nx)
      Gu_fun(x,u) -> (nc, nu)
      g_fun(x,u)  -> (nc,)

    NOTE:
    Define Gx = ∂g/∂x|_(x̄,ū), Gu = ∂g/∂u|_(x̄,ū)
    Linearization: g(x,u) ≈ g(x̄,ū) + Gx (x - x̄) + Gu (u - ū) <= 0
    Rearranged: Gx x + Gu u <= -(g(x̄,ū) - Gx x̄ - Gu ū)
    Online, you will compute the affine upper bound (rhs) for OSQP as:
      rhs = -( g(x̄,ū) - Gx(x̄,ū) x - Gu(x̄,ū) u )
    and then encode:
      -inf <= Gx x + Gu u <= rhs
    """
    x_sym = system.state_symbolic()  # (nx,1)
    u_sym = system.input_symbolic()  # (nu,1)

    g = constraints.constraints_symbolic()  # (nc,1)
    Gx = g.jacobian(x_sym)                  # (nc,nx)
    Gu = g.jacobian(u_sym)                  # (nc,nu)

    args = list(x_sym) + list(u_sym)

    Gx_core = autowrap(Gx, args=args, backend="cython")
    Gu_core = autowrap(Gu, args=args, backend="cython")
    g_core  = autowrap(g,  args=args, backend="cython")

    def Gx_fun(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        u = np.asarray(u, dtype=float).reshape(-1)
        return np.asarray(Gx_core(*x, *u), dtype=float)

    def Gu_fun(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        u = np.asarray(u, dtype=float).reshape(-1)
        return np.asarray(Gu_core(*x, *u), dtype=float)

    def g_fun(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        u = np.asarray(u, dtype=float).reshape(-1)
        return np.asarray(g_core(*x, *u), dtype=float).reshape(-1)

    return Gx_fun, Gu_fun, g_fun


def _build_linearized_error_kernels(system: SystemModel, objective: Objective):
    """
    Build compiled kernels for nonlinear error e(x, r) and its Jacobian Ex(x, r)=de/dx.
    """
    x_sym = system.state_symbolic()  # (nx,1)
    r_sym = objective.r_sym          # (nx,1)  <-- NEW

    e_sym = objective.state_error_symbolic()  # depends on x and r
    Ex_sym = e_sym.jacobian(x_sym)

    args = list(x_sym) + list(r_sym)

    e_core = autowrap(e_sym, args=args, backend="cython")
    Ex_core = autowrap(Ex_sym, args=args, backend="cython")

    def e_fun(x: np.ndarray, r: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        r = np.asarray(r, dtype=float).reshape(-1)
        return np.asarray(e_core(*x, *r), dtype=float).reshape(-1)

    def Ex_fun(x: np.ndarray, r: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        r = np.asarray(r, dtype=float).reshape(-1)
        return np.asarray(Ex_core(*x, *r), dtype=float)

    return e_fun, Ex_fun


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


def build_qp(system: SystemModel, objective: Objective, constraints: SystemConstraints, N: int, dt: float, collision: HalfspaceCorridorCollisionConfig | None = None) -> QPStructures:
    """
    Offline "factory" that builds OSQP matrices with constant sparsity + index maps
    so that online you can overwrite only time-varying entries in-place.

    OSQP form:
      minimize_z   0.5 * zᵀ P_k z + q_kᵀ z
      subject to   l_k ≤ A_k z ≤ u_k

    Decision vector:
      z = [x₀, x₁, …, x_N, u₀, u₁, …, u_{N−1}]ᵀ ∈ ℝ^{(N+1)nx + Nnu}

    1) Objective (time-varying via error linearization)
    ---------------------------------------------------
    Let e(x) be a nonlinear "state error" (symbolic), with Jacobian E(x)=de/dx.

    Online, linearize around x̄:
      e(x) ≈ e(x̄,ū) + E (x - x̄) = E x + (e(x̄,ū) - E x̄) = E x + be

    Then stage cost:
      0.5 * (E x + be)ᵀ Q (E x + be)
        => P_x = Eᵀ Q E
           q_x = Eᵀ Q be

    Terminal uses QN instead of Q.

    Input cost is standard quadratic around u_ref:
      0.5 * (u - u_ref)ᵀ R (u - u_ref)
        => P_u = R,   q_u = -R u_ref   (constant, can live in q_template)

    We build:
      - P_init: upper-tri CSC with zeros but correct sparsity
      - q_template: baseline q (includes constant input linear term)
      - idx_Px/idx_Pu to overwrite P_init.data fast online

    2) Dynamics (equality constraints)
    --------------------------------
    For k=0..N-1 enforce:
      x_{k+1} - A_d,k x_k - B_d,k u_k = c_d,k

    We build A_init with constant sparsity and placeholders for (-A_d,k), (-B_d,k),
    and build l/u templates where equality bounds are overwritten online.

    3) Inequality constraints (nonlinear g(x,u) <= 0, linearized online)
    -------------------------------------------------------------------
    User provides:
      g(x_k, u_k) <= 0,  g ∈ ℝ^{nc}

    Linearize at (x̄,ū):
      g(x,u) ≈ g(x̄,ū) + Gx (x - x̄) + Gu (u - ū)

    Rearrange:
      Gx x + Gu u <= rhs
    where
      rhs = -(g(x̄,ū) - Gx x̄ - Gu ū )

    OSQP encoding per stage:
      -inf <= Gx x_k + Gu u_k <= rhs_k

    Offline we include placeholders for Gx and Gu in A_init, and store idx_Gx/idx_Gu.

    Returns
    -------
    QPStructures with:
      - *_init: objects to pass to prob.setup(...)
      - *_template: baseline arrays to copy each step before overwriting
      - idx_*: CSC indices into A_init.data / P_init.data for fast overwrite
      - *_fun: compiled kernels for stage-wise evaluation
    """
    nx = system.state_dim
    nu = system.input_dim
    nc_sys = constraints.constraints_dim
    nc_col = int(collision.M) if (collision is not None) else 0
    nc = nc_sys + nc_col

    # ----------------------------
    # Collision metadata
    # ----------------------------
    pos_idx = None
    b_loose = 1e6
    r_safe = 0.0
    roi = 0.0
    M_col = 0

    if nc_col > 0:
        pos_idx = collision.pos_idx
        b_loose = float(collision.b_loose)
        r_safe = float(collision.r_safe)
        roi = float(collision.roi)
        M_col = int(collision.M)

        ix, iy = pos_idx
        if ix == iy:
            raise ValueError("collision.pos_idx must have distinct indices (x,y).")
        if not (0 <= ix < nx and 0 <= iy < nx):
            raise ValueError(f"collision.pos_idx {pos_idx} out of range for nx={nx}.")


    n_z = (N + 1) * nx + N * nu
    n_eq = (N + 1) * nx
    n_ineq = N * nc
    m = n_eq + n_ineq

    # -----------------------------------------------------------------------------
    # Objective skeleton: build P_init (upper-tri, correct sparsity) and q_template.
    # -----------------------------------------------------------------------------
    Q = np.asarray(objective.Q, dtype=float)
    QN = np.asarray(objective.QN, dtype=float)
    R = np.asarray(objective.R, dtype=float)

    u_ref = np.asarray(objective.u_ref, dtype=float).reshape(nu)

    rowsP: list[int] = []
    colsP: list[int] = []
    dataP: list[float] = []

    # State blocks (k=0..N): upper-tri only
    for k in range(N + 1):
        base = k * nx
        for i in range(nx):
            for j in range(i, nx):
                rowsP.append(base + i)
                colsP.append(base + j)
                dataP.append(0.0)

    # Input blocks (k=0..N-1): upper-tri only
    u_offset = (N + 1) * nx
    for k in range(N):
        base = u_offset + k * nu
        for i in range(nu):
            for j in range(i, nu):
                rowsP.append(base + i)
                colsP.append(base + j)
                dataP.append(0.0)

    P_init = sparse.coo_matrix((dataP, (rowsP, colsP)), shape=(n_z, n_z)).tocsc()
    P_template = P_init.data.copy()

    # Baseline q: state part filled online; input part can be constant
    q_template = np.zeros(n_z, dtype=float)
    if N > 0:
        q_u_stage = -R @ u_ref
        q_template[u_offset:] = np.kron(np.ones(N), q_u_stage)

    # For setup, use a valid baseline q (not necessarily "true" yet, but consistent)
    q_init = q_template.copy()

    # Indices into P_init.data (upper-tri only)
    idx_Px = np.full((N + 1, nx, nx), -1, dtype=np.int64)
    idx_Pu = np.full((N, nu, nu), -1, dtype=np.int64)

    for k in range(N + 1):
        base = k * nx
        for i in range(nx):
            for j in range(i, nx):
                idx_Px[k, i, j] = _csc_position(P_init, base + i, base + j)

    for k in range(N):
        base = u_offset + k * nu
        for i in range(nu):
            for j in range(i, nu):
                idx_Pu[k, i, j] = _csc_position(P_init, base + i, base + j)

    # -----------------------------------------------------------------------------
    # Constraint matrix A_init (constant sparsity with stored zeros)
    # -----------------------------------------------------------------------------
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    # Initial condition: x0 = current x  (I on first nx rows/cols)
    for i in range(nx):
        rows.append(i)
        cols.append(i)
        data.append(1.0)

    # Dynamics rows for k=0..N-1:
    # x_{k+1} - Ad_k x_k - Bd_k u_k = cd_k
    for k in range(N):
        row0 = (k + 1) * nx
        col_xk = k * nx
        col_xkp1 = (k + 1) * nx
        col_uk = u_offset + k * nu

        # +I on x_{k+1}
        for i in range(nx):
            rows.append(row0 + i)
            cols.append(col_xkp1 + i)
            data.append(1.0)

        # -Ad_k on x_k (zeros now, fixed sparsity)
        for i in range(nx):
            for j in range(nx):
                rows.append(row0 + i)
                cols.append(col_xk + j)
                data.append(0.0)

        # -Bd_k on u_k (zeros now)
        for i in range(nx):
            for j in range(nu):
                rows.append(row0 + i)
                cols.append(col_uk + j)
                data.append(0.0)

    # Inequality constraints per stage: [system | collision]
    for k in range(N):
        ineq_row0 = n_eq + k * nc
        col_xk = k * nx
        col_uk = u_offset + k * nu

        # ----------------------------
        # System block (nc_sys)
        # ----------------------------
        for i in range(nc_sys):
            for j in range(nx):
                rows.append(ineq_row0 + i)
                cols.append(col_xk + j)
                data.append(0.0)

        for i in range(nc_sys):
            for j in range(nu):
                rows.append(ineq_row0 + i)
                cols.append(col_uk + j)
                data.append(0.0)

        # ----------------------------
        # Collision block (nc_col)
        # Apply to FUTURE states only:
        # stage k -> state x_{k+1}
        # Each collision row will later be:
        #   A_xy[k,j,0] * x_{k+1,ix} + A_xy[k,j,1] * x_{k+1,iy} <= b[k,j]
        # Here we only allocate sparsity with zeros.
        # ----------------------------
        if nc_col > 0:
            assert pos_idx is not None
            ix, iy = pos_idx

            col_xcol = (k + 1) * nx  # SHIFT: x_{k+1}

            for j in range(nc_col):
                r = ineq_row0 + nc_sys + j

                # placeholders (zeros), overwritten online
                rows.append(r); cols.append(col_xcol + ix); data.append(0.0)
                rows.append(r); cols.append(col_xcol + iy); data.append(0.0)



    A_init = sparse.coo_matrix((data, (rows, cols)), shape=(m, n_z)).tocsc()

    # -----------------------------------------------------------------------------
    # Bounds templates + init bounds
    # -----------------------------------------------------------------------------
    l_template = np.empty(m, dtype=float)
    u_template = np.empty(m, dtype=float)

    # equality bounds (filled online): l=u=[x0; cd_0; ...; cd_{N-1}]
    l_template[:n_eq] = 0.0
    u_template[:n_eq] = 0.0

    # inequality bounds: -inf <= (...) <= rhs, with rhs overwritten online
    l_template[n_eq:] = -np.inf
    u_template[n_eq:] = 0.0  # placeholder for system inequalities

    if nc_col > 0:
        # set collision rows to inactive by default
        for k in range(N):
            row0 = n_eq + k * nc + nc_sys
            u_template[row0 : row0 + nc_col] = b_loose

    l_init = l_template.copy()
    u_init = u_template.copy()

    # A_template is the baseline A_init.data
    A_template = A_init.data.copy()

    # -----------------------------------------------------------------------------
    # Index maps into A_init.data
    # -----------------------------------------------------------------------------
    idx_Ad = np.empty((N, nx, nx), dtype=np.int64)
    idx_Bd = np.empty((N, nx, nu), dtype=np.int64)
    idx_Gx = np.empty((N, nc_sys, nx), dtype=np.int64)
    idx_Gu = np.empty((N, nc_sys, nu), dtype=np.int64)
    idx_Cx = None
    idx_Cy = None
    if nc_col > 0:
        idx_Cx = np.empty((N, nc_col), dtype=np.int64)
        idx_Cy = np.empty((N, nc_col), dtype=np.int64)

    for k in range(N):
        col_xk = k * nx
        col_uk = u_offset + k * nu

        # dynamics blocks
        row0 = (k + 1) * nx
        for i in range(nx):
            for j in range(nx):
                idx_Ad[k, i, j] = _csc_position(A_init, row0 + i, col_xk + j)

        for i in range(nx):
            for j in range(nu):
                idx_Bd[k, i, j] = _csc_position(A_init, row0 + i, col_uk + j)

        # inequality blocks
        ineq_row0 = n_eq + k * nc
        for i in range(nc_sys):
            for j in range(nx):
                idx_Gx[k, i, j] = _csc_position(A_init, ineq_row0 + i, col_xk + j)

        for i in range(nc_sys):
            for j in range(nu):
                idx_Gu[k, i, j] = _csc_position(A_init, ineq_row0 + i, col_uk + j)

    if nc_col > 0:
        assert pos_idx is not None
        ix, iy = pos_idx

        for k in range(N):
            ineq_row0 = n_eq + k * nc
            col_xcol = (k + 1) * nx

            for j in range(nc_col):
                r = ineq_row0 + nc_sys + j
                idx_Cx[k, j] = _csc_position(A_init, r, col_xcol + ix)
                idx_Cy[k, j] = _csc_position(A_init, r, col_xcol + iy)

    # -----------------------------------------------------------------------------
    # Compiled kernels
    # -----------------------------------------------------------------------------
    Ad_fun, Bd_fun, cd_fun = _build_discretized_linearized_dynamics(system=system, dt=dt)
    Gx_fun, Gu_fun, g_fun = _build_linearized_inequality_kernels(system, constraints)
    e_fun, Ex_fun = _build_linearized_error_kernels(system, objective)

    return QPStructures(
        P_init=P_init, q_init=q_init, A_init=A_init, l_init=l_init, u_init=u_init,
        P_template=P_template, q_template=q_template, A_template=A_template, l_template=l_template, u_template=u_template,
        Q=Q, QN=QN, R=R,
        idx_Ad=idx_Ad, idx_Bd=idx_Bd, idx_Gx=idx_Gx, idx_Gu=idx_Gu, idx_Px=idx_Px, idx_Pu=idx_Pu,
        Ad_fun=Ad_fun, Bd_fun=Bd_fun, cd_fun=cd_fun, Gx_fun=Gx_fun, Gu_fun=Gu_fun, g_fun=g_fun, e_fun=e_fun, Ex_fun=Ex_fun,
        nc_sys=nc_sys, nc_col=nc_col, pos_idx=pos_idx, b_loose=b_loose, r_safe=r_safe, roi=roi, M_col=M_col, idx_Cx=idx_Cx, idx_Cy=idx_Cy)
