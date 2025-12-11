# nav_mpc/qp_formulation/qp_offline.py

from scipy import sparse
import sympy as sp
import numpy as np

from sympy.utilities.autowrap import autowrap

from models.dynamics import SystemModel
from constraints.sys_constraints import SystemConstraints
from objectives.objectives import Objective

# ---------------------------------------------------------------------
# Theta helpers: single flattened parameter vector
# ---------------------------------------------------------------------
def build_theta_symbols(nx: int, nu: int, N: int):
    """
    Create a flat list of SymPy symbols [theta0, ..., theta{T-1}]
    where T = nx + N*(nx + nu),
    in the SAME order as pack_args() builds the numeric theta.
    """
    total_len = nx + N * (nx + nu)
    theta_syms = sp.symbols(f"theta0:{total_len}", real=True)
    return list(theta_syms)


def theta_views(theta_syms, nx: int, nu: int, N: int):
    """
    Given flat theta_syms, return symbolic views:

      x0_syms    : list of length nx
      xbar_syms  : list of length N, each entry a SymPy Matrix(nx, 1)
      ubar_syms  : list of length N, each entry a SymPy Matrix(nu, 1)

    such that the layout matches pack_args().
    """
    idx = 0

    # x0
    x0_syms = theta_syms[idx:idx + nx]
    idx += nx

    xbar_syms = []
    ubar_syms = []

    for _ in range(N):
        xk_syms = theta_syms[idx:idx + nx]
        idx += nx
        uk_syms = theta_syms[idx:idx + nu]
        idx += nu

        xbar_syms.append(sp.Matrix(xk_syms))
        ubar_syms.append(sp.Matrix(uk_syms))

    return x0_syms, xbar_syms, ubar_syms


# ---------------------------------------------------------------------
# Linear equality constraints (dynamics)
# ---------------------------------------------------------------------
def build_linear_equality_constraints(
    system: SystemModel,
    N: int,
    dt: float,
    use_cython: bool,
    debug: bool = False,
):
    """
    Build callables for dynamics equality constraints of the LTV MPC:

      Aeq(theta) z = leq(theta) = ueq(theta),

    where theta is the flattened [x0, x̄ sequence, ū sequence].
    """

    nx = system.state_dim
    nu = system.input_dim

    # 1) Continuous-time linearized dynamics
    x_sym = system.state_symbolic()     # (nx, 1)
    u_sym = system.input_symbolic()     # (nu, 1)
    f_sym = system.dynamics_symbolic()  # (nx, 1)

    A_ct = f_sym.jacobian(x_sym)                # (nx, nx)
    B_ct = f_sym.jacobian(u_sym)                # (nx, nu)
    c_ct = f_sym - A_ct * x_sym - B_ct * u_sym  # (nx, 1)

    if debug:
        print("A_ct:")
        sp.pprint(A_ct)
        print("B_ct:")
        sp.pprint(B_ct)
        print("c_ct:")
        sp.pprint(c_ct)

    # 2) Discrete-time linearized dynamics (2nd-order Taylor)
    I = sp.eye(nx)
    A2 = A_ct * A_ct
    Ad_generic = I + dt * A_ct + (dt**2) * A2 / 2
    Bd_generic = dt * B_ct + (dt**2) * (A_ct * B_ct) / 2
    cd_generic = dt * c_ct + (dt**2) * (A_ct * c_ct) / 2

    if debug:
        print("Ad_generic:")
        sp.pprint(Ad_generic)
        print("Bd_generic:")
        sp.pprint(Bd_generic)
        print("cd_generic:")
        sp.pprint(cd_generic)

    # 3) Theta symbols and views
    theta_syms = build_theta_symbols(nx, nu, N)
    x0_syms, xbar_syms, ubar_syms = theta_views(theta_syms, nx, nu, N)
    x0_vec = sp.Matrix(x0_syms)

    # 4) Ad_k, Bd_k, cd_k via substitution
    Ad_list = []
    Bd_list = []
    cd_list = []

    for k in range(N):
        subs_k = {
            x_sym[i]: xbar_syms[k][i] for i in range(nx)
        } | {
            u_sym[j]: ubar_syms[k][j] for j in range(nu)
        }

        Ad_k = Ad_generic.subs(subs_k)
        Bd_k = Bd_generic.subs(subs_k)
        cd_k = cd_generic.subs(subs_k)

        Ad_list.append(Ad_k)
        Bd_list.append(Bd_k)
        cd_list.append(cd_k)

    # 5) Build Aeq, leq, ueq symbolically
    n_z  = (N + 1) * nx + N * nu
    n_eq = (N + 1) * nx

    Aeq_sym = sp.zeros(n_eq, n_z)
    leq_sym = sp.zeros(n_eq, 1)
    ueq_sym = sp.zeros(n_eq, 1)

    # Initial condition: x0 = x_init
    for i in range(nx):
        Aeq_sym[i, i] = 1.0
        leq_sym[i, 0] = x0_vec[i]
        ueq_sym[i, 0] = x0_vec[i]

    # Dynamics rows
    for k in range(N):
        row_start = (k + 1) * nx

        col_xk_start   = k * nx
        col_xkp1_start = (k + 1) * nx
        col_uk_start   = (N + 1) * nx + k * nu

        Ad_k = Ad_list[k]
        Bd_k = Bd_list[k]
        cd_k = cd_list[k]

        # +I on x_{k+1}
        for i in range(nx):
            Aeq_sym[row_start + i, col_xkp1_start + i] = 1.0

        # -Ad_k on x_k
        for i in range(nx):
            for j in range(nx):
                Aeq_sym[row_start + i, col_xk_start + j] -= Ad_k[i, j]

        # -Bd_k on u_k
        for i in range(nx):
            for j in range(nu):
                Aeq_sym[row_start + i, col_uk_start + j] -= Bd_k[i, j]

        # RHS = cd_k
        for i in range(nx):
            leq_sym[row_start + i, 0] = cd_k[i]
            ueq_sym[row_start + i, 0] = cd_k[i]

    # 6) Backend: autowrap (Cython) or lambdify (pure NumPy)
    if use_cython:
        # autowrap expects scalar arguments, so we’ll call core(*theta)
        Aeq_core = autowrap(Aeq_sym, args=theta_syms, backend="cython")
        leq_core = autowrap(leq_sym, args=theta_syms, backend="cython")
        ueq_core = autowrap(ueq_sym, args=theta_syms, backend="cython")

        def Aeq_fun(theta: np.ndarray):
            return np.asarray(Aeq_core(*theta), dtype=float)

        def leq_fun(theta: np.ndarray):
            return np.asarray(leq_core(*theta), dtype=float)

        def ueq_fun(theta: np.ndarray):
            return np.asarray(ueq_core(*theta), dtype=float)

    else:
        # Pure Python/NumPy path, much faster to start (no compilation)
        Aeq_core = sp.lambdify([theta_syms], Aeq_sym, "numpy")
        leq_core = sp.lambdify([theta_syms], leq_sym, "numpy")
        ueq_core = sp.lambdify([theta_syms], ueq_sym, "numpy")

        def Aeq_fun(theta: np.ndarray):
            return np.asarray(Aeq_core(theta), dtype=float)

        def leq_fun(theta: np.ndarray):
            return np.asarray(leq_core(theta), dtype=float)

        def ueq_fun(theta: np.ndarray):
            return np.asarray(ueq_core(theta), dtype=float)

    return Aeq_fun, leq_fun, ueq_fun


# ---------------------------------------------------------------------
# Linear inequality constraints (g(x,u) <= 0 linearized)
# ---------------------------------------------------------------------
def build_linear_inequality_constraints(
    system: SystemModel,
    constraints: SystemConstraints,
    N: int,
    use_cython: bool,
    debug: bool = False,
):
    """
    Build lambdified / autowrapped functions for the linearized inequalities.
    """

    nx = system.state_dim
    nu = system.input_dim

    # 1) Symbolic variables and nonlinear constraints
    x_sym = system.state_symbolic()      # (nx, 1)
    u_sym = system.input_symbolic()      # (nu, 1)

    g_sym = constraints.constraints_symbolic()   # (nc, 1)
    nc = g_sym.shape[0]

    Hx_sym = g_sym.jacobian(x_sym)      # (nc, nx)
    Hu_sym = g_sym.jacobian(u_sym)      # (nc, nu)

    if debug:
        print("g_sym(x,u):")
        sp.pprint(g_sym)
        print("Hx_sym = dg/dx:")
        sp.pprint(Hx_sym)
        print("Hu_sym = dg/du:")
        sp.pprint(Hu_sym)

    # 2) Theta symbols and views
    theta_syms = build_theta_symbols(nx, nu, N)
    _, xbar_syms, ubar_syms = theta_views(theta_syms, nx, nu, N)

    # 3) Stage-wise Hx_k, Hu_k, h_k
    Hx_list = []
    Hu_list = []
    h_list  = []

    for k in range(N):
        subs_k = {
            x_sym[i]: xbar_syms[k][i] for i in range(nx)
        } | {
            u_sym[j]: ubar_syms[k][j] for j in range(nu)
        }

        Hx_k = Hx_sym.subs(subs_k)      # (nc, nx)
        Hu_k = Hu_sym.subs(subs_k)      # (nc, nu)
        g_k  = g_sym.subs(subs_k)       # (nc, 1)

        # h_k = -g(x̄_k,ū_k) + Hx_k x̄_k + Hu_k ū_k
        h_k  = -g_k + Hx_k * xbar_syms[k] + Hu_k * ubar_syms[k]   # (nc, 1)

        Hx_list.append(Hx_k)
        Hu_list.append(Hu_k)
        h_list.append(h_k)

        if debug:
            print(f"\nStage k = {k}:")
            print("Hx_k:")
            sp.pprint(Hx_k)
            print("Hu_k:")
            sp.pprint(Hu_k)
            print("g_k:")
            sp.pprint(g_k)
            print("h_k:")
            sp.pprint(h_k)

    # 4) Build stacked Aineq, lineq, uineq
    n_z     = (N + 1) * nx + N * nu
    n_ineq  = N * nc

    Aineq_sym = sp.zeros(n_ineq, n_z)
    lineq_sym = sp.zeros(n_ineq, 1)
    uineq_sym = sp.zeros(n_ineq, 1)

    for k in range(N):
        row_start    = k * nc
        col_xk_start = k * nx
        col_uk_start = (N + 1) * nx + k * nu

        Hx_k = Hx_list[k]
        Hu_k = Hu_list[k]
        h_k  = h_list[k]

        for i in range(nc):
            for j in range(nx):
                Aineq_sym[row_start + i, col_xk_start + j] = Hx_k[i, j]

        for i in range(nc):
            for j in range(nu):
                Aineq_sym[row_start + i, col_uk_start + j] = Hu_k[i, j]

        for i in range(nc):
            uineq_sym[row_start + i, 0] = h_k[i, 0]

        # Lower bound = -∞ (no lower constraint from g(x,u) <= 0)
        for i in range(nc):
            lineq_sym[row_start + i, 0] = -sp.oo

    if debug:
        print("\nAineq_sym:")
        sp.pprint(Aineq_sym)
        print("lineq_sym:")
        sp.pprint(lineq_sym)
        print("uineq_sym:")
        sp.pprint(uineq_sym)

    # 5) Backend choice
    if use_cython:
        Aineq_core = autowrap(Aineq_sym, args=theta_syms, backend="cython")
        # lineq is all -inf: we *don't* need to compile it; we handle it in Python.
        uineq_core = autowrap(uineq_sym, args=theta_syms, backend="cython")

        def Aineq_fun(theta: np.ndarray):
            return np.asarray(Aineq_core(*theta), dtype=float)

        def lineq_fun(theta: np.ndarray):
            # ignored theta, constant -inf; kept for interface symmetry
            n_ineq_local = n_ineq
            return -np.inf * np.ones(n_ineq_local)

        def uineq_fun(theta: np.ndarray):
            return np.asarray(uineq_core(*theta), dtype=float).reshape(-1)

    else:
        Aineq_core = sp.lambdify([theta_syms], Aineq_sym, "numpy")
        uineq_core = sp.lambdify([theta_syms], uineq_sym, "numpy")
        n_ineq_local = n_ineq
        lineq_const = -np.inf * np.ones(n_ineq_local)

        def Aineq_fun(theta: np.ndarray):
            return np.asarray(Aineq_core(theta), dtype=float)

        def lineq_fun(theta: np.ndarray):
            return lineq_const

        def uineq_fun(theta: np.ndarray):
            return np.asarray(uineq_core(theta), dtype=float).reshape(-1)

    return Aineq_fun, lineq_fun, uineq_fun


# ---------------------------------------------------------------------
# Combined linear constraints: A, l, u
# ---------------------------------------------------------------------
def build_linear_constraints(
    system: SystemModel,
    constraints: SystemConstraints,
    N: int,
    dt: float,
    use_cython: bool,
    debug: bool = False,
):
    """
    Wrap equality + inequality lambdified/compiled functions into combined
    A_fun(theta), l_fun(theta), u_fun(theta).
    """

    Aeq_fun, leq_fun, ueq_fun = build_linear_equality_constraints(
        system, N, dt, use_cython=use_cython, debug=debug
    )
    Aineq_fun, lineq_fun, uineq_fun = build_linear_inequality_constraints(
        system, constraints, N, use_cython=use_cython, debug=debug
    )

    def A_fun(theta: np.ndarray):
        Aeq_num   = Aeq_fun(theta)
        Aineq_num = Aineq_fun(theta)
        A = np.vstack([Aeq_num, Aineq_num])

        if debug:
            print("Aeq_num shape:", Aeq_num.shape)
            print("Aineq_num shape:", Aineq_num.shape)
            print("A (stacked) shape:", A.shape)

        return A

    def l_fun(theta: np.ndarray):
        leq_num   = leq_fun(theta).reshape(-1)
        lineq_num = lineq_fun(theta).reshape(-1)
        l = np.hstack([leq_num, lineq_num])

        if debug:
            print("leq_num shape:", leq_num.shape)
            print("lineq_num shape:", lineq_num.shape)
            print("l (stacked) shape:", l.shape)

        return l

    def u_fun(theta: np.ndarray):
        ueq_num   = ueq_fun(theta).reshape(-1)
        uineq_num = uineq_fun(theta).reshape(-1)
        u = np.hstack([ueq_num, uineq_num])

        if debug:
            print("ueq_num shape:", ueq_num.shape)
            print("uineq_num shape:", uineq_num.shape)
            print("u (stacked) shape:", u.shape)

        return u

    return A_fun, l_fun, u_fun


# ---------------------------------------------------------------------
# Quadratic objective (kept Pythonic, parametric-ready)
# ---------------------------------------------------------------------
def build_quadratic_objective(
    system: SystemModel,
    objective: Objective,
    N: int,
    debug: bool = False,
):
    """
    Build callables P_fun(theta), q_fun(theta) for OSQP:

        0.5 * z^T P z + q^T z

    For now P and q are constant (do NOT depend on theta) but the interface
    is parametric so you can extend it later.
    """

    nx = system.state_dim
    nu = system.input_dim

    Q  = objective.Q   # (nx, nx)
    QN = objective.QN  # (nx, nx)
    R  = objective.R   # (nu, nu)

    x_ref = objective.x_ref.reshape(nx)
    u_ref = objective.u_ref.reshape(nu)

    # 1) P as block-diagonal
    if N > 0:
        P_state = sparse.kron(sparse.eye(N), Q)
        P_term  = QN
        P_x = sparse.block_diag([P_state, P_term], format="csc")
    else:
        P_x = sparse.csc_matrix(QN)

    if N > 0:
        P_u = sparse.kron(sparse.eye(N), R)
    else:
        P_u = sparse.csc_matrix((0, 0))

    P = sparse.block_diag([P_x, P_u], format="csc")

    if debug:
        print("P_x shape:", P_x.shape)
        print("P_u shape:", P_u.shape)
        print("P (full) shape:", P.shape)

    # 2) q for shifted quadratic
    q_x_stage = -Q @ x_ref
    if N > 0:
        q_x_all_stages = np.kron(np.ones(N), q_x_stage)
    else:
        q_x_all_stages = np.zeros(0)

    q_x_term = -QN @ x_ref

    q_u_stage = -R @ u_ref
    if N > 0:
        q_u_all = np.kron(np.ones(N), q_u_stage)
    else:
        q_u_all = np.zeros(0)

    q = np.hstack([q_x_all_stages, q_x_term, q_u_all])

    if debug:
        print("q_x_all_stages shape:", q_x_all_stages.shape)
        print("q_x_term shape:", q_x_term.shape)
        print("q_u_all shape:", q_u_all.shape)
        print("q (full) shape:", q.shape)

    # 3) Wrap as parametric-style callables
    def P_fun(theta: np.ndarray):
        # Currently ignores theta; ready for parameter-varying extension
        return P

    def q_fun(theta: np.ndarray):
        return q

    return P_fun, q_fun
