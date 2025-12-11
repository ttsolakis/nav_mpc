# nav_mpc/qp_formulation/qp_offline.py

from scipy import sparse
import sympy as sp
import numpy as np
from models.dynamics import SystemModel
from constraints.sys_constraints import SystemConstraints
from objectives.objectives import Objective

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


def build_linear_equality_constraints(system: SystemModel, N: int, dt: float, debug: bool=False):
    """
    Build lambdified functions for the dynamics equality constraints of the LTV MPC:

        xdot = f(x,u) ≈ A_ct(x̄,ū) x + B_ct(x̄,ū) u + c_ct(x̄,ū) =>

        x_{k+1} = A_d(x̄_k, ū_k) x_k + B_d(x̄_k, ū_k) u_k + c_d(x̄_k, ū_k),

    where A_d, B_d, c_d are obtained by approximate discretization.

    We then build the stacked equality constraints:

    Aeq z = leq = ueq,  with z = [x0,...,xN,u0,...,u_{N-1}],  leq = ueq = (-x0, cd_0,...,cd_{N-1}).

    with:
    
    Aeq = [ -I      0      0     ...     0  |  0      0     ...      0    ]
          [ Ad_0   -I      0     ...     0  | Bd_0    0     ...      0    ]
          [  0     Ad_1   -I     ...     0  |  0     Bd_1   ...      0    ] ∈ R^{(N+1)*nx X ((N+1)*nx + N*nu)}
          [ ...                             | ...                         ]  
          [  0      0     ...  Ad_{N-1}  -I |  0      0     ...   Bd_{N-1}] 
     

    leq = ueq = (-x0 cd_0 cd_1 cd_2 ... cd_{N_1})^T  ∈ R^{(N+1)*nx}

    in symbolic form

    and return lambdified callables:

        Aeq_fun(*args) -> (n_eq, n_z) ndarray
        leq_fun(*args) -> (n_eq,) ndarray
        ueq_fun(*args) -> (n_eq,) ndarray

    with args ordered as:

        [x0_0,...,x0_{nx-1},
         xbar0_0,...,xbar0_{nx-1}, ubar0_0,...,ubar0_{nu-1},
         ...,
         xbar{N-1}_0,...,ubar{N-1}_{nu-1}]
    """

    nx = system.state_dim
    nu = system.input_dim

    # ---------------------------------------
    #  1) Continuous-time linearized dynamics
    # ---------------------------------------
    x_sym = system.state_symbolic()     # (nx, 1)
    u_sym = system.input_symbolic()     # (nu, 1)
    f_sym = system.dynamics_symbolic()  # (nx, 1)

    # Linearization: f ≈ A_ct x + B_ct u + c_ct
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

    # -----------------------------------------------------------------------
    #  2) Discrete-time linearized dynamics (with approximate discretization)
    # -----------------------------------------------------------------------

    # 2nd-order Taylor of exp(A dt) and the affine part:
    #  A_d ≈ I + dt*A + 0.5*dt^2*A^2
    #  B_d ≈ dt*B + 0.5*dt^2*A*B
    #  c_d ≈ dt*c + 0.5*dt^2*A*c
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

    # -----------------------------------------------------------------
    #  3) Create theta symbols and views (x0, x̄_k, ū_k)
    # -----------------------------------------------------------------
    theta_syms = build_theta_symbols(nx, nu, N)
    x0_syms, xbar_syms, ubar_syms = theta_views(theta_syms, nx, nu, N)
    x0_vec = sp.Matrix(x0_syms)

    # -----------------------------------------------------------------
    #  4) Ad_k, Bd_k, cd_k by substitution: (x_sym,u_sym) -> (x̄_k,ū_k)
    # -----------------------------------------------------------------

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
    
    # -----------------------------------------------------------------
    #  5) Build equality constraints Aeq, leq, ueq with known structure
    # -----------------------------------------------------------------

    n_z  = (N + 1) * nx + N * nu
    n_eq = (N + 1) * nx

    Aeq_sym = sp.zeros(n_eq, n_z)
    leq_sym = sp.zeros(n_eq, 1)
    ueq_sym = sp.zeros(n_eq, 1)

    # 5a) initial condition: x0 = x_init
    # First nx rows: I on the x0 block
    for i in range(nx):
        Aeq_sym[i, i] = 1.0
        leq_sym[i, 0] = x0_vec[i]
        ueq_sym[i, 0] = x0_vec[i]

    # 5b) dynamics rows: x_{k+1} - Ad_k x_k - Bd_k u_k = cd_k
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

    # -----------------------------------------------------------------
    #  6) Lambdify Aeq, leq, ueq with a SINGLE argument: theta (1D array)
    # -----------------------------------------------------------------

    Aeq_fun = sp.lambdify([theta_syms], Aeq_sym, "numpy")
    leq_fun = sp.lambdify([theta_syms], leq_sym, "numpy")
    ueq_fun = sp.lambdify([theta_syms], ueq_sym, "numpy")

    return Aeq_fun, leq_fun, ueq_fun


def build_linear_inequality_constraints(system: SystemModel, constraints: SystemConstraints, N: int, debug: bool = False):
    """
    Build lambdified functions for *linearized* inequality constraints of the LTV MPC:

        Nonlinear constraints:   g(x, u) <= 0,   g: R^{nx} x R^{nu} -> R^{nc}

    Linearization around an operating point (x̄, ū):

        g(x, u) ≈ ḡ + Hx (x - x̄) + Hu (u - ū),

        where:
          Hx = ∂g/∂x |_(x̄, ū) ∈ R^{nc x nx}
          Hu = ∂g/∂u |_(x̄, ū) ∈ R^{nc x nu}
          ḡ = g(x̄, ū)         ∈ R^{nc}

    Enforcing g(x, u) <= 0 via the linearization means:

        Hx x + Hu u <= h,

        where:
          h(x̄, ū) = -ḡ + Hx x̄ + Hu ū ∈ R^{nc}.

    We then build *stacked* stage-wise inequalities over the horizon:

        lineq <= Aineq z <= uineq,   z = [x0,...,xN,u0,...,u_{N-1}].

    For each stage k = 0,...,N-1 we impose:

        Hx_k x_k + Hu_k u_k <= h_k,

    so that:

        Aineq ∈ R^{(N*nc) x ((N+1)*nx + N*nu)},
        lineq ∈ R^{N*nc},  uineq ∈ R^{N*nc}.

    We choose:

        lineq = -∞ (vector of -inf),
        uineq = stacked [h_0; h_1; ...; h_{N-1}].

    The lambdified callables have the signature:

        Aineq_fun(*args) -> (n_ineq, n_z) ndarray
        lineq_fun(*args) -> (n_ineq,) ndarray  (all -inf)
        uineq_fun(*args) -> (n_ineq,) ndarray

    with args ordered as:

        [x0_0,...,x0_{nx-1},
         xbar0_0,...,xbar0_{nx-1}, ubar0_0,...,ubar0_{nu-1},
         ...,
         xbar{N-1}_0,...,xbar{N-1}_{nx-1}, ubar{N-1}_0,...,ubar{N-1}_{nu-1}]
    """

    nx = system.state_dim
    nu = system.input_dim
    
    # ------------------------------------------------
    # 1) Symbolic variables and nonlinear constraints
    # ------------------------------------------------
    x_sym = system.state_symbolic()      # (nx, 1)
    u_sym = system.input_symbolic()      # (nu, 1)

    # User / system must provide g(x,u) in symbolic form:
    # g_sym: R^{nx} x R^{nu} -> R^{nc}, as a column Matrix.
    g_sym = constraints.constraints_symbolic()   # (nc, 1)
    nc = g_sym.shape[0]

    # Jacobians:
    Hx_sym = g_sym.jacobian(x_sym)      # (nc, nx)
    Hu_sym = g_sym.jacobian(u_sym)      # (nc, nu)

    if debug:
        print("g_sym(x,u):")
        sp.pprint(g_sym)
        print("Hx_sym = dg/dx:")
        sp.pprint(Hx_sym)
        print("Hu_sym = dg/du:")
        sp.pprint(Hu_sym)

    # ---------------------------------------------------------
    # 2) Create theta symbols and views for linearization sequence
    # ---------------------------------------------------------
    theta_syms = build_theta_symbols(nx, nu, N)
    x0_syms, xbar_syms, ubar_syms = theta_views(theta_syms, nx, nu, N)
    # x0_syms not used directly in constraints, but present in theta for consistency

    # ---------------------------------------------------------
    # 3) Stage-wise Hx_k, Hu_k, h_k via substitution
    # ---------------------------------------------------------
    Hx_list = []
    Hu_list = []
    h_list  = []    # right-hand sides

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

    # ---------------------------------------------------------
    # 4) Build stacked Aineq, lineq, uineq
    # ---------------------------------------------------------
    n_z     = (N + 1) * nx + N * nu
    n_ineq  = N * nc   # constraints for k = 0,...,N-1

    Aineq_sym = sp.zeros(n_ineq, n_z)
    lineq_sym = sp.zeros(n_ineq, 1)
    uineq_sym = sp.zeros(n_ineq, 1)

    # We enforce: Hx_k x_k + Hu_k u_k <= h_k
    # i.e., Aineq rows are [ ..., Hx_k (at x_k block) ..., Hu_k (at u_k block) ... ]
    # with upper bound uineq = h_k, lower bound -inf.

    for k in range(N):
        row_start    = k * nc
        col_xk_start = k * nx
        col_uk_start = (N + 1) * nx + k * nu

        Hx_k = Hx_list[k]
        Hu_k = Hu_list[k]
        h_k  = h_list[k]   # (nc,1)

        # Fill Aineq for state block x_k
        for i in range(nc):
            for j in range(nx):
                Aineq_sym[row_start + i, col_xk_start + j] = Hx_k[i, j]

        # Fill Aineq for input block u_k
        for i in range(nc):
            for j in range(nu):
                Aineq_sym[row_start + i, col_uk_start + j] = Hu_k[i, j]

        # Upper bound = h_k
        for i in range(nc):
            uineq_sym[row_start + i, 0] = h_k[i, 0]

        # Lower bound = -∞ (no lower constraint from g(x,u) <= 0)
        # Use -∞ symbolically so OSQP sees -np.inf after lambdify.
        for i in range(nc):
            lineq_sym[row_start + i, 0] = -sp.oo

    if debug:
        print("\nAineq_sym:")
        sp.pprint(Aineq_sym)
        print("lineq_sym:")
        sp.pprint(lineq_sym)
        print("uineq_sym:")
        sp.pprint(uineq_sym)

    # ---------------------------------------------------------
    # 5) Lambdify with a SINGLE argument: theta
    # ---------------------------------------------------------

    Aineq_fun  = sp.lambdify([theta_syms], Aineq_sym, "numpy")
    lineq_fun  = sp.lambdify([theta_syms], lineq_sym, "numpy")
    uineq_fun  = sp.lambdify([theta_syms], uineq_sym, "numpy")

    return Aineq_fun, lineq_fun, uineq_fun


def build_linear_constraints(system: SystemModel, constraints: SystemConstraints, N: int, dt: float, debug: bool = False):
    """
    Wrap equality + inequality lambdified functions into combined
    A_fun, l_fun, u_fun:

      A_fun(*args) -> stacked A (eq + ineq) as ndarray
      l_fun(*args) -> stacked l as ndarray
      u_fun(*args) -> stacked u as ndarray
    """

    Aeq_fun, leq_fun, ueq_fun = build_linear_equality_constraints(system, N, dt, debug=debug)
    Aineq_fun, lineq_fun, uineq_fun = build_linear_inequality_constraints(system, constraints, N, debug=debug)

    # --- Precompute constant lower bounds for inequalities: g(x,u) <= 0 ---
    # They are always -inf, they do NOT depend on theta.
    g_sym = constraints.constraints_symbolic()
    nc = g_sym.shape[0]
    lineq_const = -np.inf * np.ones(N * nc)

    def A_fun(theta: np.ndarray):
        """Return stacked A (eq + ineq) as dense ndarray."""
        Aeq_num   = np.asarray(Aeq_fun(theta),   dtype=float)
        Aineq_num = np.asarray(Aineq_fun(theta), dtype=float)
        A = np.vstack([Aeq_num, Aineq_num])

        if debug:
            print("Aeq_num shape:", Aeq_num.shape)
            print("Aineq_num shape:", Aineq_num.shape)
            print("A (stacked) shape:", A.shape)

        return A


    def l_fun(theta: np.ndarray):
        """Return stacked l (eq + ineq)."""
        leq_num = np.asarray(leq_fun(theta), dtype=float).reshape(-1)

        # DO NOT call lineq_fun(theta) anymore – it's constant -inf.
        l = np.hstack([leq_num, lineq_const])

        if debug:
            print("leq_num shape:", leq_num.shape)
            print("lineq_const shape:", lineq_const.shape)
            print("l (stacked) shape:", l.shape)

        return l


    def u_fun(theta: np.ndarray):
        """Return stacked u (eq + ineq)."""
        ueq_num   = np.asarray(ueq_fun(theta),   dtype=float).reshape(-1)
        uineq_num = np.asarray(uineq_fun(theta), dtype=float).reshape(-1)
        u = np.hstack([ueq_num, uineq_num])

        if debug:
            print("ueq_num shape:", ueq_num.shape)
            print("uineq_num shape:", uineq_num.shape)
            print("u (stacked) shape:", u.shape)

        return u

    return A_fun, l_fun, u_fun


def build_quadratic_objective(system: SystemModel, objective: Objective, N: int, debug: bool = False):
    """
    Build callables P_fun, q_fun for OSQP:

        0.5 * z^T P z + q^T z

    with decision vector:

        z = [x0, x1, ..., xN, u0, ..., u_{N-1}] ∈ R^{(N+1)*nx + N*nu}.

    For the simple case with identity error maps:

        e_x(x) = x - x_ref
        e_u(u) = u - u_ref

    the cost is:

        J = 0.5 * Σ_{k=0}^{N-1} (x_k - x_ref)^T Q  (x_k - x_ref)
          + 0.5 *         (x_N - x_ref)^T QN (x_N - x_ref)
          + 0.5 * Σ_{k=0}^{N-1} (u_k - u_ref)^T R  (u_k - u_ref).

    This yields a block-diagonal P and a stacked q. For now they are
    independent of x̄, ū, but we expose them via callables P_fun(*args),
    q_fun(*args) so the API matches A_fun, l_fun, u_fun and can be
    generalized later.
    """

    nx = system.state_dim
    nu = system.input_dim

    Q  = objective.Q   # (nx, nx)
    QN = objective.QN  # (nx, nx)
    R  = objective.R   # (nu, nu)

    x_ref = objective.x_ref.reshape(nx)  # (nx,)
    u_ref = objective.u_ref.reshape(nu)  # (nu,)

    # ------------------------------------------------------------
    # 1) Build P as block-diagonal:
    #
    #    P = blkdiag( Q, ..., Q, QN, R, ..., R )
    #
    #    with N stage Q blocks, 1 terminal QN, and N stage R blocks.
    # ------------------------------------------------------------

    # State part: [Q, Q, ..., Q, QN]
    if N > 0:
        P_state = sparse.kron(sparse.eye(N), Q)   # shape (N*nx, N*nx)
        P_term  = QN                              # shape (nx, nx)
        P_x = sparse.block_diag([P_state, P_term], format="csc")
    else:
        # Degenerate horizon (just terminal)
        P_x = sparse.csc_matrix(QN)

    # Input part: [R, R, ..., R]
    if N > 0:
        P_u = sparse.kron(sparse.eye(N), R)       # shape (N*nu, N*nu)
    else:
        P_u = sparse.csc_matrix((0, 0))

    # Full P = blkdiag(P_x, P_u)
    P = sparse.block_diag([P_x, P_u], format="csc")

    if debug:
        print("P_x shape:", P_x.shape)
        print("P_u shape:", P_u.shape)
        print("P (full) shape:", P.shape)

    # ------------------------------------------------------------
    # 2) Build q for the shifted quadratic:
    #
    #    0.5 (x - x_ref)^T Q (x - x_ref)
    #      = 0.5 x^T Q x - x^T Q x_ref + const
    #      = 0.5 x^T Q x + ( -Q x_ref )^T x + const
    #
    #    So each state block contributes q_x_block = -Q * x_ref,
    #    terminal block contributes q_xN_block = -Q_N * x_ref,
    #    and each input block contributes q_u_block = -R * u_ref.
    # ------------------------------------------------------------

    # State blocks (k = 0,...,N-1)
    q_x_stage = -Q @ x_ref          # (nx,)
    if N > 0:
        q_x_all_stages = np.kron(np.ones(N), q_x_stage)  # (N*nx,)
    else:
        q_x_all_stages = np.zeros(0)

    # Terminal block
    q_x_term = -QN @ x_ref          # (nx,)

    # Input blocks
    q_u_stage = -R @ u_ref          # (nu,)
    if N > 0:
        q_u_all = np.kron(np.ones(N), q_u_stage)        # (N*nu,)
    else:
        q_u_all = np.zeros(0)

    # Stack: [ all state stages, terminal, all inputs ]
    q = np.hstack([q_x_all_stages, q_x_term, q_u_all])

    if debug:
        print("q_x_all_stages shape:", q_x_all_stages.shape)
        print("q_x_term shape:", q_x_term.shape)
        print("q_u_all shape:", q_u_all.shape)
        print("q (full) shape:", q.shape)

    # ------------------------------------------------------------
    # 3) Wrap as callables P_fun, q_fun (*args) to match A_fun, l_fun, u_fun.
    #    In the simple case P and q do NOT depend on x̄, ū, so *args is unused.
    #    Later, for nonlinear error maps, you can re-use *args inside here.
    # ------------------------------------------------------------

    def P_fun(theta: np.ndarray):
        """
        Same theta layout as A_fun/l_fun/u_fun.
        Currently ignored (P is constant), but kept for future TV extensions.
        """
        return P

    def q_fun(theta: np.ndarray):
        return q

    return P_fun, q_fun