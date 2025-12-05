# nav_mpc/main.py

import time

import numpy as np
from scipy.linalg import expm
from scipy import sparse

from models.simple_pendulum_model import SimplePendulumModel
from constraints.system_constraints.simple_pendulum_sys_constraints import SimplePendulumSystemConstraints 
from qp_formulation.qp_formulation import build_linearized_system


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


def main():
    # 1) System
    system = SimplePendulumModel()
    n = system.state_dim
    m = system.input_dim

    # 2) Symbolic linearization + lambdified callables
    (A_fun, B_fun, c_fun) = build_linearized_system(system)

    N = 20
    dt = 0.01

    # Fake a sequence of operating points (x̄_k, ū_k)
    rng = np.random.default_rng(seed=42)   # <--- important: reproducible tests

    # Define reasonable ranges for the pendulum:
    # θ ∈ [-0.3 rad, 0.3 rad],  θdot ∈ [-1 rad/s, 1 rad/s]
    x_bounds_low  = np.array([-0.3, -1.0])
    x_bounds_high = np.array([ 0.3,  1.0])

    # Torque input τ ∈ [-0.2 Nm, 0.2 Nm]
    u_bounds_low  = np.array([-0.2])
    u_bounds_high = np.array([ 0.2])

    x_bar_seq = [
        rng.uniform(low=x_bounds_low, high=x_bounds_high)
        for _ in range(N)
    ]

    u_bar_seq = [
        rng.uniform(low=u_bounds_low, high=u_bounds_high)
        for _ in range(N)
    ]

    Ad_list, Bd_list, cd_list = [], [], []

    t0 = time.perf_counter()

    for k in range(N):
        x_bar = x_bar_seq[k]
        u_bar = u_bar_seq[k]

        # Pack arguments for A_fun, B_fun, c_fun
        args = list(x_bar) + list(u_bar)

        # Fast numeric evaluation (no SymPy here)
        A_k = np.array(A_fun(*args), dtype=float)
        B_k = np.array(B_fun(*args), dtype=float)
        c_k = np.array(c_fun(*args), dtype=float).reshape(-1)

        Ad_k, Bd_k, cd_k = discretize_affine(A_k, B_k, c_k, dt)

        Ad_list.append(Ad_k)
        Bd_list.append(Bd_k)
        cd_list.append(cd_k)

    nx, nu = n, m

    # Decision variable:
    #   z = [x0, x1, ..., xN, u0, ..., u_{N-1}]
    Nz = (N + 1) * nx + N * nu

    # Equality constraints:
    #  - nx rows for initial condition x0 = x_init
    #  - N*nx rows for dynamics x_{k+1} - Ad_k x_k - Bd_k u_k = cd_k
    n_eq = (N + 1) * nx

    Aeq = sparse.lil_matrix((n_eq, Nz), dtype=float)
    leq = np.zeros(n_eq, dtype=float)
    ueq = np.zeros(n_eq, dtype=float)

    # ---- 1) Initial condition: x0 = x_init ----
    x_init = np.array([0.1, 0.0])   # same as before

    # Row block 0..nx-1 acts only on x0
    Aeq[0:nx, 0:nx] = sparse.eye(nx)
    leq[0:nx] = x_init
    ueq[0:nx] = x_init

    # ---- 2) Dynamics: for k = 0,...,N-1 ----
    #    x_{k+1} - Ad_k x_k - Bd_k u_k = cd_k
    # →  [ ... -Ad_k ... I ... -Bd_k ... ] z = cd_k
    for k in range(N):
        row_start = (k + 1) * nx
        row_end   = row_start + nx

        # State block columns
        col_xk_start   = k * nx
        col_xk_end     = col_xk_start + nx
        col_xkp1_start = (k + 1) * nx
        col_xkp1_end   = col_xkp1_start + nx

        # Input block columns (after all states)
        col_uk_start = (N + 1) * nx + k * nu
        col_uk_end   = col_uk_start + nu

        Ad_k = Ad_list[k]
        Bd_k = Bd_list[k]
        cd_k = cd_list[k]

        # -Ad_k on x_k
        Aeq[row_start:row_end, col_xk_start:col_xk_end] = -Ad_k

        # +I on x_{k+1}
        Aeq[row_start:row_end, col_xkp1_start:col_xkp1_end] = sparse.eye(nx)

        # -Bd_k on u_k
        Aeq[row_start:row_end, col_uk_start:col_uk_end] = -Bd_k

        # RHS: cd_k
        leq[row_start:row_end] = cd_k
        ueq[row_start:row_end] = cd_k

    # Convert to CSC for OSQP
    Aeq = Aeq.tocsc()

    # ---------- System constraints ----------

    # Instantiate constraints for this system
    sys_constr = SimplePendulumSystemConstraints(system)
    xmin, xmax, umin, umax = sys_constr.get_bounds()

    # Decision variable: z = [x0,...,xN, u0,...,u_{N-1}]
    Nz = (N + 1) * nx + N * nu

    # Aineq z <= uineq, lineq <= Aineq z
    # We take Aineq = I so bounds directly constrain z
    Aineq = sparse.eye(Nz, format="csc")

    # Repeat bounds across horizon
    x_lower_all = np.kron(np.ones(N + 1), xmin)
    x_upper_all = np.kron(np.ones(N + 1), xmax)
    u_lower_all = np.kron(np.ones(N), umin)
    u_upper_all = np.kron(np.ones(N), umax)

    lineq = np.hstack([x_lower_all, u_lower_all])
    uineq = np.hstack([x_upper_all, u_upper_all])

    # ---------- Stack equalities + inequalities for OSQP ----------

    A = sparse.vstack([Aeq, Aineq], format="csc")
    l = np.hstack([leq, lineq])
    u = np.hstack([ueq, uineq])

    t1 = time.perf_counter()
    print(f"\nLTV dynamics assembly time: {(t1 - t0)*1e3:.3f} ms")

    print("\n=== LTV dynamics assembly check ===")
    print(f"State dim n = {n}, input dim m = {m}, horizon N = {N}")
    print("Ad_k shape:", Ad_list[0].shape,
          "Bd_k shape:", Bd_list[0].shape,
          "cd_k shape:", cd_list[0].shape)
    print("Aeq shape:", Aeq.shape)
    print("leq shape:", leq.shape)
    print("Aineq shape:", Aineq.shape)
    print("A (full) shape:", A.shape)
    print("l shape:", l.shape)
    print("u shape:", u.shape)


if __name__ == "__main__":
    main()
