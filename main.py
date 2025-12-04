# nav_mpc/main.py

import time

import numpy as np
from scipy.linalg import expm

from models.simple_pendulum_model import SimplePendulumModel
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

    # Build big LTV matrices
    A_x = np.zeros((N * n, (N + 1) * n), dtype=float)
    B_u = np.zeros((N * n, N * m), dtype=float)
    d = np.zeros(N * n, dtype=float)

    for k in range(N):
        row_start = k * n
        row_end = row_start + n

        col_xk = k * n
        col_xkp1 = (k + 1) * n

        Ad_k = Ad_list[k]
        Bd_k = Bd_list[k]
        cd_k = cd_list[k]

        A_x[row_start:row_end, col_xk:col_xk + n] = -Ad_k
        A_x[row_start:row_end, col_xkp1:col_xkp1 + n] = np.eye(n)

        col_uk = k * m
        B_u[row_start:row_end, col_uk:col_uk + m] = -Bd_k

        d[row_start:row_end] = cd_k

    x_init = np.array([0.1, 0.0])

    A_ic = np.zeros((n, (N + 1) * n + N * m), dtype=float)
    b_ic = x_init.copy()
    A_ic[:, 0:n] = np.eye(n)

    A_eq_dyn = np.hstack([A_x, B_u])
    b_eq_dyn = d

    A_eq = np.vstack([A_ic, A_eq_dyn])
    b_eq = np.concatenate([b_ic, b_eq_dyn])

    t1 = time.perf_counter()
    print(f"\nLTV dynamics assembly time: {(t1 - t0)*1e3:.3f} ms")

    print("\n=== LTV dynamics assembly check ===")
    print(f"State dim n = {n}, input dim m = {m}, horizon N = {N}")
    print("Ad_k shape:", Ad_list[0].shape,
          "Bd_k shape:", Bd_list[0].shape,
          "cd_k shape:", cd_list[0].shape)
    print("A_x shape:", A_x.shape)
    print("B_u shape:", B_u.shape)
    print("d shape:", d.shape)
    print("A_eq shape:", A_eq.shape)
    print("b_eq shape:", b_eq.shape)


if __name__ == "__main__":
    main()
