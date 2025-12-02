# nav_mpc/main.py

import numpy as np
from planner.planner import TinyMPCPlanner


def main():
    # Simple 2D double integrator:
    # x = [position, velocity]^T
    # x_{k+1} = A x_k + B u_k
    dt = 0.1
    A = np.array([
        [1.0, dt],
        [0.0, 1.0],
    ])
    B = np.array([
        [0.5 * dt**2],
        [dt],
    ])

    nx = A.shape[0]
    nu = B.shape[1]

    Q = np.diag([1.0, 0.1])
    R = np.diag([0.01])

    N = 20

    planner = TinyMPCPlanner(A, B, Q, R, N, rho=1.0, verbose=True)

    x0 = np.array([0.0, 0.0])
    x_goal = np.array([1.0, 0.0])

    # Constant state reference over horizon
    x_ref = np.tile(x_goal.reshape(nx, 1), (1, N))
    # Zero input reference
    u_ref = np.zeros((nu, N - 1))

    u0, X_pred, U_pred = planner.solve(x0, x_ref, u_ref)

    print("u0 (first control) =", u0)
    print("X_pred shape:", X_pred.shape)
    print("U_pred shape:", U_pred.shape)


if __name__ == "__main__":
    main()
