# nav_mpc/main.py
from models.simple_pendulum_model import SimplePendulumModel
import sympy as sp

import numpy as np
from planner.planner import OSQPMPCPlanner

def main():


    # Lets start here calling the symbolic dynamics of the pendulum

    pendulum = SimplePendulumModel()

    x_sym = pendulum.state_symbols()        # or pendulum.x_sym
    u_sym = pendulum.input_symbols()        # or pendulum.u_sym
    f_sym = pendulum.dynamics_symbolic()    # or pendulum.f_sym

    # Compute continuous-time Jacobians (symbolic)
    A_sym = f_sym.jacobian(x_sym)           # df/dx, shape (2,2)
    B_sym = f_sym.jacobian(u_sym)           # df/du, shape (2,1)

    # Define operating point (x*, u*) = (pi, 0)
    x_star = np.array([np.pi, 0.0])
    u_star = np.array([0.0])

    # Build substitution dict: {x0: pi, x1: 0, u0: 0}
    subs_dict = {
        x_sym[0]: x_star[0],   # x0 -> pi
        x_sym[1]: x_star[1],   # x1 -> 0
        u_sym[0]: u_star[0],   # u0 -> 0 (not used in A, but it's fine)
    }

    A_at_star_sym = A_sym.subs(subs_dict)      # still a SymPy Matrix
    B_at_star_sym = B_sym.subs(subs_dict)

    # Convert to numpy arrays if needed
    A_at_star = np.array(A_at_star_sym, dtype=float)
    B_at_star = np.array(B_at_star_sym, dtype=float)    

    print("Pendulum symbolic state x_sym:", x_sym)
    print("Pendulum symbolic input u_sym:", u_sym)
    print("Pendulum symbolic dynamics f_sym(x,u):", f_sym)
    print("Pendulum A_sym = df/dx:")
    sp.pprint(A_sym)
    print("Pendulum B_sym = df/du:")
    sp.pprint(B_sym)

    print("A(x*,u*) =")
    print(A_at_star)
    print("B(x*,u*) =")
    print(B_at_star)


    # Rest of code can remain as is for now unconnected to the pendulum:
    
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

    planner = OSQPMPCPlanner(A, B, Q, R, N, verbose=True)

    x0 = np.array([0.0, 0.0])
    x_goal = np.array([1.0, 0.0])

    # Constant state reference over horizon
    x_ref = np.tile(x_goal.reshape(nx, 1), (1, N))
    u_ref = np.zeros((nu, N - 1))  # currently ignored, kept for symmetry

    u0, X_pred, U_pred = planner.solve(x0, x_ref, u_ref)

    print("u0 (first control) =", u0)
    print("X_pred shape:", X_pred.shape)
    print("U_pred shape:", U_pred.shape)


if __name__ == "__main__":
    main()
