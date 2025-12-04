# nav_mpc/main.py
import numpy as np
from models.simple_pendulum_model import SimplePendulumModel
from qp_formulation.qp_formulation import build_linearized_system

from scipy.linalg import expm  # or put at top of file

def discretize(A, B, c, dt):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    c = np.asarray(c, dtype=float).reshape(-1)

    n = A.shape[0]
    m = B.shape[1]

    B_tilde = np.hstack([B, c.reshape(-1, 1)])  # n x (m+1)

    big_dim = n + m + 1
    M = np.zeros((big_dim, big_dim))
    M[:n, :n] = A
    M[:n, n:] = B_tilde

    M_d = expm(M * dt)

    Ad = M_d[:n, :n]
    B_tilde_d = M_d[:n, n:]
    Bd = B_tilde_d[:, :m]
    cd = B_tilde_d[:, m]

    return Ad, Bd, cd

def main():

   # 1) Choose a system model (symbolic)
    system = SimplePendulumModel()

    # 2) Build continuous-time affine linearization symbolically:
    #    ẋ = A(x̄,ū) x + B(x̄,ū) u + c(x̄,ū)
    (A_bar_sym, B_bar_sym, c_bar_sym, x_bar_sym, u_bar_sym, xdot_lin_sym) = build_linearized_system(system)

    # ------------------------------------------------------------------
    # 3) Choose a *numeric* operating point (x̄, ū)
    #    For the simple pendulum, e.g. downward equilibrium: θ = 0, θdot = 0, τ = 0
    # ------------------------------------------------------------------
    x_bar_val = np.array([0.0, 0.0])   # [θ̄, θ̄dot]
    u_bar_val = np.array([0.0])        # [τ̄]

    # Build substitution dict: {x̄_i -> value, ū_j -> value}
    subs_numeric = {}
    for i in range(system.state_dim):
        subs_numeric[x_bar_sym[i]] = float(x_bar_val[i])
    for j in range(system.input_dim):
        subs_numeric[u_bar_sym[j]] = float(u_bar_val[j])

    # Evaluate A(x̄,ū), B(x̄,ū), c(x̄,ū) numerically
    A_num = np.array(A_bar_sym.subs(subs_numeric), dtype=float)
    B_num = np.array(B_bar_sym.subs(subs_numeric), dtype=float)
    c_num = np.array(c_bar_sym.subs(subs_numeric), dtype=float).reshape(-1)

    print("\n--- Numeric linearized continuous-time system at (x̄,ū) ---")
    print("x̄ =", x_bar_val)
    print("ū =", u_bar_val)
    print("A_num =\n", A_num)
    print("B_num =\n", B_num)
    print("c_num =\n", c_num)

    dt = 0.1  # example sampling time [s]
    Ad, Bd, cd = discretize(A_num, B_num, c_num, dt)

    print("\n--- Discretized affine system at (x̄,ū) ---")
    print(f"dt = {dt}")
    print("Ad =\n", Ad)
    print("Bd =\n", Bd)
    print("cd =\n", cd)



    # # Build substitution dict: {x0: pi, x1: 0, u0: 0}
    # subs_dict = {
    #     x_sym[0]: x_star[0],   # x0 -> pi
    #     x_sym[1]: x_star[1],   # x1 -> 0
    #     u_sym[0]: u_star[0],   # u0 -> 0 (not used in A, but it's fine)
    # }
 
    # A_at_star_sym = A_sym.subs(subs_dict)      # still a SymPy Matrix
    # B_at_star_sym = B_sym.subs(subs_dict)

    # # Convert to numpy arrays if needed
    # A_at_star = np.array(A_at_star_sym, dtype=float)
    # B_at_star = np.array(B_at_star_sym, dtype=float)    

    # print("Pendulum symbolic state x_sym:", x_sym)
    # print("Pendulum symbolic input u_sym:", u_sym)
    # print("Pendulum symbolic dynamics f_sym(x,u):", f_sym)
   

    # print("A(x*,u*) =")
    # print(A_at_star)
    # print("B(x*,u*) =")
    # print(B_at_star)


    # # Rest of code can remain as is for now unconnected to the pendulum:
    
    # # Simple 2D double integrator:
    # # x = [position, velocity]^T
    # # x_{k+1} = A x_k + B u_k
    # dt = 0.1
    # A = np.array([
    #     [1.0, dt],
    #     [0.0, 1.0],
    # ])
    # B = np.array([
    #     [0.5 * dt**2],
    #     [dt],
    # ])

    # nx = A.shape[0]
    # nu = B.shape[1]

    # Q = np.diag([1.0, 0.1])
    # R = np.diag([0.01])

    # N = 20

    # planner = OSQPMPCPlanner(A, B, Q, R, N, verbose=True)

    # x0 = np.array([0.0, 0.0])
    # x_goal = np.array([1.0, 0.0])

    # # Constant state reference over horizon
    # x_ref = np.tile(x_goal.reshape(nx, 1), (1, N))
    # u_ref = np.zeros((nu, N - 1))  # currently ignored, kept for symmetry

    # u0, X_pred, U_pred = planner.solve(x0, x_ref, u_ref)

    # print("u0 (first control) =", u0)
    # print("X_pred shape:", X_pred.shape)
    # print("U_pred shape:", U_pred.shape)


if __name__ == "__main__":
    main()
