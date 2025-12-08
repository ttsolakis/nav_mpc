# nav_mpc/main.py

import numpy as np
from scipy import sparse
import osqp
import time

from utils.system_info import print_system_info

from models.simple_pendulum_model import SimplePendulumModel
from constraints.system_constraints.simple_pendulum_sys_constraints import SimplePendulumSystemConstraints
from objectives.simple_pendulum_objective import SimplePendulumObjective
from qp_formulation.qp_formulation import build_linear_constraints, build_quadratic_objective
from simulation.simulator import ContinuousSimulator, SimulatorConfig
from simulation.plotting.plotter import plot_state_input_trajectories
from simulation.animation.simple_pendulum_animation import animate_pendulum


def pack_args(x0: np.ndarray, x_bar_seq: np.ndarray, u_bar_seq: np.ndarray, N: int) -> list:
    """
    Build argument list in the SAME order as used in build_linear_*:

      [x0_0,...,x0_{nx-1},
       xbar0_0,...,xbar0_{nx-1}, ubar0_0,...,ubar0_{nu-1},
       ...,
       xbar{N-1}_0,...,xbar{N-1}_{nx-1}, ubar{N-1}_0,...,ubar{N-1}_{nu-1}]
    """
    nx = x0.shape[0]
    # nu can be inferred from u_bar_seq
    nu = u_bar_seq.shape[1] if N > 0 else 0

    args = []

    # x0
    args.extend(np.asarray(x0).reshape(nx))

    # stages 0..N-1
    for k in range(N):
        args.extend(np.asarray(x_bar_seq[k]).reshape(nx))
        args.extend(np.asarray(u_bar_seq[k]).reshape(nu))

    return args


def main():
    # -----------------------------------
    # ---------- Problem Setup ----------
    # -----------------------------------

    # System, objective, constraints
    system      = SimplePendulumModel()
    objective   = SimplePendulumObjective(system)
    constraints = SimplePendulumSystemConstraints(system)
    
    # Initial and reference states
    x_init = np.array([np.pi - 0.1, 0.0])
    x_ref  = np.array([np.pi, 0.0])

    # Horizon and sampling time
    N  = 40
    dt = 0.01

    # Simulation parameters
    nsim    = 500
    sim_cfg = SimulatorConfig(dt=dt, method="rk4", substeps=1)
    sim     = ContinuousSimulator(system, sim_cfg)

    # -----------------------------------
    # ---------- QP Formulation ---------
    # -----------------------------------

    # Symbolic linearization + lambdified callables
    A_fun, l_fun, u_fun = build_linear_constraints(system, constraints, N, dt, debug=False)
    P_fun, q_fun        = build_quadratic_objective(system, objective, N, debug=False)

    # Initial linearization points: all zeros
    x_bar_seq = np.zeros((N + 1, system.state_dim))
    u_bar_seq = np.zeros((N,     system.input_dim))

    # Pack args for the callables in the correct order
    args = pack_args(x_init, x_bar_seq, u_bar_seq, N)

    # Evaluate QP data once (initial)
    A = np.array(A_fun(*args), dtype=float)
    A = sparse.csc_matrix(A)
    l = np.array(l_fun(*args), dtype=float).reshape(-1)
    u = np.array(u_fun(*args), dtype=float).reshape(-1)
    P = P_fun(*args)             # sparse.csc_matrix
    q = q_fun(*args).reshape(-1)

    # Initialize OSQP
    prob = osqp.OSQP()
    prob.setup(P, q, A, l, u, warm_starting=True, verbose=False)

    # -----------------------------------
    # ------------ Main Loop ------------
    # -----------------------------------

    x = x_init.copy()

    # store trajectories (lists first)
    x_traj = [x.copy()]   # include initial state
    u_traj = []           # nsim control inputs

    total_opt_time = 0.0
    min_opt_time = float("inf")
    max_opt_time = 0.0
    total_sim_time = 0.0
    min_sim_time = float("inf")
    max_sim_time = 0.0
    total_eQP_time = 0.0
    min_eQP_time = float("inf")
    max_eQP_time = 0.0

    for i in range(nsim):

        # 1) Solve current QP
        start_opt_time = time.perf_counter()

        res = prob.solve()
        if res.info.status != "solved":
            raise ValueError(
                f"OSQP did not solve the problem at step {i}! Status: {res.info.status}"
            )

        z = res.x

        end_opt_time = time.perf_counter()

        nx = system.state_dim
        nu = system.input_dim

        X_flat = z[0:(N + 1) * nx]
        U_flat = z[(N + 1) * nx:]
        X = X_flat.reshape(N + 1, nx)
        U = U_flat.reshape(N,     nu)

        # 2) Simulate closed-loop step and store trajectories
        start_sim_time = time.perf_counter()

        u0 = U[0]
        x  = sim.step(x, u0)
        x_traj.append(x.copy())
        u_traj.append(u0.copy())

        end_sim_time = time.perf_counter()
        

        # 3) Evaluate QP around new (x̄, ū)
        start_eQP_time = time.perf_counter()

        x_init = x.copy()
        x_bar_seq = X.copy()
        u_bar_seq = U.copy()
        # TODO: Shift previous trajectory one step forward for linearization: 
        # p̄_{t|k} = p_{t-1|k+1} for k = 0..N-1 # p̄_{t|N} = 2*p_{t-1|N} - p_{t-1|N-1} 
        # where p̄_{t|:} is the linearization sequence and p_{t-1|:} is the previous optimal sequence of either x or u depends on use. x_bar_seq = X u_bar_seq = U

        args = pack_args(x_init, x_bar_seq, u_bar_seq, N)
        A_new = np.array(A_fun(*args), dtype=float)
        A_new = sparse.csc_matrix(A_new)
        l_new = np.array(l_fun(*args), dtype=float).reshape(-1)
        u_new = np.array(u_fun(*args), dtype=float).reshape(-1)
        P_new = P_fun(*args)             # still constant in the simple case
        q_new = q_fun(*args).reshape(-1)
        prob.update(Px=sparse.triu(P_new).data, Ax=A_new.data, q=q_new, l=l_new, u=u_new)

        end_eQP_time = time.perf_counter()
        
        # 4) Accumulate timers for profiling
        opt_time = (end_opt_time - start_opt_time) * 1e3  # ms
        total_opt_time += opt_time
        min_opt_time = min(min_opt_time, opt_time)
        max_opt_time = max(max_opt_time, opt_time)

        sim_time = (end_sim_time - start_sim_time) * 1e3  # ms
        total_sim_time += sim_time
        min_sim_time = min(min_sim_time, sim_time)
        max_sim_time = max(max_sim_time, sim_time)

        eQP_time = (end_eQP_time - start_eQP_time) * 1e3  # ms
        total_eQP_time += eQP_time
        min_eQP_time = min(min_eQP_time, eQP_time)
        max_eQP_time = max(max_eQP_time, eQP_time)

        print(f"Step {i}: x = {x}, u0 = {u0}", "optimization time: ", {opt_time}, "simulation time: ", {sim_time}, "QP evaluation time: ", {eQP_time})

    avg_opt_time        = total_opt_time / nsim
    avg_sim_time        = total_sim_time / nsim
    avg_eQP_time = total_eQP_time / nsim


    print("\n=== Timing statistics over MPC loop ===")
    print(f"Problem size: N = {N}, nx = {system.state_dim}, nu = {system.input_dim}, nc TODO add number of constraints")
    print(f"Optimization time:   avg = {avg_opt_time:.3f} ms, "
          f"min = {min_opt_time:.3f} ms, max = {max_opt_time:.3f} ms")
    print(f"Simulation time:     avg = {avg_sim_time:.3f} ms, "
          f"min = {min_sim_time:.3f} ms, max = {max_sim_time:.3f} ms")
    print(f"QP evaluation time:  avg = {avg_eQP_time:.3f} ms, "
          f"min = {min_eQP_time:.3f} ms, max = {max_eQP_time:.3f} ms")
    print_system_info()

    # # ----------------------------------------
    # # ------------ Plot & Animate ------------
    # # ---------------------------------------- 
    # x_traj = np.vstack(x_traj)       # shape (nsim+1, nx)
    # u_traj = np.vstack(u_traj)       # shape (nsim,   nu)
    # total_time = dt * np.arange(x_traj.shape[0])  # length nsim+1

    # # Plot state and input trajectories (generic)
    # plot_state_input_trajectories(system, total_time, x_traj, u_traj, x_ref=x_ref, show=False)

    # # For the pendulum, torque bounds are in SimplePendulumSystemConstraints
    # _, _, umin, umax = constraints.get_bounds()
    # umax_scalar = float(np.max(np.abs(umax)))

    # animate_pendulum(system, total_time, x_traj, u_traj, umax=umax_scalar, show=False)


if __name__ == "__main__":
    main()
