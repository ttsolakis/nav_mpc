# nav_mpc/main.py

from scipy import sparse
import numpy as np
import osqp
import time

# Import MPC2QP functionality, timing and plotting utilities (generic stuff)
from mpc2qp import build_linear_constraints, build_quadratic_objective, update_qp, pack_args, extract_solution
from utils.profiling import init_timing_stats, update_timing_stats, print_timing_summary
from simulation.simulator import ContinuousSimulator, SimulatorConfig
from simulation.plotting.plotter import plot_state_input_trajectories

# Import system, objective, constraints and animation (user-defined for the specific problem)
from models.simple_pendulum_model import SimplePendulumModel
from objectives.simple_pendulum_objective import SimplePendulumObjective
from constraints.system_constraints.simple_pendulum_sys_constraints import SimplePendulumSystemConstraints
from simulation.animation.simple_pendulum_animation import animate_pendulum

def main():
    # -----------------------------------
    # ---------- Problem Setup ----------
    # -----------------------------------

    # Enable debugging & profiling
    debugging = False
    profiling = True
    show_system_info = True
    
    # System, objective, constraints
    system      = SimplePendulumModel()
    objective   = SimplePendulumObjective(system)
    constraints = SimplePendulumSystemConstraints(system)
    
    # Initial and reference states
    x_init = np.array([0.0, 0.0])
    x_ref  = np.array([np.pi, 0.0])

    # Horizon and sampling time
    N  = 70
    dt = 0.01

    # Simulation parameters
    nsim    = 300
    sim_cfg = SimulatorConfig(dt=dt, method="rk4", substeps=10)
    sim     = ContinuousSimulator(system, sim_cfg)

    # -----------------------------------
    # ---------- QP Formulation ---------
    # -----------------------------------

    # Initialize timing stats
    timing_stats = init_timing_stats()

    # Problem dimensions
    nx = system.state_dim
    nu = system.input_dim
    nc = constraints.constraints_dim

    # Symbolic linearization + lambdified callables
    print("Building QP...")
    A_fun, l_fun, u_fun = build_linear_constraints(system, constraints, N, dt, debug=False)
    P_fun, q_fun        = build_quadratic_objective(system, objective, N, debug=False)

    # Evaluate QP data once (initial)
    print("Setting up OSQP problem...")
    x_bar_seq = np.zeros((N + 1, nx))
    u_bar_seq = np.zeros((N,     nu))
    args = pack_args(x_init, x_bar_seq, u_bar_seq, N)
    A = np.array(A_fun(*args), dtype=float)
    A = sparse.csc_matrix(A)
    l = np.array(l_fun(*args), dtype=float).reshape(-1)
    u = np.array(u_fun(*args), dtype=float).reshape(-1)
    P = P_fun(*args)             # sparse.csc_matrix
    q = q_fun(*args).reshape(-1)
    prob = osqp.OSQP()
    prob.setup(P, q, A, l, u, warm_starting=True, verbose=False)

    # -----------------------------------
    # ------------ Main Loop ------------
    # -----------------------------------

    # Initial state
    x = x_init.copy()

    # Store trajectories for plotting / animation
    x_traj = [x.copy()]   
    u_traj = []         

    print("Running main loop...")
    for i in range(nsim):

        # 1) Solve current QP and extract solution
        start_opt_time = time.perf_counter()

        res = prob.solve()
        if res.info.status != "solved": raise ValueError(f"OSQP did not solve the problem at step {i}! Status: {res.info.status}")
        X, U = extract_solution(res, nx, nu, N)

        end_opt_time = time.perf_counter()

        # 2) Simulate closed-loop step and store trajectories
        start_sim_time = time.perf_counter()

        u0 = U[0]
        x  = sim.step(x, u0)
        x_traj.append(x.copy())
        u_traj.append(u0.copy())

        end_sim_time = time.perf_counter()
        
        # 3) Evaluate QP around new (x0, x̄, ū)
        start_eQP_time = time.perf_counter()

        update_qp(prob, x, X, U, N, A_fun, l_fun, u_fun, P_fun, q_fun)

        end_eQP_time = time.perf_counter()

        # 4) Per-step prints
        if debugging:
            print(f"Step {i}: X = {X}, U = {U}")
            print(f"Step {i}: x = {x}, u0 = {u0}")
        
        # 5) Profiling updates
        if profiling:
            update_timing_stats(timing_stats, start_opt_time, end_opt_time, start_sim_time, end_sim_time, start_eQP_time, end_eQP_time)

    print("Main loop complete.")

    if profiling:
        print_timing_summary(timing_stats, N=N, nx=nx, nu=nu, nc=nc, show_system_info=show_system_info)

    # # ----------------------------------------
    # # ------------ Plot & Animate ------------
    # # ---------------------------------------- 

    print("Plotting and saving...")
    x_traj = np.vstack(x_traj)       # shape (nsim+1, nx)
    u_traj = np.vstack(u_traj)       # shape (nsim,   nu)
    total_time = dt * np.arange(x_traj.shape[0])  # length nsim+1

    # Plot state and input trajectories (generic)
    plot_state_input_trajectories(system, total_time, x_traj, u_traj, x_ref=x_ref, show=False)

    print("Animating and saving...")
    # For the pendulum, torque bounds are in SimplePendulumSystemConstraints
    _, _, umin, umax = constraints.get_bounds()
    umax_scalar = float(np.max(np.abs(umax)))

    animate_pendulum(system, total_time, x_traj, u_traj, umax=umax_scalar, show=False, save_gif=True)

if __name__ == "__main__":
    main()
