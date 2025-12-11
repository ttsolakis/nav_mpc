# nav_mpc/main.py

import numpy as np
import osqp
import time

# Import MPC2QP functionality, timing and plotting utilities (generic stuff)
from mpc2qp import build_linear_constraints, build_quadratic_objective, set_qp, update_qp, extract_solution
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

    # Use Cython for speed in embedded systems (online functions ~5x times faster than pure Python)
    use_cython = True
    
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

    print("Building QP...")

    start_bQP_time = time.perf_counter()

    A_fun, l_fun, u_fun = build_linear_constraints(system, constraints, N, dt, use_cython, debug=False)
    P_fun, q_fun        = build_quadratic_objective(system, objective, N, debug=False)

    end_bQP_time = time.perf_counter()
    duration_seconds = end_bQP_time - start_bQP_time
    minutes, seconds = divmod(duration_seconds, 60)

    print(f"QP built in {int(minutes):02} minutes {int(seconds):02} seconds.")


    # -----------------------------------
    # ------------ Main Loop ------------
    # -----------------------------------

    print("Initializations...")

    # Initialize timing stats
    timing_stats = init_timing_stats()

    # Problem dimension
    nx = system.state_dim
    nu = system.input_dim
    nc = constraints.constraints_dim

    # Initial state
    x = x_init.copy()

    # Initialize nominal trajectories
    x_bar_seq = np.zeros((N + 1, nx))
    u_bar_seq = np.zeros((N,     nu))

    # Store trajectories for plotting / animation
    x_traj = [x.copy()]   
    u_traj = [] 

    # Initialize OSQP solver
    prob = osqp.OSQP()
    P, q, A, l, u, A_row_idx, A_col_idx, P_row_idx, P_col_idx = set_qp(x, x_bar_seq, u_bar_seq, N, A_fun, l_fun, u_fun, P_fun, q_fun)
    prob.setup(P, q, A, l, u, warm_starting=True, verbose=False)

    print("Running main loop...")
    for i in range(nsim):

        # 1) Evaluate QP around new (x0, x̄, ū)
        start_eQP_time = time.perf_counter()

        if i > 0:
            update_qp(prob, x, X, U, N, A_fun, l_fun, u_fun, P_fun, q_fun, A_row_idx, A_col_idx, P_row_idx, P_col_idx)
                      
        end_eQP_time = time.perf_counter()

        # 2) Solve current QP and extract solution
        start_opt_time = time.perf_counter()

        osqp_time_limit = dt-(end_eQP_time-start_eQP_time)  # Solver has this much time to solve within a control cycle
        prob.update_settings(time_limit=osqp_time_limit)  
        res = prob.solve()
        if res.info.status not in ["solved", "solved inaccurate"]: raise ValueError(f"OSQP did not solve the problem at step {i}! Status: {res.info.status}")
        X, U = extract_solution(res, nx, nu, N)

        end_opt_time = time.perf_counter()

        # 3) Simulate closed-loop step and store trajectories
        start_sim_time = time.perf_counter()

        u0 = U[0]
        x  = sim.step(x, u0)
        x_traj.append(x.copy())
        u_traj.append(u0.copy())

        end_sim_time = time.perf_counter()

        # 4) Per-step prints
        if debugging:
            print(f"Step {i}: X = {X}, U = {U}")
            print(f"Step {i}: x = {x}, u0 = {u0}")
        
        # 5) Profiling updates
        if profiling and i > 0:
            update_timing_stats(printing=False, stats=timing_stats, start_eval_time=start_eQP_time, end_eval_time=end_eQP_time, start_opt_time=start_opt_time, end_opt_time=end_opt_time, start_sim_time=start_sim_time, end_sim_time=end_sim_time)

    if profiling:
        print_timing_summary(timing_stats, N=N, nx=nx, nu=nu, nc=nc, show_system_info=show_system_info)

    # ----------------------------------------
    # ------------ Plot & Animate ------------
    # ---------------------------------------- 

    print("Plotting and saving...")
    x_traj = np.vstack(x_traj)       # (nsim+1, nx)
    u_traj = np.vstack(u_traj)       # (nsim,   nu)
    total_time = dt * np.arange(x_traj.shape[0])  # length nsim+1

    # Plot state and input trajectories (generic, uses constraints internally)
    plot_state_input_trajectories(system, constraints, total_time, x_traj, u_traj, x_ref=x_ref, show=False)

    print("Animating and saving...")
    animate_pendulum(system, constraints, total_time, x_traj, u_traj, show=False, save_gif=True)

if __name__ == "__main__":
    main()
