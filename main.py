# nav_mpc/main.py

import numpy as np
import osqp
import time

# Import MPC2QP functionality, timing and plotting utilities (generic stuff)
from mpc2qp import build_qp, make_workspace, update_qp, extract_solution
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

    # Embedded setting (time-limited solver)
    embedded = True
    
    # System, objective, constraints
    system      = SimplePendulumModel()
    objective   = SimplePendulumObjective(system)
    constraints = SimplePendulumSystemConstraints(system)
    
    # Initial and reference states
    x_init = np.array([0.0, 0.0])
    x_ref  = np.array([np.pi, 0.0])

    # Horizon, sampling time
    N  = 40   # Steps
    dt = 0.02 # seconds

    # Simulation parameters
    tsim    = 2.0  # seconds
    sim_cfg = SimulatorConfig(dt=dt, method="rk4", substeps=10)

    # -----------------------------------
    # ---------- QP Formulation ---------
    # -----------------------------------

    print("Building QP...")

    start_bQP_time = time.perf_counter()

    # QP matrices and vectors have a standard structure and sparsity:
    # Build everything that is constant once, and prepare the exact memory addresses where the time-varying numbers will be written later.”
    qp = build_qp(system=system, objective=objective, constraints=constraints, N=N, dt=dt)

    end_bQP_time = time.perf_counter()
    
    print(f"QP built in {int(divmod(end_bQP_time - start_bQP_time, 60)[0]):02} minutes {int(divmod(end_bQP_time - start_bQP_time, 60)[1]):02} seconds.")

    # -----------------------------------
    # ------------ Main Loop ------------
    # -----------------------------------

    print("Initializations...")

    # Initialize timing stats
    timing_stats = init_timing_stats()

    # Problem dimensions
    nx = system.state_dim
    nu = system.input_dim
    nc = constraints.constraints_dim

    # Initial state
    x = x_init.copy()

    # Initialize OSQP solver
    prob = osqp.OSQP()
    prob.setup(qp.P0, qp.q0, qp.A, qp.l0, qp.u0, warm_starting=True, verbose=False)

    # Preallocate arrays & warmup numba compilation
    ws = make_workspace(N=N, nx=nx, nu=nu, nc=nc, A_data=qp.A.data, l0=qp.l0, u0=qp.u0)
    X = np.zeros((N+1, nx))
    U = np.zeros((N,   nu))
    update_qp(prob, x, X, U, qp, ws)  
    
    # Store trajectories for plotting / animation
    x_traj = [x.copy()]   
    u_traj = [] 

    # Initialize simulator
    sim = ContinuousSimulator(system, sim_cfg)

    print("Running main loop...")

    nsim = int(tsim/dt)
    for i in range(nsim):

        # 1) Evaluate QP around new (x0, x̄, ū)
        start_eQP_time = time.perf_counter()

        update_qp(prob, x, X, U, qp, ws)
                      
        end_eQP_time = time.perf_counter()

        # 2) Solve current QP and extract solution
        start_opt_time = time.perf_counter()

        if embedded: 
            prob.update_settings(time_limit=dt-(end_eQP_time-start_eQP_time))  # set time limit for embedded setting
        res = prob.solve()
        if res.info.status not in ["solved", "solved inaccurate"]: 
            raise ValueError(f"OSQP did not solve the problem at step {i}! Status: {res.info.status}")
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
    plot_state_input_trajectories(system, constraints, dt, x_traj, u_traj, x_ref=x_ref, show=False)

    print("Animating and saving...")
    animate_pendulum(system, constraints, dt, x_traj, u_traj, show=False, save_gif=True)

if __name__ == "__main__":
    main()
