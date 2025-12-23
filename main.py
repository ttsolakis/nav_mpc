# nav_mpc/main.py

import numpy as np
import osqp
import time

# Import MPC2QP functionality, timing and plotting utilities
from mpc2qp import build_qp, make_workspace, update_qp, extract_solution
from utils.profiling import init_timing_stats, update_timing_stats, print_timing_summary
from utils.print_solution import print_solution
from simulation.simulator import ContinuousSimulator, SimulatorConfig
from simulation.plotting.plotter import plot_state_input_trajectories
from simulation.environment.occupancy_map import OccupancyMapConfig, OccupancyMap2D
from simulation.lidar import LidarSimulator2D, LidarConfig


def main():

    # -----------------------------------
    # ---------- Problem Setup ----------
    # -----------------------------------

    # Import system, objective, constraints and animation via setup_<problem>.py file
    from problem_setup import setup_simple_rover
    problem_name, system, objective, constraints, animation = setup_simple_rover.setup_problem()

    print(f"Setting up: {problem_name}")

    # Enable debugging & profiling info
    debugging = False
    profiling = True
    system_info = True

    # Embedded setting (time-limited solver)
    embedded = True
    
    # Initial state
    x_init = np.array([-2.0, 0.0, 0.0, 0.0, 0.0])

    # Horizon, sampling time
    N  = 40    # steps
    dt = 0.1  # seconds

    # Simulation parameters
    tsim    = 20.0  # seconds
    sim_cfg = SimulatorConfig(dt=dt, method="rk4", substeps=10)

    # -----------------------------------
    # ---------- QP Formulation ---------
    # -----------------------------------

    print("Building QP...")

    start_bQP_time = time.perf_counter()

    # QP matrices and vectors have a standard structure and sparsity:
    # Build everything that is constant once, and prepare the exact memory
    # addresses where the time-varying numbers will be written later.
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
    prob.setup(qp.P_init, qp.q_init, qp.A_init, qp.l_init, qp.u_init, warm_starting=True, verbose=False)

    # Preallocate arrays & warmup numba compilation
    ws = make_workspace(N=N, nx=nx, nu=nu, nc=nc, A_data=qp.A_init.data, l_init=qp.l_init, u_init=qp.u_init, P_data=qp.P_init.data, q_init=qp.q_init)
    X = np.zeros((N+1, nx))
    U = np.zeros((N,   nu))
    update_qp(prob, x, X, U, qp, ws)  
    
    # Store trajectories for plotting / animation
    x_traj = [x.copy()]   
    u_traj = [] 
    X_pred_traj = []

    # Initialize simulator
    sim = ContinuousSimulator(system, sim_cfg)

    # -------------------------------
    # ---------- Lidar sim ----------
    # -------------------------------
    
    occ_cfg = OccupancyMapConfig(map_path="map.png", world_width_m=5.0, occupied_threshold=127, invert=False)
    occ_map = OccupancyMap2D.from_png(occ_cfg)

    lidar_cfg = LidarConfig(range_max=8.0, angle_increment=np.deg2rad(0.72), seed=1, noise_std=0.0, drop_prob=0.0, ray_step=None)
    lidar = LidarSimulator2D(occ_map=occ_map, cfg=lidar_cfg)

    # Optional: store scans for debugging/plotting later
    scans = []

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
            time_limit = dt - (end_eQP_time - start_eQP_time)
            prob.update_settings(time_limit=max(1e-5, time_limit))
        res = prob.solve()
        if res.info.status not in ["solved", "solved inaccurate"]: 
            raise ValueError(f"OSQP did not solve the problem at step {i}! Status: {res.info.status}")
        X, U = extract_solution(res, nx, nu, N)
        u0 = U[0]

        end_opt_time = time.perf_counter()

        # 3) Per-step prints
        if debugging:
            print_solution(i, x, u0, X, U)

        # 4) Simulate closed-loop step and store trajectories
        start_sim_time = time.perf_counter()

        x  = sim.step(x, u0)
        x_traj.append(x.copy())
        u_traj.append(u0.copy())
        X_pred_traj.append(X.copy())
        scan = lidar.scan(np.array([float(x[0]), float(x[1]), float(x[2])], dtype=float))
        scans.append(scan)

        end_sim_time = time.perf_counter()
 
        # 5) Profiling updates
        if profiling and i > 0:
            update_timing_stats(printing=False, stats=timing_stats, start_eval_time=start_eQP_time, end_eval_time=end_eQP_time, start_opt_time=start_opt_time, end_opt_time=end_opt_time, start_sim_time=start_sim_time, end_sim_time=end_sim_time)

    if profiling:
        print_timing_summary(timing_stats, N=N, nx=nx, nu=nu, nc=nc, system_info=system_info)

    # ----------------------------------------
    # ------------ Plot & Animate ------------
    # ---------------------------------------- 

    print("Plotting and saving...")
    plot_state_input_trajectories(system, constraints, dt, x_traj, u_traj, x_ref=objective.x_ref, show=False)

    print("Animating and saving...")
    animation(
        system=system,
        constraints=constraints,
        dt=dt,
        x_traj=x_traj,
        u_traj=u_traj,
        x_goal=objective.x_ref,
        X_pred_traj=X_pred_traj,
        lidar_scans=scans,        
        occ_map=occ_map,              
        occ_resolution=0.05,
        occ_origin=(-10.0, -10.0),
        occ_threshold=127,
        occ_invert=False,
        show=False,
        save_gif=True
    )


if __name__ == "__main__":
    main()
