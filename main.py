# nav_mpc/main.py
import numpy as np
import osqp
import time

# Import MPC2QP functionality, timing and logging utilities
from core.mpc2qp import build_qp, make_workspace, update_qp, solve_qp
from utils.profiling import init_timing_stats, update_timing_stats, print_timing_summary
from utils.online_logger import OnlineLogger

# Import simulation test harness components (simulation, plotting, sensing, path following)
from simulation.simulator import ContinuousSimulator, SimulatorConfig
from simulation.plotting.plotter import plot_state_input_trajectories
from simulation.environment.occupancy_map import OccupancyMapConfig, OccupancyMap2D
from simulation.lidar import LidarSimulator2D, LidarConfig
from simulation.path_following import RRTStarConfig, rrt_star_plan, smooth_and_resample_path, make_reference_builder


def main():

    # -----------------------------------
    # ---------- Problem Setup ----------
    # -----------------------------------

    # Import system, objective, constraints and animation via setup_<problem>.py file
    from core.problem_setup import setup_path_tracking_unicycle
    problem_name, system, objective, constraints, collision, animation = setup_path_tracking_unicycle.setup_problem()

    print(f"Setting up: {problem_name}")

    # Enable profiling info
    profiling = True
    hardware_info = True

    # Solver settings (time-limited solver & printing online)
    embedded = True
    debugging = False
    
    # Initial & goal states
    x_init = np.array([-1.0, -2.0, np.pi / 2, 0.0, 0.0])  # Initial system state
    x_goal = np.array([2.0, 2.0, 0.0, 0.0, 0.0])          # Terminal system state
    velocity_ref = 0.5                                    # Desired cruising speed [m/s]

    # Horizon, sampling time and total simulation time
    N    = 25    # steps
    dt   = 0.1   # seconds
    tsim = 15.0  # seconds

    # Simulation configuration
    sim_cfg = SimulatorConfig(dt=dt, method="rk4", substeps=10)
    sim = ContinuousSimulator(system, sim_cfg)

    # Occupancy map configuration
    occ_cfg = OccupancyMapConfig(map_path="map.png", world_width_m=5.0, occupied_threshold=127, invert=False)
    occ_map = OccupancyMap2D.from_png(occ_cfg)

    # Lidar configuration
    lidar_cfg = LidarConfig(range_max=8.0, angle_increment=np.deg2rad(0.72), seed=1, noise_std=0.0, drop_prob=0.0, ray_step=None)
    lidar = LidarSimulator2D(occ_map=occ_map, cfg=lidar_cfg)

    # Path generator configuration
    rrt_cfg = RRTStarConfig(max_iters=6000, step_size=0.10, neighbor_radius=0.30, goal_sample_rate=0.10, collision_check_step=0.02, seed=1)

    # --------------------------------------
    # -------- Global Path Planning --------
    # --------------------------------------

    print("Computing global path to goal and setup reference builder...")

    start_global_time = time.perf_counter()

    global_path = rrt_star_plan(occ_map=occ_map, start_xy=x_init[:2], goal_xy=x_goal[:2], inflation_radius_m=0.15+0.1, cfg=rrt_cfg)
    global_path = smooth_and_resample_path(global_path, ds= 0.05, smoothing=0.01, k=3)
    ref_builder = make_reference_builder(pos_idx=(0, 1), phi_idx=2, v_idx=3, x_goal=x_goal, v_ref=velocity_ref, goal_indices=[4], window=40, max_lookahead_points=N, stop_radius=0.25, stop_ramp=0.50)

    end_global_time = time.perf_counter()
    
    print(f"Global path computed and reference builder set up in {int(divmod(end_global_time - start_global_time, 60)[0]):02} minutes {int(divmod(end_global_time - start_global_time, 60)[1]):02} seconds.")

    # ------------------------------------
    # ---------- QP Formulation ----------
    # ------------------------------------

    print("Building QP...")

    start_bQP_time = time.perf_counter()

    qp = build_qp(system=system, objective=objective, constraints=constraints, N=N, dt=dt, collision=collision)

    end_bQP_time = time.perf_counter()
    
    print(f"QP built in {int(divmod(end_bQP_time - start_bQP_time, 60)[0]):02} minutes {int(divmod(end_bQP_time - start_bQP_time, 60)[1]):02} seconds.")

    # ----------------------------------------
    # ------------ Initializations------------
    # ----------------------------------------

    print("Initializations...")

    # Problem dimensions
    nx = system.state_dim
    nu = system.input_dim
    nc_sys = qp.nc_sys

    # Initial state, reference and environment scan
    x = x_init.copy()
    Xref_seq = ref_builder(global_path=global_path, x=x, N=N)
    pose = np.array([x[0], x[1], x[2]], dtype=float)
    scan = lidar.scan(pose)
    obstacles_xy = lidar.points_world_from_scan(scan, pose).astype(float, copy=False)
    
    # Initialize OSQP solver and workspace
    prob = osqp.OSQP()
    prob.setup(qp.P_init, qp.q_init, qp.A_init, qp.l_init, qp.u_init, warm_starting=True, verbose=False)
    ws = make_workspace(N=N, nx=nx, nu=nu, nc_sys=nc_sys, nc_col=qp.nc_col, A_data=qp.A_init.data, l_init=qp.l_init, u_init=qp.u_init, P_data=qp.P_init.data, q_init=qp.q_init)
    X = np.tile(x.reshape(1, -1), (N + 1, 1))
    U = np.zeros((N, nu))
    update_qp(prob, x, X, U, qp, ws, Xref_seq, obstacles_xy=obstacles_xy)

    # Logs for profiling/plotting/animation
    timing_stats = init_timing_stats()
    logger = OnlineLogger()
    logger.x_traj.append(x.copy())   # initial condition

    # -----------------------------------
    # ------------ Main Loop ------------
    # -----------------------------------

    print("Running main loop...")

    nsim = int(tsim/dt)
    for i in range(nsim):

        # 1) Update QP around new (x0, x̄, ū, r̄) and new obstacle scan
        start_uQP_time = time.perf_counter()

        A_xy, b_xy = update_qp(prob, x, X, U, qp, ws, Xref_seq, obstacles_xy=obstacles_xy)
        
        end_uQP_time = time.perf_counter()

        # 2) Solve current QP and extract solution
        start_sQP_time = time.perf_counter()

        time_limit = dt - (end_uQP_time - start_uQP_time)
        X, U, u0 = solve_qp(prob, nx, nu, N, embedded, time_limit, i, debugging=debugging, x=x)

        end_sQP_time = time.perf_counter()

        # 3) Simulate closed-loop step, lidar scanning, build reference, and log data for plotting/animation
        start_sim_time = time.perf_counter()

        x  = sim.step(x, u0)
        Xref_seq = ref_builder(global_path=global_path, x=x, N=N)
        pose = np.array([x[0], x[1], x[2]], dtype=float)
        scan = lidar.scan(pose)
        obstacles_xy = lidar.points_world_from_scan(scan, pose).astype(float, copy=False)
        logger.log(x=x, u0=u0, X=X, Xref_seq=Xref_seq, scan=scan, A_xy=A_xy, b=b_xy)

        end_sim_time = time.perf_counter()
 
        # 4) Profiling updates
        if profiling and i > 0:
            update_timing_stats(printing=False, stats=timing_stats, start_eval_time=start_uQP_time, end_eval_time=end_uQP_time, start_opt_time=start_sQP_time, end_opt_time=end_sQP_time, start_sim_time=start_sim_time, end_sim_time=end_sim_time)

    if profiling:
        print_timing_summary(timing_stats, N=N, nx=nx, nu=nu, nc=qp.nc_sys + qp.nc_col, system_info=hardware_info, dt=dt)

    # ----------------------------------------
    # ------------ Plot & Animate ------------
    # ---------------------------------------- 

    print("Plotting and saving...")
    plot_state_input_trajectories(system, constraints, dt, logger.x_traj, logger.u_traj, x_ref=x_goal, show=False)

    print("Animating and saving...")
    animation(system=system, constraints=constraints, dt=dt, x_traj=logger.x_traj, u_traj=logger.u_traj, x_goal=x_goal, X_pred_traj=logger.X_pred_traj, X_ref_traj=logger.X_ref_traj, lidar_scans=logger.scans, occ_map=occ_map, global_path=global_path, collision=collision, col_bounds_traj=logger.col_bounds_traj, col_Axy_traj=logger.col_Axy_traj, show=False, save_gif=True)

if __name__ == "__main__":
    main()
