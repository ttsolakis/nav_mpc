# nav_mpc/main.py

import numpy as np
import osqp
import time

# Import MPC2QP functionality and timing utilities
from mpc2qp import build_qp, make_workspace, update_qp, extract_solution
from utils.profiling import init_timing_stats, update_timing_stats, print_timing_summary
from utils.print_solution import print_solution

# Import simulation test harness components (simulation, plotting, sensing, guidance)
from simulation.simulator import ContinuousSimulator, SimulatorConfig
from simulation.plotting.plotter import plot_state_input_trajectories
from simulation.environment.occupancy_map import OccupancyMapConfig, OccupancyMap2D
from simulation.lidar import LidarSimulator2D, LidarConfig
from simulation.path_generators import rrt_star_plan, RRTStarConfig

def build_reference_from_path(
    global_path: np.ndarray,
    x: np.ndarray,
    N: int,
    path_idx: int,
    window: int = 40,
    max_lookahead_points: int | None = 12,
) -> tuple[np.ndarray, int]:
    assert global_path.ndim == 2 and global_path.shape[1] == 2
    M = global_path.shape[0]
    nx = x.shape[0]

    path_idx = int(max(0, min(path_idx, M - 1)))
    i_start = path_idx
    i_end = min(M, path_idx + window + 1)

    p = np.array([x[0], x[1]], dtype=float)
    segment = global_path[i_start:i_end]
    if segment.shape[0] == 0:
        i0 = path_idx
    else:
        d2 = np.sum((segment - p[None, :]) ** 2, axis=1)
        i0 = i_start + int(np.argmin(d2))

    if i0 < path_idx:
        i0 = path_idx
    new_path_idx = i0

    # limit how far ahead we reference (prevents "runaway" horizon)
    if max_lookahead_points is None:
        i_max = M - 1
    else:
        i_max = min(M - 1, new_path_idx + int(max_lookahead_points))

    Xref = np.zeros((N + 1, nx), dtype=float)
    for k in range(N + 1):
        idx = min(new_path_idx + k, i_max)  # <-- clamp by i_max
        Xref[k, 0] = global_path[idx, 0]
        Xref[k, 1] = global_path[idx, 1]
        Xref[k, 3] = 0.0
        Xref[k, 4] = 0.0

    for k in range(N):
        dx = Xref[k + 1, 0] - Xref[k, 0]
        dy = Xref[k + 1, 1] - Xref[k, 1]
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            Xref[k, 2] = Xref[k - 1, 2] if k > 0 else float(x[2])
        else:
            Xref[k, 2] = np.arctan2(dy, dx)

    Xref[N, 2] = Xref[N - 1, 2] if N > 0 else float(x[2])

    return Xref, new_path_idx



# Add these imports at the top of nav_mpc/main.py
from scipy.interpolate import splprep, splev

def smooth_and_resample_path(
    path_xy: np.ndarray,
    *,
    ds: float = 0.05,
    smoothing: float = 0.01,
    k: int = 3,
    dense_factor: int = 30,
) -> np.ndarray:
    """
    Fit a parametric spline to path_xy and resample points approximately equidistant
    in arc-length with spacing ds.

    Args:
        path_xy: (M,2) polyline waypoints
        ds: desired spacing [m] between consecutive resampled points (geometry-based)
        smoothing: splprep smoothing factor s
        k: spline degree
        dense_factor: internal oversampling factor to build arc-length map

    Returns:
        path_resampled: (K,2) points with ~ds spacing along arc-length.
    """
    path_xy = np.asarray(path_xy, dtype=float)
    if path_xy.ndim != 2 or path_xy.shape[1] != 2:
        raise ValueError(f"path_xy must be (M,2), got {path_xy.shape}")
    if ds <= 0:
        raise ValueError("ds must be > 0")

    # Remove consecutive duplicates
    diffs = np.diff(path_xy, axis=0)
    keep = np.ones(path_xy.shape[0], dtype=bool)
    keep[1:] = np.linalg.norm(diffs, axis=1) > 1e-9
    path_xy = path_xy[keep]

    if path_xy.shape[0] < 2:
        raise ValueError("Need at least 2 distinct points to fit a spline.")

    # spline degree constraints: splprep requires m > k
    k = int(np.clip(k, 1, 5))
    k = min(k, path_xy.shape[0] - 1)

    x = path_xy[:, 0]
    y = path_xy[:, 1]

    # Fit parametric spline: x(u), y(u)
    tck, _ = splprep([x, y], s=float(smoothing), k=k)

    # Dense sampling in u to build an arc-length map
    M = path_xy.shape[0]
    n_dense = max(300, dense_factor * M)
    u_dense = np.linspace(0.0, 1.0, n_dense)
    x_dense, y_dense = splev(u_dense, tck)
    dense = np.column_stack([x_dense, y_dense])

    seg = np.diff(dense, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    s_cum = np.concatenate([[0.0], np.cumsum(seglen)])
    L = float(s_cum[-1])

    if L < 1e-9:
        return dense[:1].copy()

    # Target arc-length positions: 0, ds, 2ds, ...
    s_targets = np.arange(0.0, L, ds)
    if s_targets.size == 0 or s_targets[-1] < L:
        s_targets = np.append(s_targets, L)

    # Invert s(u) via interpolation on dense samples
    u_targets = np.interp(s_targets, s_cum, u_dense)

    # Evaluate spline at u_targets
    x_out, y_out = splev(u_targets, tck)
    out = np.column_stack([x_out, y_out])

    return out


def main():

    # -----------------------------------
    # ---------- Problem Setup ----------
    # -----------------------------------

    # Import system, objective, constraints and animation via setup_<problem>.py file
    from problem_setup import setup_path_tracking_rover
    problem_name, system, objective, constraints, animation = setup_path_tracking_rover.setup_problem()

    print(f"Setting up: {problem_name}")

    # Enable debugging & profiling info
    debugging = False
    profiling = True
    system_info = True

    # Embedded setting (time-limited solver)
    embedded = True
    
    # Initial state
    x_init = np.array([-1.0, -2.0, np.pi/2, 0.0, 0.0])

    # Goal state
    x_goal= np.array([2.0, 2.0, 0.0, 0.0, 0.0])
    
    # Horizon, sampling time and total simulation time
    N    = 20    # steps
    dt   = 0.1   # seconds
    tsim = 40.0  # seconds

    # Simulation configuration
    sim_cfg = SimulatorConfig(dt=dt, method="rk4", substeps=10)

    # Occupancy map configuration
    occ_cfg = OccupancyMapConfig(map_path="map.png", world_width_m=5.0, occupied_threshold=127, invert=False)

    # Lidar configuration
    lidar_cfg = LidarConfig(range_max=8.0, angle_increment=np.deg2rad(0.72), seed=1, noise_std=0.0, drop_prob=0.0, ray_step=None)

    # Path generator configuration
    rrt_cfg = RRTStarConfig(max_iters=6000, step_size=0.10, neighbor_radius=0.30, goal_sample_rate=0.10, collision_check_step=0.02, seed=1)

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

    # Initialize simulator
    sim = ContinuousSimulator(system, sim_cfg)

    # Initialize occupancy map
    occ_map = OccupancyMap2D.from_png(occ_cfg)
    
    # Initialize lidar simulator
    lidar = LidarSimulator2D(occ_map=occ_map, cfg=lidar_cfg)

    # Initialize timing stats
    timing_stats = init_timing_stats()

    # Problem dimensions
    nx = system.state_dim
    nu = system.input_dim
    nc = constraints.constraints_dim

    # Initial state
    x = x_init.copy()

    # Reference state sequence (N+1, nx)
    Xref_seq = np.tile(x_goal.reshape(1, -1), (N + 1, 1))

    # Initialize OSQP solver
    prob = osqp.OSQP()
    prob.setup(qp.P_init, qp.q_init, qp.A_init, qp.l_init, qp.u_init, warm_starting=True, verbose=False)

    # Preallocate arrays & warmup numba compilation
    ws = make_workspace(N=N, nx=nx, nu=nu, nc=nc, A_data=qp.A_init.data, l_init=qp.l_init, u_init=qp.u_init, P_data=qp.P_init.data, q_init=qp.q_init)
    X = np.zeros((N+1, nx))
    U = np.zeros((N,   nu))
    update_qp(prob, x, X, U, qp, ws, Xref_seq)
    
    # Store data for plotting & animation
    x_traj = [x.copy()]   
    u_traj = [] 
    X_pred_traj = []
    scans = []

    # Compute global path to goal
    print("Computing global path to goal...")
    robot_radius = 0.15
    margin = 0.10
    path_idx = 0
    ref_window = 40  # tune this
    start_xy = x_init[:2]
    goal_xy  = x_goal[:2] 
    global_path = rrt_star_plan(occ_map=occ_map, start_xy=start_xy, goal_xy=goal_xy, inflation_radius_m=robot_radius+margin, cfg=rrt_cfg)
    global_path = smooth_and_resample_path(global_path, ds= 0.2, smoothing=0.01, k=3)

    print("Running main loop...")

    nsim = int(tsim/dt)
    for i in range(nsim):

        # 1) Evaluate QP around new (x0, x̄, ū, r̄)
        start_eQP_time = time.perf_counter()

        Xref_seq, path_idx = build_reference_from_path(global_path=global_path, x=x, N=N, path_idx=path_idx, window=ref_window, max_lookahead_points=12)

        update_qp(prob, x, X, U, qp, ws, Xref_seq)
                      
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
        x_goal=x_goal,
        X_pred_traj=X_pred_traj,
        lidar_scans=scans,
        occ_map=occ_map,
        global_path=global_path,
        show=False,
        save_gif=True,
    )


if __name__ == "__main__":
    main()
