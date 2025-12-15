# nav_mpc/main.py

import numpy as np
import osqp
import time

# Import MPC2QP functionality, timing and plotting utilities (generic stuff)
# from mpc2qp import build_linear_constraints, build_quadratic_objective, set_qp, update_qp, extract_solution
from mpc2qp import build_qp, update_qp_fast, extract_solution
from utils.profiling import init_timing_stats, update_timing_stats, print_timing_summary
from simulation.simulator import ContinuousSimulator, SimulatorConfig
from simulation.plotting.plotter import plot_state_input_trajectories

# Import system, objective, constraints and animation (user-defined for the specific problem)
from models.double_pendulum_model import DoublePendulumModel
from objectives.double_pendulum_objective import DoublePendulumObjective
from constraints.system_constraints.double_pendulum_sys_constraints import DoublePendulumSystemConstraints
from simulation.animation.double_pendulum_animation import animate_double_pendulum

def main():
    # -----------------------------------
    # ---------- Problem Setup ----------
    # -----------------------------------

    # Enable debugging & profiling
    debugging = False
    profiling = True
    show_system_info = True

    # Use Cython for speed in embedded systems (online functions ~5x times faster than pure Python)
    use_cython = False
    
    # System, objective, constraints
    system      = DoublePendulumModel()
    objective   = DoublePendulumObjective(system)
    constraints = DoublePendulumSystemConstraints(system)
    
    # Initial and reference states
    x_init = np.array([0.0, 0.0, 0.0, 0.0])
    x_ref  = np.array([np.pi, np.pi, 0.0, 0.0])

    # Horizon and sampling time
    N  = 100
    dt = 0.02

    # Simulation parameters
    nsim    = 300
    sim_cfg = SimulatorConfig(dt=dt, method="rk4", substeps=10)
    sim     = ContinuousSimulator(system, sim_cfg)

    # -----------------------------------
    # ---------- QP Formulation ---------
    # -----------------------------------

    print("Building QP...")

    start_bQP_time = time.perf_counter()

    # QP matrices and vectors have a standard structure and sparsity. So:
    # Build everything that is constant once, and prepare the exact memory addresses where the time-varying numbers will be written later.”
    qp = build_qp(system=system, objective=objective, constraints=constraints, N=N, dt=dt)

    end_bQP_time = time.perf_counter()
    duration_seconds = end_bQP_time - start_bQP_time
    minutes, seconds = divmod(duration_seconds, 60)
    print(f"QP built in {int(minutes):02} minutes {int(seconds):02} seconds.")


    # # -----------------------------------
    # # ------------ Main Loop ------------
    # # -----------------------------------

    # print("Initializations...")

    # # Initialize timing stats
    # timing_stats = init_timing_stats()

    # # Problem dimension
    # nx = system.state_dim
    # nu = system.input_dim
    # nc = constraints.constraints_dim

    # # Initial state
    # x = x_init.copy()

    # # Initialize OSQP solver
    # prob = osqp.OSQP()
    # prob.setup(P0, q0, A, l0, u0, warm_starting=True, verbose=False)

    # # -----------------------------
    # # Preallocate update arrays (once)  ✅ MUST COME BEFORE WARMUP
    # # -----------------------------
    # Ax_new = A.data.copy()
    # l_new  = l0.copy()
    # u_new  = u0.copy()

    # # Preallocate scratch (once) — ✅ MUST COME BEFORE WARMUP
    # Xbar   = np.empty((N+1, nx), dtype=float)
    # Ubar   = np.empty((N,   nu), dtype=float)
    # Ad_all = np.empty((N, nx, nx), dtype=float)
    # Bd_all = np.empty((N, nx, nu), dtype=float)
    # cd_all = np.empty((N, nx), dtype=float)

    # update_timing = {"shift":0.0, "linearize":0.0, "fill":0.0, "osqp_update":0.0}

    # # -----------------------------
    # # Warmup numba compilation ✅ now variables exist
    # # -----------------------------
    # dummy_X = np.zeros((N+1, nx))
    # dummy_U = np.zeros((N,   nu))
    # update_qp_fast(
    #     prob=prob, x=x, X=dummy_X, U=dummy_U,
    #     Ax_new=Ax_new, l_new=l_new, u_new=u_new,
    #     Xbar=Xbar, Ubar=Ubar, Ad_all=Ad_all, Bd_all=Bd_all, cd_all=cd_all,
    #     Ax_template=Ax_template, l_template=l_template, u_template=u_template,
    #     idx_Ad=idx_Ad, idx_Bd=idx_Bd,
    #     Ad_fun=Ad_fun, Bd_fun=Bd_fun, cd_fun=cd_fun,
    #     N=N, nx=nx, nu=nu,
    #     timing=None
    # )

    # update_timing = {"shift":0.0, "linearize":0.0, "fill":0.0, "osqp_update":0.0}

    # # Store trajectories for plotting / animation
    # x_traj = [x.copy()]   
    # u_traj = [] 

    # print("Running main loop...")
    # for i in range(nsim):

    #     # 1) Evaluate QP around new (x0, x̄, ū)
    #     start_eQP_time = time.perf_counter()

    #     if i > 0:
    #         update_qp_fast(
    #             prob=prob,
    #             x=x,
    #             X=X,
    #             U=U,
    #             Ax_new=Ax_new,
    #             l_new=l_new,
    #             u_new=u_new,
    #             Xbar=Xbar,
    #             Ubar=Ubar,
    #             Ad_all=Ad_all,
    #             Bd_all=Bd_all,
    #             cd_all=cd_all,
    #             Ax_template=Ax_template,
    #             l_template=l_template,
    #             u_template=u_template,
    #             idx_Ad=idx_Ad,
    #             idx_Bd=idx_Bd,
    #             Ad_fun=Ad_fun,
    #             Bd_fun=Bd_fun,
    #             cd_fun=cd_fun,
    #             N=N,
    #             nx=nx,
    #             nu=nu,
    #             timing=update_timing,   # NEW
    #         )
                      
    #     end_eQP_time = time.perf_counter()

    #     # 2) Solve current QP and extract solution
    #     start_opt_time = time.perf_counter()

    #     if use_cython:
    #         osqp_time_limit = dt-(end_eQP_time-start_eQP_time)  # Solver has this much time to solve within a control cycle
    #         prob.update_settings(time_limit=osqp_time_limit)  
    #     res = prob.solve()
    #     if res.info.status not in ["solved", "solved inaccurate"]: raise ValueError(f"OSQP did not solve the problem at step {i}! Status: {res.info.status}")
    #     X, U = extract_solution(res, nx, nu, N)

    #     end_opt_time = time.perf_counter()

    #     # 3) Simulate closed-loop step and store trajectories
    #     start_sim_time = time.perf_counter()

    #     u0 = U[0]
    #     x  = sim.step(x, u0)
    #     x_traj.append(x.copy())
    #     u_traj.append(u0.copy())

    #     end_sim_time = time.perf_counter()

    #     # 4) Per-step prints
    #     if debugging:
    #         print(f"Step {i}: X = {X}, U = {U}")
    #         print(f"Step {i}: x = {x}, u0 = {u0}")
        
    #     # 5) Profiling updates
    #     if profiling and i > 0:
    #         update_timing_stats(printing=False, stats=timing_stats, start_eval_time=start_eQP_time, end_eval_time=end_eQP_time, start_opt_time=start_opt_time, end_opt_time=end_opt_time, start_sim_time=start_sim_time, end_sim_time=end_sim_time)

    # if profiling:
    #     print_timing_summary(timing_stats, N=N, nx=nx, nu=nu, nc=nc, show_system_info=show_system_info)


    # total_updates = nsim - 1
    # print("\n=== update_qp_fast micro-timing (avg per call) ===")
    # for k,v in update_timing.items():
    #     print(f"{k:>12s}: {1e3*v/total_updates:8.3f} ms")

    # # ----------------------------------------
    # # ------------ Plot & Animate ------------
    # # ---------------------------------------- 

    # print("Plotting and saving...")
    # x_traj = np.vstack(x_traj)       # (nsim+1, nx)
    # u_traj = np.vstack(u_traj)       # (nsim,   nu)
    # total_time = dt * np.arange(x_traj.shape[0])  # length nsim+1

    # # Plot state and input trajectories (generic, uses constraints internally)
    # plot_state_input_trajectories(system, constraints, total_time, x_traj, u_traj, x_ref=x_ref, show=False)

    # print("Animating and saving...")
    # animate_double_pendulum(system, constraints, total_time, x_traj, u_traj, show=False, save_gif=True)

if __name__ == "__main__":
    main()
