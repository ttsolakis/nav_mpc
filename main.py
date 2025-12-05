# nav_mpc/main.py

import numpy as np
import osqp
import time

from models.simple_pendulum_model import SimplePendulumModel
from objectives.simple_pendulum_objective import SimplePendulumObjective
from constraints.system_constraints.simple_pendulum_sys_constraints import SimplePendulumSystemConstraints 
from qp_formulation.qp_formulation import build_linearized_system
from simulation.simulator import ContinuousSimulator, SimulatorConfig
from planner.planner import assemble_ltv_mpc_qp

from simulation.plotting.plotter import plot_state_input_trajectories
from simulation.animation.simple_pendulum_animation import animate_pendulum


def main():
    # -----------------------------------
    # ---------- Problem Setup ----------
    # -----------------------------------

    # System, objective, constraints
    system      = SimplePendulumModel()
    objective   = SimplePendulumObjective(system)
    constraints = SimplePendulumSystemConstraints(system)
    
    # Initial and reference states
    x_init = np.array([np.pi-0.1, 0.0])
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
    A_fun, B_fun, c_fun = build_linearized_system(system)

    # Initial linearization points: all zeros
    x_bar_seq = np.zeros((N + 1, system.state_dim))
    u_bar_seq = np.zeros((N, system.input_dim))

    # Assemble initial QP
    P, q, A, l, u = assemble_ltv_mpc_qp(system, A_fun, B_fun, c_fun, objective, constraints, N, dt, x_init, x_ref, x_bar_seq, u_bar_seq)

    # Initialize OSQP
    prob = osqp.OSQP()
    prob.setup(P, q, A, l, u, warm_starting=True)

    # -----------------------------------
    # ------------ Main Loop ------------
    # -----------------------------------

    x = x_init.copy()

    # store trajectories (lists first)
    x_traj = [x.copy()]   # include initial state
    u_traj = []           # nsim control inputs

    total_opt_time = 0.0
    total_sim_time = 0.0
    total_assembly_time = 0.0
    total_setup_time = 0.0
    
    for i in range(nsim):

        start_opt_time = time.perf_counter()

        # 1) Solve current QP
        res = prob.solve()
        if res.info.status != "solved":
            raise ValueError(f"OSQP did not solve the problem at step {i}! Status: {res.info.status}")

        z = res.x

        # 2) Extract predicted state and input trajectories
        X_flat = z[0:(N + 1) * system.state_dim]
        U_flat = z[(N + 1) * system.state_dim:]
        X = X_flat.reshape(N + 1, system.state_dim)
        U = U_flat.reshape(N, system.input_dim)

        end_opt_time = time.perf_counter()
        opt_time = (end_opt_time - start_opt_time)*1e3  # ms

        start_sim_time = time.perf_counter()

        # 3) Simulate closed-loop step and store trajectories
        u0 = U[0]
        x  = sim.step(x, u0)
        x_traj.append(x.copy())
        u_traj.append(u0.copy())
        print(f"Step {i}: x = {x}, u0 = {u0}")

        end_sim_time = time.perf_counter()
        sim_time = (end_sim_time - start_sim_time)*1e3  # ms
  
        start_assembly_time = time.perf_counter()
        # 5) Rebuild the QP around new (x̄, ū)
        x_init    = x.copy()
        x_bar_seq = X
        u_bar_seq = U
        P, q, A, l, u = assemble_ltv_mpc_qp(system, A_fun, B_fun, c_fun, objective, constraints, N, dt, x_init, x_ref, x_bar_seq, u_bar_seq)

        end_assembly_time = time.perf_counter()
        assembly_time = (end_assembly_time - start_assembly_time)*1e3  # ms

        # For now: re-create OSQP each step (simple & correct).
        # Later you can optimize this with prob.update() and fixed sparsity.
        start_setup_time = time.perf_counter()

        prob = osqp.OSQP()
        prob.setup(P, q, A, l, u, warm_starting=True)

        end_setup_time = time.perf_counter()
        setup_time = (end_setup_time - start_setup_time)*1e3  # ms

        total_opt_time += opt_time
        total_sim_time += sim_time
        total_assembly_time += assembly_time
        total_setup_time += setup_time

        
    avg_opt_time = total_opt_time / nsim
    avg_sim_time = total_sim_time / nsim
    avg_assembly_time = total_assembly_time / nsim
    avg_setup_time = total_setup_time / nsim

    print(f"\nAverage optimization time: {avg_opt_time:.3f} ms")
    print(f"Average simulation time:     {avg_sim_time:.3f} ms")
    print(f"Average QP assembly time:    {avg_assembly_time:.3f} ms")
    print(f"Average QP setup time:     {avg_setup_time:.3f} ms")

    # ----------------------------------------
    # ------------ Plot & Animate ------------
    # ---------------------------------------- 
    # Prepare results for plotting and animation
    x_traj = np.vstack(x_traj)       # shape (nsim+1, nx)
    u_traj = np.vstack(u_traj)       # shape (nsim,   nu)
    total_time = dt * np.arange(x_traj.shape[0])  # length nsim+1

    # Plot state and input trajectories (generic)
    plot_state_input_trajectories(system, total_time, x_traj, u_traj, x_ref=x_ref, show=False)

    # For the pendulum, torque bounds are in SimplePendulumSystemConstraints
    _, _, umin, umax = constraints.get_bounds()
    umax_scalar = float(np.max(np.abs(umax)))

    animate_pendulum(system, total_time, x_traj, u_traj, umax=umax_scalar, show=False)


if __name__ == "__main__":
    main()
