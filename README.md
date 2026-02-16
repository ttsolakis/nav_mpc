# 🧭 nav_mpc — Realtime Nonlinear MPC for Autonomous Navigation

**nav_mpc** is a lightweight, high-performance Python framework for navigation using **real-time** Model Predictive Control (MPC). MPC is an attractive control approach because it naturally handles constraints of different types and flexibly incorporates diverse control objectives. 

However, nonlinear MPC is often computationally expensive. For many systems, the solver time can exceed the control-loop period, especially on embedded hardware where computation is limited. 

In contrast, Quadratic Programs (QPs) can be solved extremely quickly, and OSQP is particularly well suited for this. The core idea behind this framework is to convert a fully nonlinear MPC problem into a Linear Time-Varying (LTV) MPC problem that can be solved fast enough so that the linearization error remains small and does not degrade system performance.

The framework combines:

- **Symbolic definition**: Users define the nonlinear dynamics, constraints, and objective symbolically, exactly as they would on paper. 
- **Automatic QP formulation** : The framework linearizes the original problem and constructs the corresponding parametric QP approximation automatically. 
- **Cython compilation**: All functions that must be evaluated online for the parametric QP are compiled with Cython to achieve optimal runtime performance. 
- **Real-time OSQP solving**: The QP is solved with OSQP extremely fast, using a configurable time limit to guarantee real-time feasibility. 
- **Integrated simulator for rapid prototyping**: The same symbolic model used by the MPC is also used for simulation, with built-in plotting and animation tools to iterate quickly for rapid development before deploying on embedded hardware.
- **Integrated ROS2 functionality**: The core functionality of nav_mpc and the simulation harness are wrapped with ROS2 nodes to test the framework in asynchronous, ROS-style information exchange.

Together, these components enable deterministic nonlinear MPC on modest hardware, suitable for embedded robotic applications such as **UGVs**, **USVs**, **UAVs**, and more.

<p align="center">
  <img src="docs/nav_mpc_bd.png" width="900">
</p>

<p align="center"><em>Architecture overview of nav_mpc: symbolic nonlinear MPC compiled into real-time LTV-QP control with integrated simulation loop.</em></p>

---

## ✨ Key Features

### 🔧 1. Fully parametric, symbolic MPC pipeline
- Symbolic linearization around operating trajectories
- Automatic Jacobians and discrete-time dynamics
- QP constructed explicitly for transparency & speed

### ⚡ 2. C-accelerated online QP evaluation (SymPy autowrap + Cython) and OSQP solution
- Expensive symbolic expressions compiled to native machine code
- Fast online QP assembly with Cython
- Fast online QP solution with OSQP
- Suitable for Jetson / Raspberry Pi / embedded CPUs

### 🧱 3. Clean modular architecture
```
nav_mpc/
├── core/               # symbolic problem definitions + MPC→QP core
├── simulation/         # map/path/lidar + simulator + plotting/animations
├── utils/              # profiling, debugging, logging, system info
├── nav_mpc_ros/        # ROS 2 package wrapping core/ + simulation/ as nodes
├── docs/               # documents and examples
└── main.py             # ROS-agnostic MPC runner (example entry point)
```

### 🔌 **4. Extensible to arbitrary systems**
- Simple pendulum
- Double pendulum 
- Kinematic Rover (UGV)
- Cybership (ASV)

--- 

## 🎯 Why nav_mpc? 

**nav_mpc** provides:

- An easy way to define a full nonlinear MPC problem — dynamics, constraints, and objectives are written symbolically, just like on paper. 
- A fast development workflow in Python with integrated simulation and result generation, combined with Cython compilation for ultra-fast numerical evaluation. 
- Real-time performance: the controller runs ultra-fast with deterministic timing, making it suitable for embedded hardware with tight control-loop deadlines. 
- A clean, minimal set of dependencies and a research-friendly architecture that enables rapid prototyping, fast iteration, and straightforward extension to new robotic systems. 
- A smooth transition to embedded hardware via ROS2 wrappers. 

---

## 🚀 Getting Started

### 🧪 Install and Run in pure Python (ROS-agnostic)

```bash
git clone https://github.com/ttsolakis/nav_mpc.git
cd nav_mpc
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```
This will:
- linearize the nonlinear problem and build the parametric QP offline
- run realtime LTV-MPC in closed loop
- print detailed timing statistics
- generate plots/animations under `results/` (auto-generated at root)

---

### 🤖 Build and Run with ROS2

> This section assumes a working ROS2 installation and an existing colcon workspace (e.g., dev_ws).
> Tested with ROS2 Jazzy on Ubuntu 24.04.

```bash
cd ~/dev_ws
source /opt/ros/<distro>/setup.bash
colcon build --packages-select nav_mpc_ros --symlink-install
source install/setup.bash
ros2 launch nav_mpc_ros nav_mpc_sim.launch.py
```
This will:
- start the full ROS2 simulation pipeline (map, path, lidar, simulator, MPC)
- ROS2 nodes wrap and reuse the same `core/` MPC logic and `simulation/` components
- Spawn RVIZ for visualization

---

## 🕹️ Examples

Examples run with:

OS:       Linux 6.14.0-37-generic  
Machine:  x86_64  
CPU:      Intel(R) Core(TM) i7-7500U CPU @ 2.70GHz  
Cores:    4 logical

---

### Double Pendulum

Double pendulum swing-up and stabilization with LTV-MPC:

<img src="examples/double_pendulum/double_pendulum_animation.gif" width="400">

Performance with N = 40, dt = 0.02 s on a laptop CPU:

| Stage | Mean | Min | Max |
|-------|-------|-------|-------|
| QP eval | 1.19 ms | 1.12 ms | 2.71 ms |
| QP solve | 0.26 ms | 0.23 ms | 1.47 ms |
| Total MPC | **1.45 ms** | **1.35 ms** | **3.88 ms** |

Notice that Max time for Total MPC can stay deterministically below dt 
while getting optimal performance from OSQP (3.88 ms << 20 ms).

---

### Simple Rover

Simple kinematic rover (unicycle) model path tracking with LTV-MPC and 36 half-space corridor constraints per stage:

<img src="docs/examples/simple_unicycle/unicycle_animation.gif" width="400">

Performance with N = 25, dt = 0.1 s on a laptop CPU:

| Stage | Mean | Min | Max |
|-------|-------|-------|-------|
| QP eval | 3.06 ms | 2.77 ms | 5.44 ms |
| QP solve | 0.71 ms | 0.57 ms | 3.05 ms |
| Total MPC | **3.77 ms** | **3.36 ms** | **6.21 ms** |

Notice that Max time for Total MPC can stay deterministically below dt 
while getting optimal performance from OSQP (6.21 ms << 100 ms).

---

### Simple Rover with ROS2

Simple rover running with the full ROS2 navigation pipeline:

<img src="docs/examples/nav_mpc_ros/nav_mpc_ros.gif" width="400">


---

### Cybership

Cybership model (ASV) path tracking with LTV-MPC and 16 half-space corridor constraints per stage:

<img src="docs/examples/cybership/cybership_animation.gif" width="400">

Performance with N = 30, dt = 0.1 s on a laptop CPU:

| Stage | Mean | Min | Max |
|-------|-------|-------|-------|
| QP eval | 2.93 ms | 2.56 ms | 4.73 ms |
| QP solve | 1.38 ms | 0.61 ms | 22.56 ms |
| Total MPC | **4.32 ms** | **3.18 ms** | **25.73 ms** |

Notice that Max time for Total MPC can stay deterministically below dt 
while getting optimal performance from OSQP (25.73 ms < 100 ms) even
for a complex nonlinear hydrodynamic ASV model.

---


## 📄 License — MIT

Permissive, suitable for commercial + academic use.

---

## 📖 Citation

If you use this framework in academic work, please cite or link to:

Anastasios Tsolakis, *nav_mpc: Realtime Nonlinear MPC via LTV-MPC*, 
GitHub repository, 2025.


---

## 📬 Contact

Anastasios (Tasos) Tsolakis  
📧 tas.tsolakis@gmail.com  
🌐 https://ttsolakis.github.io

---

> 🚧 **Work in Progress**
>
> This project is under active development. 
> APIs, file structure, and features may change.
> The framework is functional and examples run end-to-end, but some components are still evolving.
