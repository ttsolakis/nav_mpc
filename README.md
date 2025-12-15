# 🧭 nav_mpc — Realtime Nonlinear MPC for Autonomous Navigation

**nav_mpc** is a lightweight, high-performance Python framework for navigation using **realtime** Model Predictive Control (MPC). 

MPC is an attractive control approach because it naturally handles constraints of different types and flexibly incorporates diverse control objectives. However, nonlinear MPC is often computationally expensive. For many systems, the solver time can exceed the control-loop period, especially on embedded hardware where computation is limited.

In contrast, Quadratic Programs (QPs) can be solved extremely quickly, and OSQP is particularly well suited for this. The core idea behind this framework is to convert a fully nonlinear MPC problem into a Linear Time-Varying (LTV) MPC problem that can be solved so fast that the linearization error remains small and does not degrade system performance.

The framework combines:

- **Symbolic definition**: Users define the nonlinear dynamics, constraints, and objective symbolically, exactly as they would on paper.
- **Automatic QP formulation** : The framework linearizes the problem and constructs the corresponding parametric QP automatically.
- **Cython compilation**: All functions that must be evaluated online for the parametric QP are compiled with Cython to achieve optimal runtime performance.  
- **Realtime-safe OSQP solving**: The QP is solved with OSQP at very high speed, using a configurable time limit to guarantee realtime feasibility.
- **Integrated simulator for rapid prototyping**: The same symbolic model used by the MPC is also used for simulation, with built-in plotting and animation tools to iterate quickly before deploying on embedded hardware.

Together, these components enable nonlinear MPC to run reliably and deterministically even on modest computing platforms, making it suitable for embedded robotic applications such as **ground vehicles (UGVs)**, **surface vessels (USVs)**, **aerial vehicles (UAVs)**, and more.

---

## ✨ Key Features

### 🔧 **1. Fully parametric, symbolic MPC pipeline**
- Symbolic linearization around operating trajectories  
- Automatic Jacobians and discrete-time dynamics  
- QP constructed explicitly (A, l, u, P, q) for transparency & speed

### ⚡ **2. C-accelerated QP evaluation via SymPy autowrap + Cython**
- Expensive symbolic expressions compiled to native machine code  
- Runtime QP evaluation **up to 5× faster** than simple Python
- Ideal for Jetson, Raspberry Pi, and embedded control CPUs  

### 🤖 **3. Clean modular architecture**
```
nav_mpc/
├── models/             # system dynamics (symbolic)
├── constraints/        # system + collision constraints (symbolic)
├── objectives/         # cost functions (symbolic)
├── mpc2qp/             # core: offline QP formulation + fast online updates
├── simulation/         # simulator, plotting, animations
├── utils/              # profiling, system info
└── wrappers/           # ROS2 interface (coming)
```

### 🔌 **4. Extensible to arbitrary systems**
- Simple pendulum (included)
- Double pendulum (included)
- Rover MPC (coming)  


---

## 🎯 Why nav_mpc?
**nav_mpc** provides:
- An easy way to define a full nonlinear MPC problem — dynamics, constraints, and objectives are written symbolically, just like on paper.
- A fast development workflow in Python, combined with Cython compilation for ultra-fast numerical evaluation.
- Realtime performance: the controller runs ultra-fast with deterministic timing, making it suitable for embedded hardware with tight control-loop deadlines.
- A clean, minimal set of dependencies and a research-friendly architecture that enables rapid prototyping, fast iteration, and straightforward extension to new robotic systems.

---

## 🚀 Getting Started

### 1️⃣ Installation

```bash
git clone https://github.com/ttsolakis/nav_mpc.git
cd nav_mpc
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2️⃣ Run an example

```bash
python main.py
```

This will:

- run nonlinear MPC on given model, objective, constraints  
- generate plots + animations  
- print realtime timing statistics  

### 3️⃣ Enable embedded deployment

Inside `main.py`:

```python
embedded = True
```

This sets a time limit for OSQP so that the total control loop remains time-feasible.

---

## 🧪 Examples

Examples run with:

OS:       Linux 6.14.0-37-generic  
Machine:  x86_64  
CPU:      Intel(R) Core(TM) i7-7500U CPU @ 2.70GHz  
Cores:    4 logical


### Simple Pendulum

Simple pendulum swing-up and stabilization with LTV-MPC:

<img src="examples/simple_pendulum/pendulum_animation.gif" width="400">
<img src="examples/simple_pendulum/state_trajectories.png" width="400">
<img src="examples/simple_pendulum/input_trajectories.png" width="400">

Performance with N = 40, dt = 0.02 s on a laptop CPU:

| Stage | Mean | Min | Max |
|-------|-------|-------|-------|
| QP eval | 1.12 ms | 1.04 ms | 4.22 ms |
| QP solve | 0.18 ms | 0.13 ms | 0.82 ms |
| Total MPC | **1.29 ms** | **1.18 ms** | **5.04 ms** |

Notice that Max time for Total MPC can stay deterministically below dt 
while getting optimal performance from OSQP.

---

### Double Pendulum

Double pendulum swing-up and stabilization with LTV-MPC:

<img src="examples/double_pendulum/double_pendulum_animation.gif" width="400">
<img src="examples/double_pendulum/state_trajectories.png" width="400">
<img src="examples/double_pendulum/input_trajectories.png" width="400">

Performance with N = 40, dt = 0.02 s on a laptop CPU:

| Stage | Mean | Min | Max |
|-------|-------|-------|-------|
| QP eval | 1.19 ms | 1.12 ms | 2.71 ms |
| QP solve | 0.26 ms | 0.23 ms | 1.47 ms |
| Total MPC | **1.45 ms** | **1.35 ms** | **3.88 ms** |

Notice that Max time for Total MPC can stay deterministically below dt 
while getting optimal performance from OSQP.

---

## 📄 License — MIT

Permissive, suitable for commercial + academic use.

---

## 📬 Contact

**Anastasios (Tasos) Tsolakis**  
📧 tas.tsolakis@gmail.com  
🌐 https://ttsolakis.github.io  
