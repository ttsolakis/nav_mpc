# nav_mpc

Platform-agnostic navigation MPC framework built on top of [TinyMPC](https://tinympc.org).

## Goals

- Use nonlinear models (e.g. AGV, ASV, UAV) and automatically build a linear MPC approximation.
- Solve fast QP problems with TinyMPC for mid/low-level navigation.
- Keep the design modular:
  - `models/` – system dynamics
  - `objectives/` – stage costs
  - `constraints/` – system and collision constraints
  - `qp_formulation/` – linearization & QP setup
  - `planner/` – TinyMPC wrapper
  - `simulation/` – simulator, plotting, animation
  - `wrappers/` – ROS2 and pure-Python entry points

## Quick start (dev)

```bash
cd ~/dev_ws/src/nav_mpc
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install tinympc numpy

python main.py  # runs a simple MPC test

