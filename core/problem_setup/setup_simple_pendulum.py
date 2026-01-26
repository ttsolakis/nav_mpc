# nav_mpc/setup_simple_pendulum.py

from core.models.simple_pendulum_model import SimplePendulumModel
from core.objectives.simple_pendulum_objective import SimplePendulumObjective
from core.constraints.system_constraints.simple_pendulum_sys_constraints import SimplePendulumSystemConstraints
from simulation.animation.simple_pendulum_animation import animate_pendulum


def setup_problem():
    name = "Simple Pendulum Swing-Up and Stabilization"
    system = SimplePendulumModel()
    objective = SimplePendulumObjective(system)
    constraints = SimplePendulumSystemConstraints(system)
    animator = animate_pendulum
    return name, system, objective, constraints, animator
