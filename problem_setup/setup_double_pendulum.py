# nav_mpc/setup_double_pendulum.py

from models.double_pendulum_model import DoublePendulumModel
from objectives.double_pendulum_objective import DoublePendulumObjective
from constraints.system_constraints.double_pendulum_sys_constraints import DoublePendulumSystemConstraints
from simulation.animation.double_pendulum_animation import animate_double_pendulum


def setup_problem():
    name = "Double Pendulum Swing-Up and Stabilization"
    system = DoublePendulumModel()
    objective = DoublePendulumObjective(system)
    constraints = DoublePendulumSystemConstraints(system)
    animator = animate_double_pendulum
    return name, system, objective, constraints, animator
