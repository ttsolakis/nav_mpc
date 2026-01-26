# nav_mpc/setup_simple_rover.py

from core.models.simple_rover_model import SimpleRoverModel
from core.objectives.simple_rover_objective import SimpleRoverObjective
from core.constraints.system_constraints.simple_rover_sys_constraints import SimpleRoverSystemConstraints
from simulation.animation.rover_animation import animate_rover


def setup_problem():
    name = "Simple Rover Setpoint Tracking"
    system = SimpleRoverModel()
    objective = SimpleRoverObjective(system)
    constraints = SimpleRoverSystemConstraints(system)
    animator = animate_rover
    return name, system, objective, constraints, animator
