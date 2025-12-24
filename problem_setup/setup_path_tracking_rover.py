# nav_mpc/setup_path_tracking_rover.py

from models.simple_rover_model import SimpleRoverModel
from objectives.rover_path_tracking_objective import RoverPathTrackingObjective
from constraints.system_constraints.simple_rover_sys_constraints import SimpleRoverSystemConstraints
from simulation.animation.rover_animation import animate_rover


def setup_problem():
    name = "Rover Path Tracking"
    system = SimpleRoverModel()
    objective = RoverPathTrackingObjective(system)
    constraints = SimpleRoverSystemConstraints(system)
    animator = animate_rover
    return name, system, objective, constraints, animator
