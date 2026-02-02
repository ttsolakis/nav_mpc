# nav_mpc/problem_setup/setup_path_tracking_cybership.py

from core.models.cybership_model import CybershipModel
from core.objectives.cybership_path_tracking_objective import CybershipPathTrackingObjective
from core.constraints.system_constraints.cybership_sys_constraints import CybershipSystemConstraints
from core.constraints.collision_constraints.halfspace_corridor import HalfspaceCorridorCollisionConfig
# from simulation.animation.cybership_animation import animate_cybership


def setup_problem():
    name = "Cybership Path Tracking"
    system = CybershipModel()
    objective = CybershipPathTrackingObjective(system)
    constraints = CybershipSystemConstraints(system)
    collision = HalfspaceCorridorCollisionConfig()
    # animator = animate_cybership

    return name, system, objective, constraints, collision
