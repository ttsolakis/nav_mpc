# nav_mpc/problem_setup/setup_path_tracking_unicycle.py

from core.models.unicycle_kinematic_model import UnicycleKinematicModel
from core.objectives.unicycle_path_tracking_objective import UnicyclePathTrackingObjective
from core.constraints.system_constraints.unicycle_kinematic_sys_constraints import UnicycleKinematicSystemConstraints
from core.constraints.collision_constraints.halfspace_corridor import HalfspaceCorridorCollisionConfig
from simulation.animation.unicycle_animation import animate_unicycle


def setup_problem():
    name = "Unicycle Path Tracking"
    system = UnicycleKinematicModel()
    objective = UnicyclePathTrackingObjective(system)
    constraints = UnicycleKinematicSystemConstraints(system)
    collision = HalfspaceCorridorCollisionConfig()
    animator = animate_unicycle
    return name, system, objective, constraints, collision, animator
