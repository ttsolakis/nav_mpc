# nav_mpc/simulation/path_generators/rrt_star.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from simulation.environment.occupancy_map import OccupancyMap2D


@dataclass
class RRTStarConfig:
    max_iters: int = 5000
    step_size: float = 0.25             # [m] extension length
    goal_sample_rate: float = 0.10      # probability of sampling the goal
    neighbor_radius: float = 0.75       # [m] for rewiring (can be adapted later)
    collision_check_step: float = 0.02  # [m] discretization along segment
    seed: int = 0


@dataclass
class _Node:
    x: float
    y: float
    parent: Optional[int]
    cost: float


def _inflate_occupancy(occ_map: OccupancyMap2D, inflation_radius_m: float) -> OccupancyMap2D:
    """
    Inflate occupied cells by a disk of radius inflation_radius_m (in meters).
    Returns a NEW OccupancyMap2D with the SAME geometry (origin/resolution/extents)
    as the original map.

    This function is robust to OccupancyMap2D implementations that store extra
    geometry metadata (e.g. `derived`).
    """
    if inflation_radius_m <= 0.0:
        return occ_map

    r_px = int(np.ceil(inflation_radius_m / occ_map.res))
    if r_px <= 0:
        return occ_map

    occ = occ_map.occ.copy()  # (H,W) bool

    # Disk kernel
    yy, xx = np.ogrid[-r_px : r_px + 1, -r_px : r_px + 1]
    disk = (xx * xx + yy * yy) <= (r_px * r_px)

    H, W = occ.shape
    inflated = occ.copy()

    ys, xs = np.where(occ)
    if ys.size == 0:
        return occ_map.__class__(inflated, occ_map.cfg, getattr(occ_map, "derived", None)) \
            if hasattr(occ_map, "derived") else occ_map.__class__(inflated, occ_map.cfg)

    for y, x in zip(ys, xs, strict=False):
        y0 = max(0, y - r_px)
        y1 = min(H, y + r_px + 1)
        x0 = max(0, x - r_px)
        x1 = min(W, x + r_px + 1)

        ky0 = y0 - (y - r_px)
        ky1 = ky0 + (y1 - y0)
        kx0 = x0 - (x - r_px)
        kx1 = kx0 + (x1 - x0)

        inflated[y0:y1, x0:x1] |= disk[ky0:ky1, kx0:kx1]

    # Construct a new map while preserving geometry metadata if present
    if hasattr(occ_map, "derived"):
        return occ_map.__class__(inflated, occ_map.cfg, occ_map.derived)
    return occ_map.__class__(inflated, occ_map.cfg)



def _sample_free(rng: np.random.Generator, occ_map: OccupancyMap2D) -> Tuple[float, float]:
    x = rng.uniform(occ_map.xmin, occ_map.xmax)
    y = rng.uniform(occ_map.ymin, occ_map.ymax)
    return float(x), float(y)


def _nearest(nodes: List[_Node], x: float, y: float) -> int:
    pts = np.array([(n.x, n.y) for n in nodes], dtype=float)
    d2 = (pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2
    return int(np.argmin(d2))


def _steer(x_from: float, y_from: float, x_to: float, y_to: float, step: float) -> Tuple[float, float]:
    dx = x_to - x_from
    dy = y_to - y_from
    dist = float(np.hypot(dx, dy))
    if dist < 1e-12:
        return x_from, y_from
    scale = min(1.0, step / dist)
    return (x_from + scale * dx, y_from + scale * dy)


def _segment_is_free(occ_map: OccupancyMap2D, x0: float, y0: float, x1: float, y1: float, ds: float) -> bool:
    dist = float(np.hypot(x1 - x0, y1 - y0))
    if dist < 1e-12:
        return occ_map.is_free_world(x0, y0)

    n = max(2, int(np.ceil(dist / ds)) + 1)
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)

    for x, y in zip(xs, ys, strict=False):
        if occ_map.is_occupied_world(float(x), float(y)):
            return False
    return True


def _neighbors(nodes: List[_Node], x: float, y: float, radius: float) -> List[int]:
    pts = np.array([(n.x, n.y) for n in nodes], dtype=float)
    d2 = (pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2
    return [int(i) for i in np.where(d2 <= radius * radius)[0]]


def _extract_path(nodes: List[_Node], goal_idx: int) -> np.ndarray:
    path = []
    i = goal_idx
    while i is not None:
        n = nodes[i]
        path.append([n.x, n.y])
        i = n.parent
    path.reverse()
    return np.asarray(path, dtype=float)


def rrt_star_plan(occ_map: OccupancyMap2D, start_xy: np.ndarray, goal_xy: np.ndarray, inflation_radius_m: float, cfg: Optional[RRTStarConfig] = None) -> np.ndarray:
    """
    Returns a polyline path as array shape (M,2) from start to goal.
    Raises ValueError if start/goal infeasible (in inflated map) or if no path found.
    """
    if cfg is None:
        cfg = RRTStarConfig()

    start_xy = np.asarray(start_xy, dtype=float).reshape(2)
    goal_xy = np.asarray(goal_xy, dtype=float).reshape(2)

    # Inflate obstacles by (robot_radius + margin)
    occ_inf = _inflate_occupancy(occ_map, inflation_radius_m)

    sx, sy = float(start_xy[0]), float(start_xy[1])
    gx, gy = float(goal_xy[0]), float(goal_xy[1])

    if occ_inf.is_occupied_world(sx, sy):
        raise ValueError("Start is in (inflated) obstacle / out of bounds.")
    if occ_inf.is_occupied_world(gx, gy):
        raise ValueError("Goal is in (inflated) obstacle / out of bounds.")

    rng = np.random.default_rng(cfg.seed)

    nodes: List[_Node] = [_Node(sx, sy, parent=None, cost=0.0)]
    best_goal_idx: Optional[int] = None
    best_goal_cost: float = np.inf

    for _ in range(cfg.max_iters):
        # 1) sample
        if rng.random() < cfg.goal_sample_rate:
            x_rand, y_rand = gx, gy
        else:
            x_rand, y_rand = _sample_free(rng, occ_inf)

        # 2) nearest + steer
        i_near = _nearest(nodes, x_rand, y_rand)
        x_new, y_new = _steer(nodes[i_near].x, nodes[i_near].y, x_rand, y_rand, cfg.step_size)

        # 3) collision check
        if not _segment_is_free(occ_inf, nodes[i_near].x, nodes[i_near].y, x_new, y_new, cfg.collision_check_step):
            continue

        # 4) choose parent among neighbors (RRT*)
        neigh = _neighbors(nodes, x_new, y_new, cfg.neighbor_radius)
        best_parent = i_near
        best_cost = nodes[i_near].cost + float(np.hypot(x_new - nodes[i_near].x, y_new - nodes[i_near].y))

        for j in neigh:
            cand = nodes[j]
            c = cand.cost + float(np.hypot(x_new - cand.x, y_new - cand.y))
            if c < best_cost:
                if _segment_is_free(occ_inf, cand.x, cand.y, x_new, y_new, cfg.collision_check_step):
                    best_cost = c
                    best_parent = j

        new_idx = len(nodes)
        nodes.append(_Node(x_new, y_new, parent=best_parent, cost=best_cost))

        # 5) rewire neighbors through new node
        for j in neigh:
            if j == best_parent:
                continue
            cand = nodes[j]
            c_through = best_cost + float(np.hypot(cand.x - x_new, cand.y - y_new))
            if c_through + 1e-12 < cand.cost:
                if _segment_is_free(occ_inf, x_new, y_new, cand.x, cand.y, cfg.collision_check_step):
                    cand.parent = new_idx
                    cand.cost = c_through

        # 6) check goal connection (try to connect directly)
        dist_to_goal = float(np.hypot(gx - x_new, gy - y_new))
        if dist_to_goal <= cfg.step_size:
            if _segment_is_free(occ_inf, x_new, y_new, gx, gy, cfg.collision_check_step):
                goal_cost = best_cost + dist_to_goal
                if goal_cost < best_goal_cost:
                    best_goal_cost = goal_cost
                    # add goal node as explicit node for easy extraction
                    best_goal_idx = len(nodes)
                    nodes.append(_Node(gx, gy, parent=new_idx, cost=goal_cost))

    if best_goal_idx is None:
        raise ValueError("RRT* failed to find a path within max_iters.")

    return _extract_path(nodes, best_goal_idx)
