# nav_mpc/simulation/environment/occupancy_map.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass(frozen=True, slots=True)
class OccupancyMapConfig:
    """
    User-facing config (Option B: "world-first").

    Provide ONE of:
      - world_width_m: desired map width in meters
      - world_diag_m:  desired map diagonal in meters

    The map will be centered at the WORLD origin (0,0).
    That means: origin (bottom-left pixel in world coords) is derived automatically.

    Notes:
    - Pixels are assumed square.
    - If you specify world_width_m, height is derived from aspect ratio.
    - If you specify world_diag_m, both width/height are derived from aspect ratio.
    """
    map_path: str | Path

    # World size specification (pick exactly ONE)
    world_width_m: float | None = None
    world_diag_m: float | None = None

    # Thresholding
    occupied_threshold: int = 127     # grayscale threshold: <= is occupied
    invert: bool = False              # if your image uses white=occupied, black=free


@dataclass(frozen=True, slots=True)
class OccupancyMapDerived:
    """
    Derived geometry / placement (computed from image + OccupancyMapConfig).
    """
    resolution: float                 # [m/pixel]
    origin: Tuple[float, float]       # (x0, y0) world coords of bottom-left pixel
    world_width_m: float              # [m]
    world_height_m: float             # [m]


class OccupancyMap2D:
    """
    2D occupancy grid loaded from an image.

    Conventions:
    - World frame: x right, y up.
    - World origin (0,0) is at the CENTER of the image.
    - origin = world coordinate of the bottom-left pixel (px=0, py=0).
    - Image storage: numpy array rows start at top, so we flip y when indexing.
    - self.occ is boolean array (H, W): True=occupied.
    """

    def __init__(self, occ: np.ndarray, cfg: OccupancyMapConfig, derived: OccupancyMapDerived):
        if occ.dtype != np.bool_:
            raise TypeError("occ must be a boolean array where True=occupied.")
        if derived.resolution <= 0:
            raise ValueError("resolution must be > 0")

        self.occ = occ
        self.cfg = cfg
        self.derived = derived

        self.height, self.width = occ.shape

        self.res = float(derived.resolution)
        self.x0, self.y0 = float(derived.origin[0]), float(derived.origin[1])

        # Map extents in world coordinates
        self.xmin = self.x0
        self.ymin = self.y0
        self.xmax = self.x0 + self.width * self.res
        self.ymax = self.y0 + self.height * self.res

    # --------------------------
    # Path resolution helpers
    # --------------------------

    @staticmethod
    def _project_root() -> Path:
        # occupancy_map.py is: nav_mpc/simulation/environment/occupancy_map.py
        # parents[2] = nav_mpc
        return Path(__file__).resolve().parents[2]

    @staticmethod
    def _resolve_map_path(map_path: str | Path) -> Path:
        p = Path(map_path)

        # 1) absolute path
        if p.is_absolute():
            return p

        pr = OccupancyMap2D._project_root()
        looked: list[Path] = []

        # 2) relative to project root (supports "simulation/environment/maps/map.png")
        cand = pr / p
        looked.append(cand)
        if cand.exists():
            return cand

        # 3) common map locations
        candidates = [
            pr / "maps" / p,                                   # nav_mpc/maps/...
            pr / "simulation" / "environment" / "maps" / p,     # nav_mpc/simulation/environment/maps/...
        ]
        looked.extend(candidates)
        for c in candidates:
            if c.exists():
                return c

        # 4) relative to cwd
        cand = Path.cwd() / p
        looked.append(cand)
        if cand.exists():
            return cand

        # fallback for error message
        return looked[0]

    # --------------------------
    # Constructors / I/O
    # --------------------------

    @staticmethod
    def from_png(cfg: OccupancyMapConfig) -> "OccupancyMap2D":
        """
        Load a PNG (or any format supported by matplotlib.image.imread),
        convert to grayscale, threshold to occupied/free, and derive geometry.

        World placement:
          - map is CENTERED at world origin (0,0).
        """
        import matplotlib.image as mpimg

        map_path = OccupancyMap2D._resolve_map_path(cfg.map_path)
        if not map_path.exists():
            pr = OccupancyMap2D._project_root()
            raise FileNotFoundError(
                f"Map image not found: {map_path}\n"
                f"Looked for:\n"
                f"  - absolute path\n"
                f"  - <project_root> / <map_path>\n"
                f"  - <project_root>/maps/<map_path>\n"
                f"  - <project_root>/simulation/environment/maps/<map_path>\n"
                f"Project root: {pr}\n"
                f"Tip: put your file under nav_mpc/simulation/environment/maps/ and pass map_path='map.png'."
            )

        img = mpimg.imread(str(map_path))

        # img can be:
        # - (H, W) grayscale float
        # - (H, W, 3) RGB float
        # - (H, W, 4) RGBA float
        if img.ndim == 3:
            rgb = img[..., :3]
            gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        else:
            gray = img.astype(float)

        # Normalize to 0..255 for thresholding
        if gray.max() <= 1.0 + 1e-9:
            gray_u8 = (255.0 * np.clip(gray, 0.0, 1.0)).astype(np.uint8)
        else:
            gray_u8 = np.clip(gray, 0.0, 255.0).astype(np.uint8)

        occ = gray_u8 <= int(cfg.occupied_threshold)
        if cfg.invert:
            occ = ~occ

        Hpx, Wpx = occ.shape  # rows, cols

        # ---- derive resolution from world_width_m OR world_diag_m ----
        has_w = cfg.world_width_m is not None
        has_d = cfg.world_diag_m is not None
        if has_w == has_d:
            raise ValueError("OccupancyMapConfig: provide exactly ONE of world_width_m or world_diag_m.")

        if has_w:
            world_w = float(cfg.world_width_m)
            if world_w <= 0:
                raise ValueError("world_width_m must be > 0")
            res = world_w / float(Wpx)  # [m/pixel]
        else:
            world_d = float(cfg.world_diag_m)
            if world_d <= 0:
                raise ValueError("world_diag_m must be > 0")
            diag_px = float(np.sqrt(float(Wpx * Wpx + Hpx * Hpx)))
            res = world_d / diag_px  # [m/pixel]

        world_w = float(Wpx) * res
        world_h = float(Hpx) * res

        # Center map at world origin:
        # bottom-left origin is (-world_w/2, -world_h/2)
        origin = (-0.5 * world_w, -0.5 * world_h)

        derived = OccupancyMapDerived(
            resolution=res,
            origin=(float(origin[0]), float(origin[1])),
            world_width_m=world_w,
            world_height_m=world_h,
        )

        return OccupancyMap2D(occ=occ, cfg=cfg, derived=derived)

    # --------------------------
    # World <-> Pixel transforms
    # --------------------------

    def world_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert world (x,y) to integer pixel indices (px, py) where:
          px in [0, W-1], py in [0, H-1]
        Here py=0 is the bottom row in *world* coordinates.
        """
        px = int(np.floor((x - self.x0) / self.res))
        py = int(np.floor((y - self.y0) / self.res))
        return px, py

    def pixel_to_rowcol(self, px: int, py: int) -> Tuple[int, int]:
        """
        Convert (px, py) where py=0 is bottom in world-space pixel coords
        to numpy indexing (row, col) where row=0 is top.
        """
        col = px
        row = (self.height - 1) - py
        return row, col

    def in_bounds_world(self, x: float, y: float) -> bool:
        return (self.xmin <= x < self.xmax) and (self.ymin <= y < self.ymax)

    def in_bounds_pixel(self, px: int, py: int) -> bool:
        return (0 <= px < self.width) and (0 <= py < self.height)

    # --------------------------
    # Occupancy queries
    # --------------------------

    def is_occupied_world(self, x: float, y: float) -> bool:
        """
        Returns True if (x,y) lies in an occupied pixel.
        Out-of-bounds is treated as occupied (useful as a boundary wall).
        """
        px, py = self.world_to_pixel(x, y)
        if not self.in_bounds_pixel(px, py):
            return True
        row, col = self.pixel_to_rowcol(px, py)
        return bool(self.occ[row, col])

    def is_free_world(self, x: float, y: float) -> bool:
        return not self.is_occupied_world(x, y)

    # --------------------------
    # Debug plotting (world coords)
    # --------------------------

    def plot(
        self,
        *,
        start_xy: np.ndarray | Tuple[float, float] | None = None,
        goal_xy: np.ndarray | Tuple[float, float] | None = None,
        grid_step: float = 1.0,
        show_pixel_grid: bool = False,
        pixel_grid_step: int = 10,
        annotate_xy: bool = True,
        title: str = "Occupancy map (world coordinates)",
        figsize: Tuple[float, float] = (7.0, 7.0),
        show: bool = True,
    ):
        """
        Plot occupancy map in WORLD coordinates with optional 1m grid and reminder of extents.
        Very useful to debug: start/goal out-of-bounds or inside (inflated) obstacles.

        Args:
            start_xy, goal_xy: optional markers plotted in world coordinates.
            grid_step: world grid spacing [m] for labeled ticks.
            show_pixel_grid: overlay a faint pixel grid (can be heavy for large images).
            pixel_grid_step: spacing in pixels for the pixel grid overlay.
            annotate_xy: write a small text box with world extents and start/goal occupancy.
        """
        import matplotlib.pyplot as plt

        occ = np.asarray(self.occ, dtype=float)
        occ_plot = np.flipud(occ)  # so y increases upward in world

        extent = [self.xmin, self.xmax, self.ymin, self.ymax]

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(
            occ_plot,
            extent=extent,
            origin="lower",
            cmap="gray_r",
            interpolation="nearest",
        )

        ax.set_aspect("equal", "box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(title)

        # World grid/ticks
        if grid_step and grid_step > 0:
            xs = np.arange(np.floor(self.xmin), np.ceil(self.xmax) + 1e-9, grid_step)
            ys = np.arange(np.floor(self.ymin), np.ceil(self.ymax) + 1e-9, grid_step)
            ax.set_xticks(xs)
            ax.set_yticks(ys)
            ax.grid(True, which="both", color="k", alpha=0.25, linewidth=0.8)

        # Optional pixel-grid overlay (lightweight version)
        if show_pixel_grid and pixel_grid_step > 0:
            # draw pixel grid lines in world coords
            # vertical lines (columns)
            for px in range(0, self.width + 1, int(pixel_grid_step)):
                xw = self.x0 + px * self.res
                ax.axvline(xw, color="k", alpha=0.08, linewidth=0.6)
            # horizontal lines (rows)
            for py in range(0, self.height + 1, int(pixel_grid_step)):
                yw = self.y0 + py * self.res
                ax.axhline(yw, color="k", alpha=0.08, linewidth=0.6)

        # Markers
        if start_xy is not None:
            sx, sy = float(start_xy[0]), float(start_xy[1])
            ax.plot(sx, sy, "go", markersize=10, label="start")
        if goal_xy is not None:
            gx, gy = float(goal_xy[0]), float(goal_xy[1])
            ax.plot(gx, gy, "r*", markersize=12, label="goal")

        if start_xy is not None or goal_xy is not None:
            ax.legend(loc="upper right")

        # # Annotation box with extents + occupancy checks
        # if annotate_xy:
        #     lines = [
        #         f"extent x: [{self.xmin:.2f}, {self.xmax:.2f}] m",
        #         f"extent y: [{self.ymin:.2f}, {self.ymax:.2f}] m",
        #         f"res: {self.res:.4f} m/px  (W={self.width}, H={self.height})",
        #     ]
        #     if start_xy is not None:
        #         sx, sy = float(start_xy[0]), float(start_xy[1])
        #         lines.append(f"start: ({sx:.2f},{sy:.2f})  in_bounds={self.in_bounds_world(sx, sy)}  occ={self.is_occupied_world(sx, sy)}")
        #     if goal_xy is not None:
        #         gx, gy = float(goal_xy[0]), float(goal_xy[1])
        #         lines.append(f"goal:  ({gx:.2f},{gy:.2f})  in_bounds={self.in_bounds_world(gx, gy)}  occ={self.is_occupied_world(gx, gy)}")

        #     ax.text(
        #         0.02,
        #         0.02,
        #         "\n".join(lines),
        #         transform=ax.transAxes,
        #         fontsize=9,
        #         va="bottom",
        #         ha="left",
        #         bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="none"),
        #     )

        if show:
            plt.show()

        return fig, ax
