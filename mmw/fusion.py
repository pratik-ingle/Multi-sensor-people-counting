"""Bring every sensor's point cloud into one store coordinate frame and drop impossible points."""
from __future__ import annotations

import numpy as np

from .geometry import Pose, sph_to_cart
from .layout import Layout


def sensor_poses(meta: dict, which: str = "calibrated") -> list[Pose]:
    return [Pose.from_dict(d) for d in meta[f"sensors_{which}"]]


def to_world(points: np.ndarray, poses: list[Pose]) -> np.ndarray:
    """points: (K, 7) run table -> (K, 3) world xyz using each row's sensor pose."""
    xyz = np.empty((len(points), 3))
    sid = points[:, 1].astype(int)
    ps = sph_to_cart(points[:, 2].astype(float), points[:, 3].astype(float), points[:, 4].astype(float))
    for k, pose in enumerate(poses):
        m = sid == k
        if m.any():
            xyz[m] = pose.to_world(ps[m])
    return xyz


def plausible(xyz: np.ndarray, layout: Layout, zmin=0.1, zmax=2.3, obstacle_margin=0.05) -> np.ndarray:
    """Keep points at human height, inside the room and not inside furniture.
    The furniture test removes many multipath ghosts, which appear 'behind' reflective shelves."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    ok = (z > zmin) & (z < zmax) & (x > 0) & (x < layout.width) & (y > 0) & (y < layout.depth)
    for o in layout.obstacles:
        if not o.seating:
            ok &= ~o.contains(x, y, obstacle_margin)
    return ok
