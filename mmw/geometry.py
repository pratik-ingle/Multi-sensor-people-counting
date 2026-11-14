"""Sensor poses and coordinate conversions.

Sensor frame follows the TI convention: x = right, y = boresight, z = up (of the board).
Spherical: azimuth = atan2(x, y), elevation = asin(z / r).
"""
from __future__ import annotations

import numpy as np


def rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Sensor-to-world rotation. yaw turns about world z; pitch tilts the boresight
    (negative = down, -90 = ceiling mount looking at the floor)."""
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    rz = np.array([[np.cos(y), -np.sin(y), 0.0], [np.sin(y), np.cos(y), 0.0], [0.0, 0.0, 1.0]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(p), -np.sin(p)], [0.0, np.sin(p), np.cos(p)]])
    return rz @ rx


def cart_to_sph(p: np.ndarray):
    r = np.linalg.norm(p, axis=1)
    az = np.arctan2(p[:, 0], p[:, 1])
    el = np.arcsin(np.clip(p[:, 2] / np.maximum(r, 1e-9), -1.0, 1.0))
    return r, az, el


def sph_to_cart(r, az, el) -> np.ndarray:
    c = np.cos(el)
    return np.stack([r * c * np.sin(az), r * c * np.cos(az), r * np.sin(el)], axis=1)


class Pose:
    def __init__(self, position, yaw_deg: float, pitch_deg: float):
        self.position = np.asarray(position, dtype=float)
        self.yaw_deg = float(yaw_deg)
        self.pitch_deg = float(pitch_deg)
        self.R = rotation(self.yaw_deg, self.pitch_deg)

    def to_sensor(self, pw: np.ndarray) -> np.ndarray:
        return (pw - self.position) @ self.R

    def to_world(self, ps: np.ndarray) -> np.ndarray:
        return ps @ self.R.T + self.position

    def as_dict(self) -> dict:
        return {"position": self.position.tolist(), "yaw_deg": self.yaw_deg, "pitch_deg": self.pitch_deg}

    @classmethod
    def from_dict(cls, d: dict) -> "Pose":
        return cls(d["position"], d["yaw_deg"], d["pitch_deg"])
