"""Thermal state of the showroom: room air, furniture, people, and warm seat imprints."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from .meshes import HumanTemplate, MeshBuilder, box, build_furniture, grid_plane
from .radiometry import apparent_radiance

PEOPLE_EMISSIVITY = 0.97


@dataclass
class Environment:
    ambient_c: float = 22.0          # room air
    hvac_swing_c: float = 0.4        # slow air-conditioning cycle
    hvac_period_s: float = 1200.0
    reflected_offset_c: float = 1.0  # surroundings seen in reflections (ceiling, lights) are slightly warmer
    sun_box: tuple | None = None     # sunlit floor patch behind the storefront glass
    sun_delta_c: float = 0.0
    imprint_peak_c: float = 6.0      # seat warming after a long sit
    imprint_build_s: float = 60.0
    imprint_tau_s: float = 120.0     # cooling time constant of upholstery

    @classmethod
    def from_dict(cls, d: dict | None) -> "Environment":
        d = dict(d or {})
        sun = d.pop("sun_patch", None)
        env = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if sun:
            env.sun_box = tuple(sun["box"])
            env.sun_delta_c = float(sun.get("delta_c", 3.0))
        return env

    def air(self, t: float) -> float:
        return self.ambient_c + self.hvac_swing_c * np.sin(2 * np.pi * t / self.hvac_period_s)


class ThermalScene:
    def __init__(self, layout_cfg: dict, env: Environment, seats: list[dict], rng: np.random.Generator):
        self.env = env
        self.seats = seats
        W, D = float(layout_cfg["width"]), float(layout_cfg["depth"])
        mb = MeshBuilder()

        fv, ff = grid_plane(0, 0, W, D, 0.0, 0.25)  # floor: slight non-uniformity + sun patch
        nx, ny = int(round(W / 0.25)) + 1, int(round(D / 0.25)) + 1
        mottling = gaussian_filter(rng.normal(0, 1, (ny, nx)), 6)
        mottling = 0.25 * mottling / (mottling.std() + 1e-9)
        dT = 0.3 + mottling.ravel()
        if env.sun_box is not None:
            x0, y0, x1, y1 = env.sun_box
            edge = 0.6
            sx = np.clip(np.minimum(fv[:, 0] - x0, x1 - fv[:, 0]) / edge, 0, 1)
            sy = np.clip(np.minimum(fv[:, 1] - y0, y1 - fv[:, 1]) / edge, 0, 1)
            dT = dT + env.sun_delta_c * sx * sy
        mb.add((fv, ff), dT=dT, eps=0.95)
        mb.add(box(-0.1, -0.1, 0, W + 0.1, 0.0, 3.0), dT=-1.0, eps=0.9)   # storefront glass (opaque in LWIR)
        mb.add(box(-0.1, D, 0, W + 0.1, D + 0.1, 3.0), dT=-0.2, eps=0.92)
        mb.add(box(-0.1, 0, 0, 0.0, D, 3.0), dT=-0.2, eps=0.92)
        mb.add(box(W, 0, 0, W + 0.1, D, 3.0), dT=-0.2, eps=0.92)
        for ob in layout_cfg.get("obstacles", []):
            build_furniture(mb, ob, rng)
        self.V, self.F, self.dT, self.t_fixed, self.eps = mb.arrays()
        self.human = HumanTemplate()
        self.imprints: list[list] = []  # [seat_id, t_end, amplitude]

    # ---------------------------------------------------------------- static scene
    def static_radiance(self, t: float) -> np.ndarray:
        air = self.env.air(t)
        T = np.where(np.isnan(self.t_fixed), air + self.dT, self.t_fixed)
        return apparent_radiance(T, self.eps, air + self.env.reflected_offset_c).astype(np.float32)

    # ---------------------------------------------------------------- people
    def people(self, snap: dict, t: float):
        """Posed body meshes -> (vertices (N*V, 3), radiance (N*V,), person ids (N*V,))."""
        N = len(snap["pos"])
        if N == 0:
            return np.zeros((0, 3), np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32)
        verts = self.human.pose(snap).reshape(-1, 3)
        air = self.env.air(t)
        f = snap["tfac"][:, self.human.mat]                               # (N, V)
        T = air + f * (snap["tskin"][:, None] - air) + 0.25 * self.human.noise[None, :]
        rad = apparent_radiance(T, PEOPLE_EMISSIVITY, air + self.env.reflected_offset_c).ravel()
        ids = np.repeat(np.arange(1, N + 1, dtype=np.float32), self.human.n_verts)
        return verts.astype(np.float32), rad.astype(np.float32), ids

    # ---------------------------------------------------------------- seat imprints
    def on_events(self, events: list[tuple]):
        for kind, seat, t_end, seconds in events:
            if kind == "stood_up":
                amp = self.env.imprint_peak_c * (1 - np.exp(-seconds / self.env.imprint_build_s))
                self.imprints.append([seat, t_end, amp])

    def imprint_quads(self, t: float):
        """Warm patches left on cushions: horizontal seat patch (+ backrest patch on sofas)."""
        env = self.env
        alive, quads, temps = [], [], []
        air = env.air(t)
        for seat, t_end, amp in self.imprints:
            dT = amp * np.exp(-(t - t_end) / env.imprint_tau_s)
            if dT < 0.15:
                continue
            alive.append([seat, t_end, amp])
            s = self.seats[seat]
            fx, fy = np.cos(s["facing"]), np.sin(s["facing"])
            lx, ly = -fy, fx
            cx, cy = s["xb"] + 0.12 * fx, s["yb"] + 0.12 * fy
            z = s["h"] + 0.006
            corners = [(cx + a * 0.21 * lx + b * 0.22 * fx, cy + a * 0.21 * ly + b * 0.22 * fy, z)
                       for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            quads.append(corners)
            temps.append(air + dT)
            if s.get("back"):
                bx, by = s["xb"] - 0.14 * fx, s["yb"] - 0.14 * fy
                quads.append([(bx - 0.2 * lx, by - 0.2 * ly, s["h"] + 0.1), (bx + 0.2 * lx, by + 0.2 * ly, s["h"] + 0.1),
                              (bx + 0.2 * lx, by + 0.2 * ly, s["h"] + 0.42), (bx - 0.2 * lx, by - 0.2 * ly, s["h"] + 0.42)])
                temps.append(air + 0.6 * dT)
        self.imprints = alive
        if not quads:
            return np.zeros((0, 3), np.float32), np.zeros(0, np.float32)
        V = np.array(quads, dtype=np.float32).reshape(-1, 3)
        rad = apparent_radiance(np.repeat(temps, 4), 0.95, air + env.reflected_offset_c).astype(np.float32)
        return V, rad
