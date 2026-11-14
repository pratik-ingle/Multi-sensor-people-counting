"""TI IWR6843AOP point-cloud model.

This does not simulate chirps/FFTs. It reproduces what the on-chip 3D People Tracking
processing chain emits per frame -- a sparse point cloud of (range, azimuth, elevation,
Doppler, SNR) -- including the effects that matter for counting:

* field of view and range limits, antenna-gain roll-off towards the FoV edge
* radar-equation SNR falloff (1/R^4) with Swerling-style fluctuation and CFAR threshold
* static clutter removal: still people lose most of their points
* self-occlusion, person-person and furniture occlusion
* limited angular resolution: nearby scatterers (and nearby people) merge into one point
* multipath ghosts mirrored behind metal/glass furniture, random false alarms
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from .geometry import Pose, cart_to_sph
from .layout import Box


@dataclass
class RadarParams:
    fov_az_deg: float = 60.0
    fov_el_deg: float = 60.0
    min_range: float = 0.4
    max_range: float = 8.0
    range_res: float = 0.07
    vel_res: float = 0.08
    max_vel: float = 4.0
    ang_res_deg: float = 29.0
    merge_ang_deg: float = 15.0
    snr_ref_db: float = 24.0
    ref_range: float = 3.0
    cfar_db: float = 12.0
    beam_edge_loss_db: float = 6.0
    static_mode: str = "scr"
    minor_motion_pd: float = 0.15
    max_points: int = 800
    false_alarm_rate: float = 2.0
    multipath_prob: float = 0.10
    multipath_loss_db: float = 6.0

    @classmethod
    def from_dict(cls, d: dict) -> "RadarParams":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def _faces(b: Box):
    """Vertical faces of a box: (axis, coordinate, outward sign, (lo, hi) extent along the other axis)."""
    return [(0, b.x0, -1.0, (b.y0, b.y1)), (0, b.x1, 1.0, (b.y0, b.y1)),
            (1, b.y0, -1.0, (b.x0, b.x1)), (1, b.y1, 1.0, (b.x0, b.x1))]


class Radar:
    def __init__(self, sid: int, pose: Pose, params: RadarParams):
        self.sid = sid
        self.pose = pose
        self.p = params

    # ---------------------------------------------------------------- occlusion
    def _person_occlusion(self, P, own, people) -> np.ndarray:
        C, H, Rr = people["pos"], people["height"], people["radius"]
        s = self.pose.position
        near = np.hypot(C[:, 0] - s[0], C[:, 1] - s[1]) < self.p.max_range + 1.0
        idx = np.nonzero(near)[0]
        if len(idx) < 2 or len(P) == 0:
            return np.zeros(len(P), dtype=bool)
        C, H, Rr = C[idx], H[idx], Rr[idx]
        D2 = P[:, :2] - s[:2]
        L2 = (D2 ** 2).sum(1)
        t = ((C[None, :, :] - s[:2]) * D2[:, None, :]).sum(-1) / np.maximum(L2, 1e-9)[:, None]
        closest = s[:2] + t[..., None] * D2[:, None, :]
        dist = np.hypot(C[None, :, 0] - closest[..., 0], C[None, :, 1] - closest[..., 1])
        z_at = s[2] + t * (P[:, None, 2] - s[2])
        L = np.sqrt(L2)[:, None]
        in_front = (t > 0) & (t < 1 - (Rr[None, :] + 0.05) / np.maximum(L, 1e-6))
        hit = (dist < Rr[None, :]) & (z_at < H[None, :]) & in_front & (L2[:, None] > 0.04)
        hit &= idx[None, :] != own[:, None]  # a body does not occlude itself here (handled by normals)
        return hit.any(1)

    def _obstacle_occlusion(self, P, obstacles) -> np.ndarray:
        s = self.pose.position
        D = P - s
        occ = np.zeros(len(P), dtype=bool)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / D
            for b in obstacles:
                lo = np.array([b.x0, b.y0, 0.0])
                hi = np.array([b.x1, b.y1, b.height])
                t1 = (lo - s) * inv
                t2 = (hi - s) * inv
                tmin = np.nanmax(np.minimum(t1, t2), axis=1)
                tmax = np.nanmin(np.maximum(t1, t2), axis=1)
                occ |= (tmax >= tmin) & (tmin > 0) & (tmin < 1 - 1e-3)
        return occ

    # ---------------------------------------------------------------- multipath
    def _ghosts(self, P, V, rcs, obstacles, rng):
        s = self.pose.position
        gP, gV, gR = [], [], []
        for b in obstacles:
            if not b.reflective:
                continue
            for axis, coord, sign, (lo, hi) in _faces(b):
                if (s[axis] - coord) * sign <= 0:
                    continue  # sensor is behind this face
                pm = P.copy()
                pm[:, axis] = 2 * coord - P[:, axis]
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = (coord - s[axis]) / (pm[:, axis] - s[axis])
                q = s + t[:, None] * (pm - s)
                other = 1 - axis
                ok = ((P[:, axis] - coord) * sign > 0) & (q[:, other] >= lo) & (q[:, other] <= hi) \
                    & (q[:, 2] >= 0) & (q[:, 2] <= b.height) & (rng.random(len(P)) < self.p.multipath_prob)
                if ok.any():
                    vm = V[ok].copy()
                    vm[:, axis] *= -1
                    gP.append(pm[ok])
                    gV.append(vm)
                    gR.append(rcs[ok] - self.p.multipath_loss_db)
        if not gP:
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
        return np.concatenate(gP), np.concatenate(gV), np.concatenate(gR)

    # ---------------------------------------------------------------- main
    def measure(self, sc: dict, people: dict, obstacles: list[Box], rng: np.random.Generator) -> np.ndarray:
        """Return this sensor's point cloud for one frame: (K, 5) = range, az, el, doppler, snr_db."""
        p = self.p
        s = self.pose.position
        fov_az, fov_el = np.radians(p.fov_az_deg), np.radians(p.fov_el_deg)

        P, V, Nrm, rcs, own = sc["P"], sc["V"], sc["N"], sc["rcs"], sc["owner"]
        d = P - s
        r = np.linalg.norm(d, axis=1)
        keep = (r > p.min_range) & (r < p.max_range)
        if keep.any():
            _, az, el = cart_to_sph(self.pose.to_sensor(P[keep]))
            sub = np.nonzero(keep)[0]
            keep[sub] = (np.abs(az) <= fov_az) & (np.abs(el) <= fov_el)
        keep &= (Nrm * -d).sum(1) > 0  # surface faces the sensor (top normals face a ceiling unit)
        P, V, rcs, own = P[keep], V[keep], rcs[keep], own[keep]
        if len(P):
            vis = ~(self._person_occlusion(P, own, people) | self._obstacle_occlusion(P, obstacles))
            P, V, rcs = P[vis], V[vis], rcs[vis]
        gP, gV, gR = self._ghosts(P, V, rcs, obstacles, rng)
        P = np.concatenate([P, gP])
        V = np.concatenate([V, gV])
        rcs = np.concatenate([rcs, gR])

        out = np.zeros((0, 5))
        if len(P):
            ps = self.pose.to_sensor(P)
            r, az, el = cart_to_sph(ps)
            ok = (r > p.min_range) & (r < p.max_range) & (np.abs(az) <= fov_az) & (np.abs(el) <= fov_el)
            r, az, el, V, rcs, P = r[ok], az[ok], el[ok], V[ok], rcs[ok], P[ok]
            vw = (V * (P - s)).sum(1) / np.maximum(r, 1e-9)                        # radial velocity (+ = away)
            vr = ((vw + p.max_vel) % (2 * p.max_vel)) - p.max_vel                  # Doppler aliasing
            beam = -p.beam_edge_loss_db * ((az / fov_az) ** 2 + (el / fov_el) ** 2)
            snr = p.snr_ref_db + rcs - 40 * np.log10(r / p.ref_range) + beam \
                + 10 * np.log10(rng.exponential(1.0, len(r)) + 1e-12)
            still = np.abs(vr) < 0.75 * p.vel_res
            if p.static_mode == "scr":
                det = ~still
            else:  # minor-motion processing: still scatterers survive occasionally, with a penalty
                det = ~still | (rng.random(len(r)) < p.minor_motion_pd)
                snr = np.where(still, snr - 3.0, snr)
            det &= snr >= p.cfar_db
            r, az, el, vr, snr = r[det], az[det], el[det], vr[det], snr[det]

            if len(r):  # one detection per (range bin, angle cell): unresolved scatterers merge
                cell = np.radians(p.merge_ang_deg)
                key = (np.floor(r / p.range_res).astype(np.int64) * 4096
                       + np.floor((az + fov_az) / cell).astype(np.int64) * 64
                       + np.floor((el + fov_el) / cell).astype(np.int64))
                _, inv = np.unique(key, return_inverse=True)
                w = 10 ** (snr / 10)
                W = np.bincount(inv, w)
                r, az, el, vr = (np.bincount(inv, w * x) / W for x in (r, az, el, vr))
                snr = 10 * np.log10(W)
                sig = np.clip(np.radians(p.ang_res_deg) / (1.5 * np.sqrt(W)), np.radians(0.5), np.radians(10))
                r = r + rng.normal(0, 0.3 * p.range_res, len(r))
                az = az + rng.normal(0, 1, len(r)) * sig
                el = el + rng.normal(0, 1, len(r)) * sig
                vr = vr + rng.normal(0, 0.5 * p.vel_res, len(r))
                out = np.column_stack([r, az, el, vr, snr])

        nfa = rng.poisson(p.false_alarm_rate)
        if nfa:
            vfa = rng.uniform(p.vel_res, 1.0, nfa) * rng.choice([-1, 1], nfa)
            fa = np.column_stack([rng.uniform(p.min_range, p.max_range, nfa), rng.uniform(-fov_az, fov_az, nfa),
                                  rng.uniform(-fov_el, fov_el, nfa), vfa, p.cfar_db + rng.exponential(1.5, nfa)])
            out = np.concatenate([out, fa])
        if len(out) > p.max_points:
            out = out[np.argsort(-out[:, 4])[:p.max_points]]
        return out.astype(np.float32)
