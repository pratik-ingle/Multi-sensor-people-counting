"""Centralised multi-sensor people tracker, modelled on TI's gtrack.

Per frame:
1. predict every track (constant-velocity Kalman filter on x, y, vx, vy)
2. associate points to tracks individually (nearest track inside its gate) -- point-level
   association keeps two people walking side by side as two tracks once both exist
3. update tracks with the SNR-weighted centroid of their points
4. cluster leftover points pooled over the last few frames (DBSCAN) and start tentative
   tracks; confirm after a few hits. Pooling lets sparse, nearly-still people be found.
5. keep slow / browsing tracks alive much longer when they get no points, because static
   clutter removal strips still people of their points (TI calls this static retention)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN


@dataclass
class TrackerParams:
    gate_m: float = 0.7
    meas_sigma_m: float = 0.25
    accel_sigma: float = 1.5
    birth_eps_m: float = 0.4
    birth_min_points: int = 6
    birth_min_dist_m: float = 0.6
    confirm_hits: int = 4
    confirm_window: int = 6
    max_miss_moving: int = 10      # frames before a moving track is dropped
    max_miss_static: int = 200     # frames before a still track is dropped (20 s at 10 fps)
    static_speed: float = 0.3
    miss_velocity_decay: float = 0.8
    merge_dist_m: float = 0.3
    birth_window: int = 5          # frames of unassigned points pooled for births (finds still people)
    # Areas where a track may NOT be held as static (doors, exits) -- like gtrack's static boundary
    # box, inverted. Without this, people who walk out leave 'parked' ghost tracks at the door.
    no_static_zones: tuple = ()    # ((x0, y0, x1, y1), ...)


# Per-pipeline tuning. Radar loses people who stand still (static clutter removal), so its tracks
# must be held for a long time without data. Thermal cameras keep seeing still people, so holding
# tracks that long only keeps false ones alive; they also need more evidence before starting a
# track. Chosen by a sweep on the simulated showroom (see README).
PIPELINE_TUNING = {
    "radar": {},
    "thermal": {"max_miss_static": 50, "birth_min_points": 9},
    "fused": {"max_miss_static": 50, "birth_min_points": 9},
}


def make_tracker(pipeline: str, dt: float, **kw) -> "Tracker":
    return Tracker(dt, TrackerParams(**{**PIPELINE_TUNING.get(pipeline, {}), **kw}))


class Track:
    __slots__ = ("tid", "x", "P", "hits", "misses", "age", "confirmed")

    def __init__(self, tid, xy, sigma):
        self.tid = tid
        self.x = np.array([xy[0], xy[1], 0.0, 0.0])
        self.P = np.diag([sigma ** 2, sigma ** 2, 1.0, 1.0])
        self.hits = 1
        self.misses = 0
        self.age = 1
        self.confirmed = False

    @property
    def speed(self) -> float:
        return float(np.hypot(self.x[2], self.x[3]))


class Tracker:
    def __init__(self, dt: float, params: TrackerParams | None = None):
        self.dt = dt
        self.p = params or TrackerParams()
        self.tracks: list[Track] = []
        self._pool: list[np.ndarray] = []
        self._next = 0
        self.F = np.eye(4)
        self.F[0, 2] = self.F[1, 3] = dt
        q = self.p.accel_sigma ** 2
        self.Q = q * np.array([[dt ** 4 / 4, 0, dt ** 3 / 2, 0], [0, dt ** 4 / 4, 0, dt ** 3 / 2],
                               [dt ** 3 / 2, 0, dt ** 2, 0], [0, dt ** 3 / 2, 0, dt ** 2]])
        self.H = np.eye(2, 4)

    def step(self, xy: np.ndarray, snr_db: np.ndarray) -> list[Track]:
        p = self.p
        for t in self.tracks:
            t.x = self.F @ t.x
            t.P = self.F @ t.P @ self.F.T + self.Q
            t.age += 1

        assigned = np.full(len(xy), -1)
        if self.tracks and len(xy):
            T = np.array([t.x[:2] for t in self.tracks])
            gates = np.array([min(p.gate_m + 0.5 * np.sqrt(max(t.P[0, 0], t.P[1, 1])), 1.2) for t in self.tracks])
            d = np.hypot(xy[:, None, 0] - T[None, :, 0], xy[:, None, 1] - T[None, :, 1])
            d = np.where(d < gates[None, :], d, np.inf)
            best = np.argmin(d, axis=1)
            ok = np.isfinite(d[np.arange(len(xy)), best])
            assigned[ok] = best[ok]

        w = 10 ** (snr_db / 10)
        for k, t in enumerate(self.tracks):
            m = assigned == k
            if m.any():
                z = (xy[m] * w[m, None]).sum(0) / w[m].sum()
                n_eff = min(int(m.sum()), 10)
                Rm = np.eye(2) * (p.meas_sigma_m ** 2 / n_eff + 0.05 ** 2)
                S = self.H @ t.P @ self.H.T + Rm
                K = t.P @ self.H.T @ np.linalg.inv(S)
                t.x = t.x + K @ (z - self.H @ t.x)
                IKH = np.eye(4) - K @ self.H
                t.P = IKH @ t.P @ IKH.T + K @ Rm @ K.T  # Joseph form stays positive-definite
                t.hits += 1
                t.misses = 0
                if not t.confirmed and t.hits >= p.confirm_hits and t.age <= p.confirm_window + p.confirm_hits:
                    t.confirmed = True
            else:
                # No points: most likely the person stopped and static clutter removal hid them.
                # Bleed off velocity so the track parks instead of coasting away.
                t.misses += 1
                t.x[2:] *= p.miss_velocity_decay
                if t.speed < p.static_speed:
                    t.P[2:, 2:] = np.eye(2) * 0.05
                    t.P[:2, 2:] = 0.0
                    t.P[2:, :2] = 0.0
                    t.P[:2, :2] = np.eye(2) * min(max(t.P[0, 0], t.P[1, 1]), 0.5)

        keep = []
        for t in self.tracks:
            if not t.confirmed:
                alive = t.misses <= 2 and t.age <= p.confirm_window + p.confirm_hits
            else:
                may_park = t.speed < p.static_speed and not any(
                    x0 <= t.x[0] <= x1 and y0 <= t.x[1] <= y1 for x0, y0, x1, y1 in p.no_static_zones)
                alive = t.misses <= (p.max_miss_static if may_park else p.max_miss_moving)
            if alive:
                keep.append(t)
        self.tracks = keep

        self._pool.append(xy[assigned < 0])
        self._pool = self._pool[-p.birth_window:]
        fx = np.concatenate(self._pool)
        if len(fx) >= p.birth_min_points:
            labels = DBSCAN(eps=p.birth_eps_m, min_samples=p.birth_min_points).fit_predict(fx)
            T = np.array([t.x[:2] for t in self.tracks]) if self.tracks else np.zeros((0, 2))
            born = []
            for lab in set(labels) - {-1}:
                c = fx[labels == lab].mean(0)
                if len(T) and np.min(np.hypot(T[:, 0] - c[0], T[:, 1] - c[1])) < p.birth_min_dist_m:
                    continue
                self.tracks.append(Track(self._next, c, p.meas_sigma_m))
                self._next += 1
                born.append(c)
            if born:  # pooled points that just started a track must not start another
                B = np.array(born)
                self._pool = [q[np.min(np.hypot(q[:, None, 0] - B[None, :, 0], q[:, None, 1] - B[None, :, 1]),
                                       axis=1) >= p.birth_min_dist_m] if len(q) else q for q in self._pool]

        conf = [t for t in self.tracks if t.confirmed]
        drop = set()
        for i in range(len(conf)):  # merge duplicates (e.g. one person seen by two sensors)
            for j in range(i + 1, len(conf)):
                a, b = conf[i], conf[j]
                if a.tid in drop or b.tid in drop:
                    continue
                if np.hypot(*(a.x[:2] - b.x[:2])) < p.merge_dist_m:
                    drop.add(b.tid if b.hits < a.hits else a.tid)
        if drop:
            self.tracks = [t for t in self.tracks if t.tid not in drop]
        return [t for t in self.tracks if t.confirmed]
