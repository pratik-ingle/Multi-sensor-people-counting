"""Floor plan: room bounds, furniture (obstacles), browsing zones and walking-navigation fields."""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Box:
    name: str
    x0: float
    y0: float
    x1: float
    y1: float
    height: float = 0.0
    reflective: bool = False
    kind: str = "block"
    seating: bool = False   # people sit on it: sensor returns inside its footprint are real

    def contains(self, x, y, margin: float = 0.0):
        x = np.asarray(x)
        y = np.asarray(y)
        return (x >= self.x0 - margin) & (x <= self.x1 + margin) & (y >= self.y0 - margin) & (y <= self.y1 + margin)

    @property
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)


@dataclass
class Zone:
    name: str
    box: Box
    weight: float
    dwell_median_s: float
    final: bool = False
    seats: list = field(default_factory=list)   # dicts: floor spot (xf, yf), body (xb, yb), h, facing, back
    sit_prob: float = 0.0
    sit_dwell_scale: float = 1.0


def _parse_seats(s: dict | None) -> list[dict]:
    if not s:
        return []
    out = []
    for x in s["x"]:
        out.append({"xf": float(x), "yf": float(s["y"]), "xb": float(x), "yb": float(s["y"]) + float(s["offset_y"]),
                    "h": float(s["height"]), "facing": float(np.radians(s.get("facing_deg", -90.0))),
                    "back": bool(s.get("backrest", False))})
    return out


class Layout:
    def __init__(self, cfg: dict, body_radius: float = 0.3):
        self.width = float(cfg["width"])
        self.depth = float(cfg["depth"])
        self.res = float(cfg.get("grid_res", 0.25))
        self.obstacles = [
            Box(o["name"], *o["box"], height=float(o.get("height", 1.0)), reflective=bool(o.get("reflective", False)),
                kind=o.get("type", "block"), seating=bool(o.get("seating", False)))
            for o in cfg.get("obstacles", [])
        ]
        self.zones = [
            Zone(z["name"], Box(z["name"], *z["box"]), float(z.get("weight", 1.0)),
                 float(z.get("dwell_median_s", 30.0)), bool(z.get("final", False)),
                 seats=_parse_seats(z.get("seats")), sit_prob=float((z.get("seats") or {}).get("sit_prob", 0.0)),
                 sit_dwell_scale=float((z.get("seats") or {}).get("dwell_scale", 1.0)))
            for z in cfg.get("zones", [])
        ]
        self.entrance = Box("entrance", *cfg["entrance"])
        self.exit = Box("exit", *cfg["exit"])

        self.nx = int(round(self.width / self.res))
        self.ny = int(round(self.depth / self.res))
        xc = (np.arange(self.nx) + 0.5) * self.res
        yc = (np.arange(self.ny) + 0.5) * self.res
        self.XC, self.YC = np.meshgrid(xc, yc)  # (ny, nx)
        blocked = (self.XC < body_radius) | (self.XC > self.width - body_radius) | \
                  (self.YC < body_radius) | (self.YC > self.depth - body_radius)
        for o in self.obstacles:
            blocked |= o.contains(self.XC, self.YC, margin=body_radius)
        self.free = ~blocked

        self._fields = {i: self._direction_field(z.box) for i, z in enumerate(self.zones)}
        self._fields["exit"] = self._direction_field(self.exit)

    # ---------------------------------------------------------------- navigation
    def _direction_field(self, target: Box):
        """Dijkstra distance-to-target over free cells, then a unit step direction per cell."""
        ny, nx = self.free.shape
        dist = np.full((ny, nx), np.inf)
        tgt = target.contains(self.XC, self.YC) & self.free
        heap = []
        for j, i in zip(*np.nonzero(tgt)):
            dist[j, i] = 0.0
            heap.append((0.0, int(j), int(i)))
        heapq.heapify(heap)
        nbrs = [(dj, di, float(np.hypot(dj, di))) for dj in (-1, 0, 1) for di in (-1, 0, 1) if dj or di]
        free = self.free
        while heap:
            d, j, i = heapq.heappop(heap)
            if d > dist[j, i]:
                continue
            for dj, di, c in nbrs:
                jj, ii = j + dj, i + di
                if not (0 <= jj < ny and 0 <= ii < nx) or not free[jj, ii]:
                    continue
                if dj and di and not (free[j, ii] and free[jj, i]):
                    continue  # no corner cutting
                nd = d + c
                if nd < dist[jj, ii]:
                    dist[jj, ii] = nd
                    heapq.heappush(heap, (nd, jj, ii))

        pad = np.pad(dist, 1, constant_values=np.inf)
        best = dist.copy()
        bdx = np.zeros((ny, nx))
        bdy = np.zeros((ny, nx))
        for dj, di, _ in nbrs:
            cand = pad[1 + dj:1 + dj + ny, 1 + di:1 + di + nx]
            better = cand < best
            best = np.where(better, cand, best)
            bdx = np.where(better, di, bdx)
            bdy = np.where(better, dj, bdy)
        v = np.stack([bdx, bdy], axis=-1)
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        v = np.divide(v, n, out=np.zeros_like(v), where=n > 0)
        return dist, v

    def direction(self, key, pos) -> np.ndarray:
        i = int(np.clip(pos[0] // self.res, 0, self.nx - 1))
        j = int(np.clip(pos[1] // self.res, 0, self.ny - 1))
        return self._fields[key][1][j, i]

    # ---------------------------------------------------------------- queries
    def walkable(self, x, y, margin: float = 0.15):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        ok = (x > margin) & (x < self.width - margin) & (y > margin) & (y < self.depth - margin)
        for o in self.obstacles:
            ok &= ~o.contains(x, y, margin)
        return ok

    def walkable_point(self, p) -> bool:
        return bool(self.walkable(p[0], p[1]))

    def zone_index(self, x, y) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.full(x.shape, -1, dtype=int)
        for k, z in enumerate(self.zones):
            out[(out < 0) & z.box.contains(x, y)] = k
        return out

    def random_point(self, box: Box, rng) -> np.ndarray:
        js, is_ = np.nonzero(box.contains(self.XC, self.YC) & self.free)
        if len(js) == 0:
            return np.array([(box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2])
        k = rng.integers(len(js))
        jitter = rng.uniform(-0.4, 0.4, 2) * self.res
        return np.array([self.XC[js[k], is_[k]], self.YC[js[k], is_[k]]]) + jitter
