"""Agent-based shopper model: groups arrive, walk to browsing zones, dwell (standing or sitting
on display sofas / beds), and leave.

Groups (couples, families with children) walk together closely, which is the hardest case for
counting, so they are modelled explicitly. Each person also gets clothing / hair attributes that
set their surface temperatures for the thermal cameras.
"""
from __future__ import annotations

import numpy as np

from .layout import Layout
from .meshes import FACE, FOREARM, HAND, N_SLOTS, NECK, SCALP_SIDE, SCALP_TOP, SHIN, SHOE, THIGH, TOP

WALK_TO_ZONE, WALK_IN_ZONE, DWELL, SHIFT, WALK_TO_EXIT, FOLLOW = range(6)
BROWSING = (DWELL, SHIFT)
STAND, SIT = 0, 1


def dress(rng: np.random.Generator, child: bool):
    """Skin temperature and, per body slot, how far the surface sits between room air (0) and skin (1).
    Thicker clothing / longer hair -> surface closer to room temperature."""
    f = np.empty(N_SLOTS)
    tskin = rng.normal(33.6, 0.5)
    f[FACE] = rng.normal(1.0, 0.02)
    hair = rng.choice(4, p=[0.5, 0.45, 0.0, 0.05] if child else [0.45, 0.35, 0.07, 0.13])  # short/long/bald/hat
    f[SCALP_TOP] = [0.78, 0.55, 0.97, 0.25][hair] + rng.normal(0, 0.04)
    f[SCALP_SIDE] = [0.72, 0.50, 0.95, 0.60][hair] + rng.normal(0, 0.04)
    top = rng.choice(4, p=[0.35, 0.30, 0.20, 0.15])  # t-shirt / long sleeve / sweater / jacket
    f[TOP] = [0.70, 0.60, 0.50, 0.30][top] + rng.normal(0, 0.04)
    f[NECK] = 0.6 if top == 3 else 0.95
    f[FOREARM] = 0.85 if top == 0 else f[TOP]
    f[HAND] = rng.normal(0.82, 0.05)
    bottom = rng.choice(3, p=[0.75, 0.12, 0.13])  # trousers / shorts / skirt
    f[THIGH] = [0.50, 0.62, 0.62][bottom] + rng.normal(0, 0.04)
    f[SHIN] = [0.48, 0.80, 0.80][bottom] + rng.normal(0, 0.04)
    f[SHOE] = rng.normal(0.22, 0.04)
    return float(tskin), np.clip(f, 0.05, 1.05)


class Person:
    __slots__ = ("pid", "gid", "leader", "pos", "vel", "height", "radius", "speed", "plan", "state",
                 "target", "dwell_left", "shift_timer", "offset", "gesture_t", "phase", "heading", "done",
                 "pose", "seat", "sit_pending", "sit_t0", "tskin", "tfac", "girth", "quirk")

    def __init__(self, pid, gid, pos, height, radius, speed, leader=None, offset=None):
        self.pid = pid
        self.gid = gid
        self.leader = leader
        self.pos = np.asarray(pos, dtype=float)
        self.vel = np.zeros(2)
        self.height = height
        self.radius = radius
        self.speed = speed
        self.plan: list[int] = []
        self.state = FOLLOW if leader is not None else WALK_TO_ZONE
        self.target = None
        self.dwell_left = 0.0
        self.shift_timer = 0.0
        self.offset = np.zeros(2) if offset is None else np.asarray(offset, dtype=float)
        self.gesture_t = 0.0
        self.phase = 0.0
        self.heading = np.pi / 2
        self.done = False
        self.pose = STAND
        self.seat = -1
        self.sit_pending = False
        self.sit_t0 = 0.0


class Crowd:
    def __init__(self, layout: Layout, cfg: dict, rng: np.random.Generator):
        self.lay = layout
        self.rng = rng
        self.people: list[Person] = []
        self.t = 0.0
        self._next_pid = 0
        self._next_gid = 0
        self.arrival_rate = float(cfg.get("group_arrivals_per_min", 10)) / 60.0
        p = np.asarray(cfg.get("group_size_probs", [1.0]), dtype=float)
        self.size_p = p / p.sum()
        self.zmin, self.zmax = cfg.get("zones_per_visit", [2, 4])
        self.checkout_prob = float(cfg.get("checkout_prob", 0.6))
        self.speed_lo, self.speed_hi = cfg.get("walk_speed", [0.8, 1.4])
        self.dwell_sigma = float(cfg.get("dwell_sigma", 0.6))
        self.max_people = int(cfg.get("max_people", 250))
        self.browse = [i for i, z in enumerate(layout.zones) if not z.final]
        self.finals = [i for i, z in enumerate(layout.zones) if z.final]
        w = np.array([layout.zones[i].weight for i in self.browse], dtype=float)
        self.browse_w = w / w.sum()
        # seats: global list, with the zone each belongs to
        self.seats: list[dict] = []
        self.zone_seats: dict[int, list[int]] = {}
        for k, z in enumerate(layout.zones):
            for s in z.seats:
                self.zone_seats.setdefault(k, []).append(len(self.seats))
                self.seats.append({**s, "zone": k})
        self.seat_taken = np.zeros(len(self.seats), dtype=bool)
        self.events: list[tuple] = []  # ("stood_up", seat_id, t, seconds_seated) -- consumed by the thermal scene

    # ---------------------------------------------------------------- spawning
    def _plan(self) -> list[int]:
        k = min(int(self.rng.integers(self.zmin, self.zmax + 1)), len(self.browse))
        plan = [int(z) for z in self.rng.choice(self.browse, size=k, replace=False, p=self.browse_w)]
        if self.finals and self.rng.random() < self.checkout_prob:
            plan.append(int(self.rng.choice(self.finals)))
        return plan

    def _spawn_group(self):
        rng = self.rng
        size = int(rng.choice(len(self.size_p), p=self.size_p)) + 1
        gid = self._next_gid
        self._next_gid += 1
        start = self.lay.random_point(self.lay.entrance, rng)
        speed = rng.uniform(self.speed_lo, self.speed_hi)
        offsets = [(-0.8, 0.45), (-0.8, -0.45), (-1.5, 0.0), (0.0, 0.7)]  # behind / beside the leader
        members: list[Person] = []
        for k in range(size):
            child = size >= 3 and k >= 1 and rng.random() < 0.5
            h = float(np.clip(rng.normal(1.25, 0.12) if child else rng.normal(1.72, 0.09), 0.95, 2.05))
            radius = float(np.clip(0.2 * h / 1.72 + rng.normal(0, 0.015), 0.12, 0.26))
            pos = start if k == 0 else start + rng.normal(0, 0.3, 2)
            if not self.lay.walkable_point(pos):
                pos = start.copy()
            off = None if k == 0 else np.array(offsets[(k - 1) % len(offsets)]) + rng.normal(0, 0.1, 2)
            person = Person(self._next_pid, gid, pos, h, radius, speed, leader=members[0] if k else None, offset=off)
            person.tskin, person.tfac = dress(rng, child)
            person.girth = float(np.clip(rng.normal(1.0, 0.07), 0.8, 1.25))
            person.quirk = float(rng.normal(0, 1))
            self._next_pid += 1
            if k == 0:
                person.plan = self._plan()
            members.append(person)
        self.people.extend(members)

    # ---------------------------------------------------------------- seats
    def _browse_spot(self, p: Person, zone_idx: int, near=None) -> np.ndarray:
        """Where to browse: sometimes a free seat (sofas, beds), otherwise a free floor point."""
        z = self.lay.zones[zone_idx]
        free = [s for s in self.zone_seats.get(zone_idx, []) if not self.seat_taken[s]]
        if free and self.rng.random() < z.sit_prob:
            if near is not None:
                free.sort(key=lambda s: np.hypot(self.seats[s]["xf"] - near[0], self.seats[s]["yf"] - near[1]))
                s = free[0]
            else:
                s = int(self.rng.choice(free))
            self.seat_taken[s] = True
            p.seat = s
            p.sit_pending = True
            return np.array([self.seats[s]["xf"], self.seats[s]["yf"]])
        if near is not None:
            cand = near + self.rng.normal(0, 0.7, 2)
            if z.box.contains(*cand) and self.lay.walkable_point(cand):
                return cand
            return np.array(near, dtype=float)
        return self.lay.random_point(z.box, self.rng)

    def _arrive(self, p: Person):
        if p.sit_pending:
            p.pose = SIT
            p.sit_pending = False
            p.sit_t0 = self.t
            p.pos = np.array([self.seats[p.seat]["xf"], self.seats[p.seat]["yf"]])

    def _stand_up(self, p: Person):
        if p.seat >= 0:
            if p.pose == SIT:
                self.events.append(("stood_up", p.seat, self.t, self.t - p.sit_t0))
            self.seat_taken[p.seat] = False
            p.seat = -1
        p.pose = STAND
        p.sit_pending = False

    # ---------------------------------------------------------------- behaviour
    def _leader(self, p: Person, dt: float) -> np.ndarray:
        lay, rng = self.lay, self.rng
        z = lay.zones[p.plan[0]] if p.plan else None
        if p.state == WALK_TO_ZONE:
            if z.box.contains(*p.pos):
                p.target = self._browse_spot(p, p.plan[0])
                p.state = WALK_IN_ZONE
            else:
                return lay.direction(p.plan[0], p.pos) * p.speed
        if p.state == WALK_TO_EXIT:
            if lay.exit.contains(*p.pos):
                p.done = True
                return np.zeros(2)
            return lay.direction("exit", p.pos) * p.speed
        if p.state in (WALK_IN_ZONE, SHIFT):
            if p.state == SHIFT:
                p.dwell_left -= dt
            d = p.target - p.pos
            dist = float(np.hypot(*d))
            if dist < 0.25:
                if p.state == WALK_IN_ZONE:
                    self._arrive(p)
                    scale = z.sit_dwell_scale if p.pose == SIT else 1.0
                    p.dwell_left = rng.lognormal(np.log(z.dwell_median_s * scale), self.dwell_sigma)
                p.state = DWELL
                p.shift_timer = rng.exponential(12.0)
                return np.zeros(2)
            spd = 0.8 * p.speed if p.state == WALK_IN_ZONE else 0.4
            return d / dist * min(spd, dist)
        if p.state == DWELL:
            p.dwell_left -= dt
            p.shift_timer -= dt
            if p.dwell_left <= 0:
                self._stand_up(p)
                p.plan.pop(0)
                p.state = WALK_TO_ZONE if p.plan else WALK_TO_EXIT
                p.target = None
            elif p.shift_timer <= 0 and p.pose == STAND:  # step to look at another item
                cand = p.pos + rng.normal(0, 0.9, 2)
                if z.box.contains(*cand) and lay.walkable_point(cand):
                    p.target = cand
                    p.state = SHIFT
                p.shift_timer = rng.exponential(12.0)
        return np.zeros(2)

    def _follower(self, p: Person, dt: float) -> np.ndarray:
        L = p.leader
        lay, rng = self.lay, self.rng
        if L.done:  # leader left the store: head for the exit independently
            self._stand_up(p)
            p.leader = None
            p.plan = []
            p.state = WALK_TO_EXIT
            return self._leader(p, dt)
        if L.state in (WALK_TO_ZONE, WALK_TO_EXIT, WALK_IN_ZONE):
            if p.pose == SIT or p.seat >= 0:
                self._stand_up(p)
            p.state = FOLLOW
            c, s = np.cos(L.heading), np.sin(L.heading)
            goal = L.pos + np.array([c * p.offset[0] - s * p.offset[1], s * p.offset[0] + c * p.offset[1]])
            d = goal - p.pos
            v = d * 1.5
            n = float(np.hypot(*v))
            if n > 1.8:
                v *= 1.8 / n
            if np.hypot(*d) > 0.2 and not lay.walkable_point(p.pos + v * dt * 3):
                key = L.plan[0] if (L.plan and L.state != WALK_TO_EXIT) else "exit"
                v = lay.direction(key, p.pos) * max(L.speed, 1.0)
            return v
        # Leader is browsing: pick an own spot nearby in the same zone (maybe a seat next to them).
        z = lay.zones[L.plan[0]]
        if p.state == FOLLOW:
            anchor = L.target if L.target is not None else L.pos
            p.target = self._browse_spot(p, L.plan[0], near=np.asarray(anchor, dtype=float))
            p.state = WALK_IN_ZONE
        if p.state in (WALK_IN_ZONE, SHIFT):
            d = p.target - p.pos
            dist = float(np.hypot(*d))
            if dist < 0.25:
                if p.state == WALK_IN_ZONE:
                    self._arrive(p)
                p.state = DWELL
                p.shift_timer = rng.exponential(10.0)
                return np.zeros(2)
            return d / dist * min(0.6, dist)
        if p.state == DWELL:
            p.shift_timer -= dt
            if p.shift_timer <= 0 and p.pose == STAND:
                cand = L.pos + rng.normal(0, 0.8, 2)
                if z.box.contains(*cand) and lay.walkable_point(cand):
                    p.target = cand
                    p.state = SHIFT
                p.shift_timer = rng.exponential(10.0)
        return np.zeros(2)

    # ---------------------------------------------------------------- dynamics
    def step(self, dt: float):
        rng = self.rng
        self.t += dt
        for _ in range(rng.poisson(self.arrival_rate * dt)):
            if len(self.people) < self.max_people:
                self._spawn_group()
        if not self.people:
            return

        desired = {}
        for p in self.people:  # leaders first so followers react to the updated leader state
            if p.leader is None:
                desired[p.pid] = self._leader(p, dt)
        for p in self.people:
            if p.pid not in desired:
                desired[p.pid] = self._follower(p, dt)

        for p in self.people:  # hand gestures while browsing produce the only Doppler of a still person
            if p.gesture_t > 0:
                p.gesture_t -= dt
            elif rng.random() < (0.12 if p.state in BROWSING else 0.02) * dt:
                p.gesture_t = rng.uniform(0.5, 2.5)

        P = np.array([p.pos for p in self.people])
        V = np.array([p.vel for p in self.people])
        D = np.array([desired[p.pid] for p in self.people])
        G = np.array([p.gid for p in self.people])
        seated = np.array([p.pose == SIT for p in self.people])
        if len(P) > 1:  # keep personal space
            diff = P[:, None, :] - P[None, :, :]
            dist = np.hypot(diff[..., 0], diff[..., 1])
            np.fill_diagonal(dist, np.inf)
            dmin = np.where(G[:, None] == G[None, :], 0.45, 0.6)
            overlap = np.clip(dmin - dist, 0.0, None)
            with np.errstate(invalid="ignore", divide="ignore"):
                push = np.nansum(diff / dist[..., None] * overlap[..., None], axis=1)
            D = D + 2.5 * push

        V = V + (D - V) * min(1.0, dt / 0.4)
        V[seated] = 0.0
        spd = np.hypot(V[:, 0], V[:, 1])
        V *= np.minimum(1.0, 2.0 / np.maximum(spd, 1e-9))[:, None]
        new = P + V * dt
        bad = ~self.lay.walkable(new[:, 0], new[:, 1])
        if bad.any():  # slide along obstacles
            tx = np.column_stack([new[bad, 0], P[bad, 1]])
            ty = np.column_stack([P[bad, 0], new[bad, 1]])
            okx = self.lay.walkable(tx[:, 0], tx[:, 1])
            oky = self.lay.walkable(ty[:, 0], ty[:, 1])
            fix = np.where(okx[:, None], tx, np.where(oky[:, None], ty, P[bad]))
            new[bad] = fix
            V[bad] = (fix - P[bad]) / dt

        for k, p in enumerate(self.people):
            p.pos = new[k]
            p.vel = V[k]
            s = float(np.hypot(*V[k]))
            if s > 0.15:
                p.heading = float(np.arctan2(V[k, 1], V[k, 0]))
            p.phase = (p.phase + 2 * np.pi * (1.8 * min(s, 1.6) / 1.2) * dt) % (2 * np.pi)
        self.people = [p for p in self.people if not p.done]

    def snapshot(self) -> dict:
        """Per-person arrays. 'pos' / 'heading' are the BODY's: a seated person's pelvis is on the seat."""
        ps = self.people
        pos, heading, seat_h = [], [], []
        for p in ps:
            if p.pose == SIT:
                s = self.seats[p.seat]
                pos.append((s["xb"], s["yb"]))
                heading.append(s["facing"])
                seat_h.append(s["h"])
            else:
                pos.append(p.pos)
                heading.append(p.heading)
                seat_h.append(0.0)
        return {
            "pid": np.array([p.pid for p in ps], dtype=int),
            "gid": np.array([p.gid for p in ps], dtype=int),
            "pos": np.array(pos, dtype=float).reshape(-1, 2),
            "vel": np.array([p.vel for p in ps], dtype=float).reshape(-1, 2),
            "height": np.array([p.height for p in ps], dtype=float),
            "radius": np.array([p.radius for p in ps], dtype=float),
            "heading": np.array(heading, dtype=float),
            "phase": np.array([p.phase for p in ps], dtype=float),
            "gesture": np.array([p.gesture_t > 0 for p in ps], dtype=bool),
            "browsing": np.array([p.state in BROWSING for p in ps], dtype=bool),
            "pose": np.array([p.pose for p in ps], dtype=int),
            "seat_h": np.array(seat_h, dtype=float),
            "girth": np.array([p.girth for p in ps], dtype=float),
            "quirk": np.array([p.quirk for p in ps], dtype=float),
            "tskin": np.array([p.tskin for p in ps], dtype=float),
            "tfac": np.array([p.tfac for p in ps], dtype=float).reshape(-1, N_SLOTS),
        }
