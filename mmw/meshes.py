"""Triangle meshes: primitives, an articulated human body, and showroom furniture.

Human: 15 rigid body parts (torso, neck, head, upper/lower arms, hands, thighs, shins, feet)
on a skeleton, posed per frame for walking (gait cycle), standing, reaching and sitting.
Each vertex carries a material slot so clothing, hair and skin get their own temperatures.
"""
from __future__ import annotations

import numpy as np

# Material slots of the human body (columns of the per-person temperature-factor table).
FACE, SCALP_TOP, SCALP_SIDE, NECK, TOP, FOREARM, HAND, THIGH, SHIN, SHOE = range(10)
N_SLOTS = 10


# ------------------------------------------------------------------ primitives
def lathe(z, r, seg=12, ax=1.0, ay=1.0, cap_bottom=True, cap_top=True):
    """Surface of revolution along +z with elliptical cross-section (ax, ay scale x and y)."""
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    n = len(z)
    phi = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    V = np.stack([np.outer(r, np.cos(phi)) * ax, np.outer(r, np.sin(phi)) * ay,
                  np.repeat(z[:, None], seg, axis=1)], axis=-1).reshape(-1, 3)
    i = np.arange(n - 1)[:, None] * seg
    k = np.arange(seg)[None, :]
    k2 = (k + 1) % seg
    a, b, c, d = (i + k).ravel(), (i + seg + k).ravel(), (i + seg + k2).ravel(), (i + k2).ravel()
    F = [np.stack([a, b, c], 1), np.stack([a, c, d], 1)]
    verts = [V]
    nv = len(V)
    if cap_bottom and r[0] > 1e-9:
        verts.append([[0.0, 0.0, z[0]]])
        F.append(np.stack([np.full(seg, nv), (k2 % seg).ravel(), k.ravel()], 1))
        nv += 1
    if cap_top and r[-1] > 1e-9:
        verts.append([[0.0, 0.0, z[-1]]])
        top = (n - 1) * seg
        F.append(np.stack([np.full(seg, nv), top + k.ravel(), top + k2.ravel()], 1))
    return np.concatenate([np.asarray(v, dtype=float) for v in verts]), np.concatenate(F).astype(np.int64)


def ellipsoid(a, b, c, rings=10, seg=14):
    t = np.linspace(0, np.pi, rings)
    return lathe(-c * np.cos(t), np.sin(t), seg=seg, ax=a, ay=b, cap_bottom=False, cap_top=False)


def box(x0, y0, z0, x1, y1, z1):
    V = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], dtype=float)
    F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], dtype=np.int64)
    return V, F


def cylinder(cx, cy, z0, z1, r, seg=14):
    V, F = lathe([z0, z0, z1, z1], [0.0, r, r, 0.0], seg=seg, cap_bottom=False, cap_top=False)
    return V + [cx, cy, 0.0], F


def frustum_open(cx, cy, z0, z1, r0, r1, seg=16):
    """Open-ended cone (a lampshade: you can look into it from above)."""
    V, F = lathe([z0, z1], [r0, r1], seg=seg, cap_bottom=False, cap_top=False)
    return V + [cx, cy, 0.0], F


def grid_plane(x0, y0, x1, y1, z, cell):
    nx, ny = max(1, int(round((x1 - x0) / cell))), max(1, int(round((y1 - y0) / cell)))
    xs, ys = np.linspace(x0, x1, nx + 1), np.linspace(y0, y1, ny + 1)
    X, Y = np.meshgrid(xs, ys)
    V = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], 1)
    i = (np.arange(ny)[:, None] * (nx + 1) + np.arange(nx)[None, :]).ravel()
    F = np.concatenate([np.stack([i, i + 1, i + nx + 2], 1), np.stack([i, i + nx + 2, i + nx + 1], 1)])
    return V, F


class MeshBuilder:
    """Collects static meshes with per-vertex thermal properties.
    dT: offset from room air temperature (passive objects follow the room);
    t_fixed: absolute temperature for powered things (lamps, screens); NaN = passive."""

    def __init__(self):
        self.V, self.F, self.dT, self.fixed, self.eps = [], [], [], [], []
        self.n = 0

    def add(self, mesh, dT=0.0, t_fixed=np.nan, eps=0.95):
        V, F = mesh
        m = len(V)
        self.V.append(V)
        self.F.append(F + self.n)
        self.dT.append(np.broadcast_to(np.asarray(dT, dtype=float), (m,)).copy())
        self.fixed.append(np.full(m, t_fixed, dtype=float))
        self.eps.append(np.full(m, eps, dtype=float))
        self.n += m

    def arrays(self):
        return (np.concatenate(self.V), np.concatenate(self.F), np.concatenate(self.dT),
                np.concatenate(self.fixed), np.concatenate(self.eps))


# ------------------------------------------------------------------ human body
def _rot_x(a):
    a = np.asarray(a, dtype=float)
    c, s = np.cos(a), np.sin(a)
    M = np.zeros(a.shape + (4, 4))
    M[..., 0, 0] = 1
    M[..., 1, 1], M[..., 1, 2], M[..., 2, 1], M[..., 2, 2] = c, -s, s, c
    M[..., 3, 3] = 1
    return M


def _rot_y(a):
    a = np.asarray(a, dtype=float)
    c, s = np.cos(a), np.sin(a)
    M = np.zeros(a.shape + (4, 4))
    M[..., 0, 0], M[..., 0, 2], M[..., 2, 0], M[..., 2, 2] = c, s, -s, c
    M[..., 1, 1] = 1
    M[..., 3, 3] = 1
    return M


def _rot_z(a):
    a = np.asarray(a, dtype=float)
    c, s = np.cos(a), np.sin(a)
    M = np.zeros(a.shape + (4, 4))
    M[..., 0, 0], M[..., 0, 1], M[..., 1, 0], M[..., 1, 1] = c, -s, s, c
    M[..., 2, 2] = 1
    M[..., 3, 3] = 1
    return M


def _trans(x, y, z):
    x, y, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float), np.asarray(z, float))
    M = np.zeros(x.shape + (4, 4))
    M[..., 0, 0] = M[..., 1, 1] = M[..., 2, 2] = M[..., 3, 3] = 1
    M[..., 0, 3], M[..., 1, 3], M[..., 2, 3] = x, y, z
    return M


def _scale(sx, sy, sz):
    sx, sy, sz = np.broadcast_arrays(np.asarray(sx, float), np.asarray(sy, float), np.asarray(sz, float))
    M = np.zeros(sx.shape + (4, 4))
    M[..., 0, 0], M[..., 1, 1], M[..., 2, 2], M[..., 3, 3] = sx, sy, sz, 1
    return M


class HumanTemplate:
    """Adult (1.72 m) body parts in local frames. Body frame: x right, y forward, z up."""

    PARTS = ["torso", "neck", "head", "uarm_l", "uarm_r", "farm_l", "farm_r", "hand_l", "hand_r",
             "thigh_l", "thigh_r", "shin_l", "shin_r", "foot_l", "foot_r"]

    def __init__(self, seed: int = 3):
        rng = np.random.default_rng(seed)
        parts = {}
        # torso from just below the hip joints (z=-0.10) to the shoulders (z=+0.50); chest depth 62 % of width
        zt = np.array([-0.10, -0.08, 0.0, 0.12, 0.25, 0.38, 0.45, 0.49, 0.52])
        rt = np.array([0.00, 0.13, 0.165, 0.150, 0.155, 0.175, 0.185, 0.15, 0.0])
        parts["torso"] = (lathe(zt, rt, seg=18, ax=1.0, ay=0.62), TOP)
        parts["neck"] = (lathe([0.50, 0.62], [0.055, 0.05], seg=10), NECK)
        hv, hf = ellipsoid(0.078, 0.098, 0.115, rings=12, seg=16)
        parts["head"] = ((hv + [0, 0.005, 0.71], hf), None)
        ua = lathe(-np.array([0.0, 0.02, 0.15, 0.28, 0.30]), [0.0, 0.052, 0.048, 0.040, 0.0], seg=10)
        fa = lathe(-np.array([0.0, 0.02, 0.13, 0.24, 0.26]), [0.0, 0.040, 0.038, 0.030, 0.0], seg=10)
        hv2, hf2 = ellipsoid(0.024, 0.045, 0.085, rings=7, seg=10)
        th = lathe(-np.array([0.0, 0.03, 0.2, 0.40, 0.43]), [0.0, 0.080, 0.072, 0.055, 0.0], seg=12)
        sh = lathe(-np.array([0.0, 0.02, 0.15, 0.40, 0.42]), [0.0, 0.050, 0.052, 0.035, 0.0], seg=10)
        fv, ff = ellipsoid(0.045, 0.125, 0.035, rings=7, seg=10)
        for side in "lr":
            parts[f"uarm_{side}"] = (ua, TOP)
            parts[f"farm_{side}"] = (fa, FOREARM)
            parts[f"hand_{side}"] = ((hv2 + [0, 0, -0.085], hf2), HAND)
            parts[f"thigh_{side}"] = (th, THIGH)
            parts[f"shin_{side}"] = (sh, SHIN)
            parts[f"foot_{side}"] = ((fv + [0, 0.065, -0.035], ff), SHOE)

        V, F, mat, pidx = [], [], [], []
        n = 0
        for k, name in enumerate(self.PARTS):
            (pv, pf), slot = parts[name]
            if name == "head":  # face in front-lower half, scalp on top, hair on sides/back
                lz = pv[:, 2] - 0.71
                m = np.where((pv[:, 1] > 0.03) & (lz < 0.035), FACE, np.where(lz > 0.02, SCALP_TOP, SCALP_SIDE))
            else:
                m = np.full(len(pv), slot)
            V.append(pv)
            F.append(pf + n)
            mat.append(m)
            pidx.append(np.full(len(pv), k))
            n += len(pv)
        self.V = np.concatenate(V)
        self.F = np.concatenate(F)
        self.mat = np.concatenate(mat)
        self.part = np.concatenate(pidx)
        self.noise = gaussian_like(rng, len(self.V))  # fixed per-vertex variation (clothing folds etc.)
        self.Vh = np.concatenate([self.V, np.ones((len(self.V), 1))], axis=1)
        self.part_rows = [np.nonzero(self.part == k)[0] for k in range(len(self.PARTS))]

    @property
    def n_verts(self) -> int:
        return len(self.V)

    def pose(self, snap: dict) -> np.ndarray:
        """World-space vertices for every person: (N, n_verts, 3)."""
        N = len(snap["pos"])
        if N == 0:
            return np.zeros((0, self.n_verts, 3))
        s = snap["height"] / 1.72
        g = snap.get("girth", np.ones(N))
        speed = np.hypot(snap["vel"][:, 0], snap["vel"][:, 1])
        sit = snap.get("pose", np.zeros(N, dtype=int)) == 1
        ph = snap["phase"]
        amp = np.where(sit, 0.0, np.clip(speed / 1.3, 0.0, 1.2))
        quirk = snap.get("quirk", np.zeros(N))  # per-person resting-pose variation

        thigh_l = np.where(sit, 1.45, 0.42 * amp * np.sin(ph) + 0.03 * quirk)
        thigh_r = np.where(sit, 1.40, -0.42 * amp * np.sin(ph) - 0.03 * quirk)
        knee_l = np.where(sit, 1.45, amp * (0.12 + 0.55 * np.clip(np.cos(ph), 0, None)) + 0.03)
        knee_r = np.where(sit, 1.40, amp * (0.12 + 0.55 * np.clip(-np.cos(ph), 0, None)) + 0.03)
        arm_l = np.where(sit, 0.35, -0.38 * amp * np.sin(ph) + 0.05)
        arm_r = np.where(sit, 0.30, 0.38 * amp * np.sin(ph) + 0.05)
        elbow = np.where(sit, 0.9, 0.2 + 0.25 * amp)
        reach = snap["gesture"] & ~sit
        arm_r = np.where(reach, 1.15, arm_r)
        elbow_r = np.where(reach, 0.35, elbow)
        lean = np.where(sit, -0.18, 0.06 * amp)
        hip_z = np.where(sit, snap.get("seat_h", np.zeros(N)) + 0.07 * s, 0.91 * s * (1 - 0.015 * amp))

        root = _trans(snap["pos"][:, 0], snap["pos"][:, 1], hip_z) @ _rot_z(snap["heading"] - np.pi / 2) \
            @ _scale(s, s, s)
        trunk = root @ _rot_x(lean)
        M = {}
        M["torso"] = trunk @ _scale(g, g, 1.0)
        M["neck"] = trunk
        M["head"] = trunk @ _trans(0, 0, 0.71) @ _rot_x(0.08 * quirk) @ _trans(0, 0, -0.71)
        for side, sx, arm, elb in (("l", -1.0, arm_l, elbow), ("r", 1.0, arm_r, elbow_r)):
            sh = trunk @ _trans(sx * 0.185 * g, 0, 0.45) @ _rot_x(arm) @ _rot_y(-sx * 0.10)  # arms hang slightly out
            M[f"uarm_{side}"] = sh
            el = sh @ _trans(0, 0, -0.30) @ _rot_x(elb)
            M[f"farm_{side}"] = el
            M[f"hand_{side}"] = el @ _trans(0, 0, -0.26)
        for side, sx, thg, kn in (("l", -1.0, thigh_l, knee_l), ("r", 1.0, thigh_r, knee_r)):
            hp = root @ _trans(sx * 0.09 * g, 0, 0) @ _rot_x(thg)
            M[f"thigh_{side}"] = hp
            kj = hp @ _trans(0, 0, -0.43) @ _rot_x(-kn)
            M[f"shin_{side}"] = kj
            M[f"foot_{side}"] = kj @ _trans(0, 0, -0.42) @ _rot_x(-(thg - kn))

        out = np.empty((N, self.n_verts, 3))
        for k, name in enumerate(self.PARTS):
            rows = self.part_rows[k]
            out[:, rows] = np.einsum("nij,vj->nvi", M[name][:, :3, :], self.Vh[rows])
        return out


def gaussian_like(rng, n, sigma=1.0):
    return rng.normal(0.0, sigma, n)


# ------------------------------------------------------------------ furniture
def build_furniture(mb: MeshBuilder, ob: dict, rng: np.random.Generator):
    """Detailed geometry for one obstacle from the layout config, chosen by its 'type'."""
    x0, y0, x1, y1 = ob["box"]
    h = float(ob.get("height", 1.0))
    kind = ob.get("type", "block")

    if kind == "shelf":  # painted steel shelving with boxed goods
        long_x = (x1 - x0) > (y1 - y0)
        L0, L1 = (x0, x1) if long_x else (y0, y1)
        for u in np.arange(L0, L1 + 1e-6, 1.25):
            mb.add(box(u - 0.03, y0, 0, u + 0.03, y1, h) if long_x else box(x0, u - 0.03, 0, x1, u + 0.03, h),
                   dT=0.0, eps=0.9)
        for z in (0.12, 0.6, 1.05, 1.5, 1.95):
            mb.add(box(x0, y0, z - 0.03, x1, y1, z), dT=0.0, eps=0.9)
            u = L0 + 0.05
            while u < L1 - 0.2:
                w = rng.uniform(0.2, 0.5)
                d = rng.uniform(0.3, 0.9) * ((y1 - y0) if long_x else (x1 - x0))
                hh = rng.uniform(0.12, 0.4) if z < 1.9 else rng.uniform(0.1, 0.3)
                c = (y0 + y1) / 2 if long_x else (x0 + x1) / 2
                goods = box(u, c - d / 2, z, min(u + w, L1), c + d / 2, z + hh) if long_x else \
                    box(c - d / 2, u, z, c + d / 2, min(u + w, L1), z + hh)
                mb.add(goods, dT=rng.normal(0.0, 0.3), eps=0.93)
                u += w + rng.uniform(0.02, 0.15)

    elif kind == "sofa_island":  # sofa facing south (customers sit here), back-to-back sofa, coffee table
        for (ys, sign) in ((y0, 1.0), (y1, -1.0)):
            seat0, seat1 = sorted((ys, ys + sign * 0.6))
            back0, back1 = sorted((ys + sign * 0.6, ys + sign * 0.85))
            mb.add(box(x0, seat0, 0.05, x1, seat1, 0.40), dT=0.1, eps=0.95)          # base
            for k in range(4):                                                     # seat cushions
                cx0 = x0 + 0.15 + k * 0.675
                mb.add(box(cx0 + 0.01, seat0, 0.40, cx0 + 0.665, seat1, 0.45), dT=0.2, eps=0.95)
            mb.add(box(x0, back0, 0.05, x1, back1, 0.9), dT=0.1, eps=0.95)          # backrest
            mb.add(box(x0, seat0, 0.05, x0 + 0.15, seat1, 0.62), dT=0.1, eps=0.95)  # armrests
            mb.add(box(x1 - 0.15, seat0, 0.05, x1, seat1, 0.62), dT=0.1, eps=0.95)
        cy = (y0 + y1) / 2
        mb.add(box(x0 + 0.6, cy - 0.45, 0.0, x1 - 0.6, cy + 0.45, 0.42), dT=0.0, eps=0.9)  # coffee table
        mb.add(cylinder(x0 + 0.9, cy, 0.42, 0.62, 0.08), dT=0.0, eps=0.9)                  # vase

    elif kind == "bed_island":  # bed: head at the south edge, customers sit on the north end
        mb.add(box(x0 + 0.55, y0 + 0.15, 0.0, x1 - 0.55, y1 - 0.05, 0.35), dT=0.0, eps=0.92)  # frame
        mb.add(box(x0 + 0.6, y0 + 0.2, 0.35, x1 - 0.6, y1 - 0.1, 0.55), dT=0.2, eps=0.95)     # mattress
        mb.add(box(x0 + 0.55, y0, 0.0, x1 - 0.55, y0 + 0.15, 1.05), dT=0.0, eps=0.92)         # headboard
        for px in (x0 + 0.7, x1 - 1.4):
            mb.add(box(px, y0 + 0.25, 0.55, px + 0.7, y0 + 0.7, 0.68), dT=0.3, eps=0.95)      # pillows
        for nx in (x0, x1 - 0.45):   # nightstands with bedside lamps (on)
            mb.add(box(nx, y0, 0.0, nx + 0.45, y0 + 0.45, 0.55), dT=0.0, eps=0.92)
            mb.add(cylinder(nx + 0.22, y0 + 0.22, 0.55, 0.78, 0.04), dT=0.0, eps=0.9)
            mb.add(frustum_open(nx + 0.22, y0 + 0.22, 0.72, 0.92, 0.15, 0.10), t_fixed=31.0, eps=0.95)
            mb.add(cylinder(nx + 0.22, y0 + 0.22, 0.78, 0.86, 0.035), t_fixed=48.0, eps=0.9)

    elif kind == "kitchen":  # base cabinets, worktop, wall units, steel fridge and hood, demo coffee machine
        mb.add(box(x0, y0, 0.0, x1, y0 + 0.6, 0.88), dT=0.0, eps=0.92)
        mb.add(box(x0, y0, 0.88, x1, y0 + 0.62, 0.92), dT=-0.2, eps=0.93)           # stone worktop
        mb.add(box(x0, y1 - 0.35, 1.45, x1 - 1.0, y1, 2.2), dT=0.0, eps=0.92)       # wall units
        mb.add(box(x1 - 0.9, y0, 0.0, x1, y1, 2.05), dT=0.0, eps=0.25)              # stainless fridge
        hx = (x0 + x1) / 2
        mb.add(box(hx - 0.45, y1 - 0.5, 1.6, hx + 0.45, y1, 1.75), dT=0.0, eps=0.3)  # steel hood
        mb.add(box(hx - 0.35, y0 + 0.05, 0.92, hx + 0.35, y0 + 0.55, 0.93), dT=0.0, eps=0.9)  # cooktop (off)
        cx = x0 + 2.0
        mb.add(box(cx, y0 + 0.2, 0.92, cx + 0.3, y0 + 0.55, 1.3), t_fixed=42.0, eps=0.9)   # coffee machine
        mb.add(cylinder(cx + 0.6, y0 + 0.35, 0.92, 1.12, 0.08), t_fixed=58.0, eps=0.85)  # kettle

    elif kind == "desk":  # checkout counter with two monitors and payment terminals
        mb.add(box(x0, y0, 0.0, x1, y1, h), dT=0.0, eps=0.92)
        for mx in (x0 + 0.8, x1 - 1.3):
            mb.add(box(mx, y0 + 0.45, h, mx + 0.5, y0 + 0.5, h + 0.35), t_fixed=33.0, eps=0.92)  # monitor
            mb.add(box(mx + 0.15, y0 + 0.5, h, mx + 0.35, y0 + 0.6, h + 0.3), t_fixed=37.0, eps=0.9)  # PSU
            mb.add(box(mx + 0.6, y0 + 0.2, h, mx + 0.75, y0 + 0.4, h + 0.06), t_fixed=29.0, eps=0.9)  # terminal

    elif kind == "lamp_table":  # lighting department: table lamps and floor lamps, most switched on
        mb.add(box(x0, y0, 0.0, x1, y1, h), dT=0.0, eps=0.92)
        cx = (x0 + x1) / 2
        for ly in np.arange(y0 + 0.6, y1 - 0.4, 0.8):
            on = rng.random() < 0.7
            r = rng.uniform(0.12, 0.2)
            mb.add(cylinder(cx, ly, h, h + 0.3, 0.035), dT=0.0, eps=0.9)
            mb.add(frustum_open(cx, ly, h + 0.25, h + 0.5, r, r * 0.65),
                   t_fixed=rng.uniform(30, 37) if on else np.nan, eps=0.95)
            if on:
                mb.add(cylinder(cx, ly, h + 0.28, h + 0.38, 0.03), t_fixed=rng.uniform(45, 60), eps=0.9)
        for ly in (y0 + 0.25, y1 - 0.25):   # floor lamps: shade at head height -- looks a lot like a head
            mb.add(cylinder(cx, ly, 0.0, 1.55, 0.02), dT=0.0, eps=0.3)
            mb.add(frustum_open(cx, ly, 1.5, 1.75, 0.2, 0.14), t_fixed=36.0, eps=0.95)
            mb.add(cylinder(cx, ly, 1.55, 1.65, 0.035), t_fixed=55.0, eps=0.9)

    else:
        mb.add(box(x0, y0, 0.0, x1, y1, h), dT=0.0, eps=0.92)
