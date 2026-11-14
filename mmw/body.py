"""Human body as a cloud of radar scatterers with walking micro-Doppler.

Each person gets ~40 surface scatterers per frame (head, shoulders, torso, swinging arms and
legs). Each scatterer carries a position, velocity, outward normal (for self-occlusion) and a
relative RCS. Sampled once per frame and shared by all sensors so multi-sensor views agree.
"""
from __future__ import annotations

import numpy as np

EMPTY = {"P": np.zeros((0, 3)), "V": np.zeros((0, 3)), "N": np.zeros((0, 3)),
         "rcs": np.zeros(0), "owner": np.zeros(0, dtype=int)}


def sample_scatterers(snap: dict, rng: np.random.Generator) -> dict:
    pos = snap["pos"]
    n = len(pos)
    if n == 0:
        return EMPTY
    vel, h, R = snap["vel"], snap["height"], snap["radius"]
    hd, ph, gest = snap["heading"], snap["phase"], snap["gesture"]
    speed = np.hypot(vel[:, 0], vel[:, 1])
    fwd = np.column_stack([np.cos(hd), np.sin(hd), np.zeros(n)])
    perp = np.column_stack([-np.sin(hd), np.cos(hd), np.zeros(n)])
    sway = rng.normal(0.0, 0.035, (n, 3)) * np.array([1.0, 1.0, 0.3])  # weight shifts of a standing person
    base_v = np.column_stack([vel, np.zeros(n)]) + sway
    size_db = 20 * np.log10(h / 1.72)  # children reflect less
    out_P, out_V, out_N, out_rcs, out_own = [], [], [], [], []

    def emit(P, V, Nrm, rcs_db, k):
        out_P.append(P.reshape(-1, 3))
        out_V.append(V.reshape(-1, 3))
        out_N.append(Nrm.reshape(-1, 3))
        out_rcs.append((rcs_db + size_db[:, None] + np.zeros((n, k))).reshape(-1))
        out_own.append(np.repeat(np.arange(n), k))

    def ring(center, radius, zlo, zhi, k):
        phi = rng.uniform(0, 2 * np.pi, (n, k))
        ux, uy = np.cos(phi), np.sin(phi)
        P = np.stack([center[:, None, 0] + radius[:, None] * ux,
                      center[:, None, 1] + radius[:, None] * uy,
                      rng.uniform(zlo[:, None], zhi[:, None], (n, k))], axis=-1)
        return P, np.stack([ux, uy, np.zeros_like(ux)], axis=-1)

    def jitter(k):
        return rng.normal(0.0, 0.01, (n, k, 3))

    # head top + shoulders (seen from a ceiling sensor)
    k = 6
    rr = np.sqrt(rng.uniform(0, 1, (n, k))) * 0.7 * R[:, None]
    a = rng.uniform(0, 2 * np.pi, (n, k))
    z = np.where(rng.uniform(size=(n, k)) < 0.5, h[:, None], 0.82 * h[:, None])
    P = np.stack([pos[:, None, 0] + rr * np.cos(a), pos[:, None, 1] + rr * np.sin(a), z], axis=-1)
    emit(P, base_v[:, None, :] + jitter(k), np.broadcast_to([0.0, 0.0, 1.0], (n, k, 3)), 0.0, k)

    # head side
    k = 4
    P, Nrm = ring(pos, np.full(n, 0.1), 0.87 * h, h, k)
    emit(P, base_v[:, None, :] + jitter(k), Nrm, -2.0, k)

    # torso
    k = 14
    P, Nrm = ring(pos, R, 0.5 * h, 0.82 * h, k)
    emit(P, base_v[:, None, :] + jitter(k), Nrm, 0.0, k)

    # arms: swing opposite to each other while walking, gesture while browsing
    k = 8
    side = np.tile([1.0, -1.0], k // 2)
    swing_phase = ph[:, None] + np.where(side > 0, 0.0, np.pi)[None, :]
    amp = np.minimum(speed / 1.2, 1.0)[:, None]
    center = pos[:, None, :] + side[None, :, None] * (R[:, None, None] + 0.06) * perp[:, None, :2] \
        + fwd[:, None, :2] * (0.2 * amp * np.sin(swing_phase))[..., None]
    phi = rng.uniform(0, 2 * np.pi, (n, k))
    P = np.stack([center[..., 0] + 0.05 * np.cos(phi), center[..., 1] + 0.05 * np.sin(phi),
                  rng.uniform(0.45 * h[:, None], 0.8 * h[:, None], (n, k))], axis=-1)
    Nrm = np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], axis=-1)
    V = base_v[:, None, :] + fwd[:, None, :] * (0.7 * speed[:, None] * np.cos(swing_phase))[..., None]
    V = V + gest[:, None, None] * rng.normal(0.0, 0.35, (n, k, 3))
    emit(P, V + jitter(k), Nrm, -4.0, k)

    # legs: lower parts move faster (feet go 0 .. 2x walking speed)
    k = 8
    side = np.tile([1.0, -1.0], k // 2)
    swing_phase = ph[:, None] + np.where(side > 0, 0.0, np.pi)[None, :]
    center = pos[:, None, :] + side[None, :, None] * 0.1 * perp[:, None, :2]
    phi = rng.uniform(0, 2 * np.pi, (n, k))
    z = rng.uniform(0.05 * h[:, None], 0.48 * h[:, None], (n, k))
    P = np.stack([center[..., 0] + 0.07 * np.cos(phi), center[..., 1] + 0.07 * np.sin(phi), z], axis=-1)
    Nrm = np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], axis=-1)
    factor = (0.48 * h[:, None] - z) / (0.43 * h[:, None])
    V = base_v[:, None, :] + fwd[:, None, :] * (speed[:, None] * factor * np.sin(swing_phase))[..., None]
    emit(P, V + jitter(k), Nrm, -3.0, k)

    P = np.concatenate(out_P)
    own = np.concatenate(out_own)
    sit = snap.get("pose", np.zeros(n, dtype=int)) == 1
    if sit.any():  # fold seated bodies: upper body lowered onto the seat, thighs forward, shins down
        idx = np.nonzero(sit[own])[0]
        o = own[idx]
        z = P[idx, 2]
        hip = 0.53 * h[o]
        seat = snap["seat_h"][o] + 0.07
        up = z >= hip
        P[idx[up], 2] = z[up] - hip[up] + seat[up]
        lo = ~up
        frac = (hip[lo] - z[lo]) / hip[lo]                       # 0 at the hip, 1 at the floor
        thigh = frac < 0.5
        reach = np.where(thigh, frac / 0.5, 1.0) * 0.45 * h[o][lo] / 1.72
        P[idx[lo], :2] += fwd[o][lo, :2] * reach[:, None]
        P[idx[lo], 2] = np.where(thigh, seat[lo], seat[lo] * (1 - (frac - 0.5) / 0.5))
    return {"P": P, "V": np.concatenate(out_V), "N": np.concatenate(out_N),
            "rcs": np.concatenate(out_rcs), "owner": own}
