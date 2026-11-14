"""Turn per-frame positions into store analytics and score them against ground truth."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .layout import Layout


def counts(frames: np.ndarray, zone_idx: np.ndarray, n_frames: int, n_zones: int):
    """Total and per-zone people count for every frame."""
    total = np.bincount(frames, minlength=n_frames)[:n_frames]
    per_zone = np.zeros((n_frames, n_zones), dtype=int)
    v = zone_idx >= 0
    np.add.at(per_zone, (frames[v], zone_idx[v]), 1)
    return total, per_zone


def heatmap(x, y, layout: Layout, dt: float, cell: float = 0.5) -> np.ndarray:
    """Person-seconds spent per square metre, shape (ny, nx)."""
    nx, ny = int(round(layout.width / cell)), int(round(layout.depth / cell))
    H, _, _ = np.histogram2d(x, y, bins=[nx, ny], range=[[0, layout.width], [0, layout.depth]])
    return H.T * dt / cell ** 2


def match_metrics(gt_slices, gt_xy, est_slices, est_xy, n_frames, gate=0.75):
    """One-to-one matching per frame. Returns summary metrics and a per-GT-row 'detected' mask."""
    tp = fp = fn = 0
    err = 0.0
    detected = np.zeros(len(gt_xy), dtype=bool)
    for f in range(n_frames):
        g0 = gt_slices[f]
        g = gt_xy[g0:gt_slices[f + 1]]
        e = est_xy[est_slices[f]:est_slices[f + 1]]
        if len(g) == 0 or len(e) == 0:
            fp += len(e)
            fn += len(g)
            continue
        d = np.hypot(g[:, None, 0] - e[None, :, 0], g[:, None, 1] - e[None, :, 1])
        r, c = linear_sum_assignment(np.where(d < gate, d, 1e6))
        ok = d[r, c] < gate
        detected[g0 + r[ok]] = True
        tp += int(ok.sum())
        err += float(d[r, c][ok].sum())
        fp += len(e) - int(ok.sum())
        fn += len(g) - int(ok.sum())
    return {"precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "mean_position_error_m": err / max(tp, 1)}, detected


def zone_summary(gt_zone, est_zone, layout: Layout, dt: float, duration_s: float) -> list[dict]:
    """Average occupancy and share of browsing time per zone -- the 'where do people engage' answer."""
    rows = []
    gt_tot = max(gt_zone.sum(), 1)
    est_tot = max(est_zone.sum(), 1)
    for k, z in enumerate(layout.zones):
        g, e = gt_zone[:, k].sum(), est_zone[:, k].sum()
        rows.append({"zone": z.name, "area_m2": z.box.area,
                     "avg_people_true": g * dt / duration_s, "avg_people_est": e * dt / duration_s,
                     "share_true": g / gt_tot, "share_est": e / est_tot,
                     "count_mae": float(np.abs(gt_zone[:, k] - est_zone[:, k]).mean())})
    return rows
