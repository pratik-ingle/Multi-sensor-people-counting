"""Run files. Same layout for simulated and (later) real recordings, so the pipeline is shared.

points.npy  float32 (K, 7): frame, sensor, range_m, azimuth_rad, elevation_rad, doppler_mps, snr_db
            -- the content of TI's compressed point-cloud TLV, one row per detected point.
gt.npy      float32 (G, 10): frame, pid, gid, x, y, vx, vy, browsing, height, seated   (simulation only)
meta.json   config, frame rate, frame count, sensor poses (true + calibrated)

With thermal cameras enabled:
thermal_dets.npy          float32 (D, 6): frame, cam, x, y, strength_K, nadir_dist_m  -- per-camera edge detections
thermal_captures.npy      int32 (C, 2): frame, cam  -- every capture (a capture without detections clears that camera)
thermal_frames.npy        uint16 (S, cams, H, W) centi-kelvin snapshots of every camera (Lepton TLinear format)
thermal_frames_index.npy  radar frame index of each snapshot
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

POINT_COLS = ["frame", "sensor", "range", "azimuth", "elevation", "doppler", "snr"]
GT_COLS = ["frame", "pid", "gid", "x", "y", "vx", "vy", "browsing", "height", "seated"]


def save_run(out_dir, meta: dict, points: np.ndarray, gt: np.ndarray | None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "points.npy", points.astype(np.float32))
    if gt is not None:
        np.save(out / "gt.npy", gt.astype(np.float32))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


def load_run(run_dir):
    d = Path(run_dir)
    meta = json.loads((d / "meta.json").read_text())
    points = np.load(d / "points.npy")
    gt = np.load(d / "gt.npy") if (d / "gt.npy").exists() else None
    return meta, points, gt


def frame_slices(frame_col: np.ndarray, n_frames: int) -> np.ndarray:
    """Row boundaries per frame for a frame-sorted table: rows of frame f are [b[f], b[f+1])."""
    return np.searchsorted(frame_col, np.arange(n_frames + 1) - 0.5)
