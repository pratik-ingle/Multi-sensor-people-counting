"""Run a showroom simulation: crowd -> radar point clouds (+ thermal camera frames and detections)."""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .body import sample_scatterers
from .crowd import Crowd
from .geometry import Pose
from .io import save_run
from .layout import Layout
from .radar import Radar, RadarParams


def build_poses(scfg: dict) -> list[Pose]:
    if scfg.get("layout", "grid") == "grid":
        return [Pose([x, y, scfg["height"]], scfg.get("yaw_deg", 0.0), scfg.get("pitch_deg", -90.0))
                for y in scfg["y"] for x in scfg["x"]]
    return [Pose(s["position"], s.get("yaw_deg", 0.0), s.get("pitch_deg", -90.0)) for s in scfg["list"]]


def perturb(poses: list[Pose], err: dict, rng) -> list[Pose]:
    """What an installer's calibration would report: true pose plus small errors."""
    pe, ae = float(err.get("pos_m", 0.0)), float(err.get("angle_deg", 0.0))
    return [Pose(p.position + rng.normal(0, pe, 3), p.yaw_deg + rng.normal(0, ae), p.pitch_deg + rng.normal(0, ae))
            for p in poses]


def run(cfg: dict, out_dir: str, verbose: bool = True):
    rng = np.random.default_rng(cfg.get("seed", 0))
    layout = Layout(cfg["layout"])
    crowd = Crowd(layout, cfg["crowd"], rng)
    params = RadarParams.from_dict(cfg.get("radar", {}))
    poses = build_poses(cfg["sensors"])
    calib = perturb(poses, cfg["sensors"].get("calibration_error", {}), rng)
    radars = [Radar(i, pose, params) for i, pose in enumerate(poses)]

    thermal = None
    if cfg.get("thermal", {}).get("enabled", False):
        from .thermal_system import ThermalSystem
        thermal = ThermalSystem(cfg, crowd.seats, rng)

    fr = float(cfg.get("frame_rate_hz", 10))
    dt = 1.0 / fr
    for _ in range(int(cfg.get("warmup_s", 0) * fr)):
        crowd.step(dt)
        if thermal is not None:  # seats already warm from earlier visitors
            thermal.scene.on_events(crowd.events)
        crowd.events = []

    n_frames = int(cfg["duration_s"] * fr)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tframes = tindex = None
    save_every = int(cfg.get("thermal", {}).get("save_every_frames", 0)) if thermal else 0
    if save_every > 0:
        n_save = (n_frames + save_every - 1) // save_every
        tframes = np.lib.format.open_memmap(out / "thermal_frames.npy", mode="w+", dtype=np.uint16,
                                            shape=(n_save, thermal.n, thermal.preset.height, thermal.preset.width))
        tindex = []

    pts, gts, tdets, tcaps = [], [], [], []
    t0 = time.time()
    for f in range(n_frames):
        crowd.step(dt)
        snap = crowd.snapshot()
        sc = sample_scatterers(snap, rng)
        for radar in radars:
            m = radar.measure(sc, snap, layout.obstacles, rng)
            if len(m):
                pts.append(np.column_stack([np.full(len(m), f), np.full(len(m), radar.sid), m]))
        if thermal is not None:
            cams, _, dets = thermal.step(crowd.t, dt, snap, crowd.events)
            if len(cams):  # every capture, including empty ones (they clear a camera's detections)
                tcaps.append(np.column_stack([np.full(len(cams), f), cams]))
            if len(dets):
                tdets.append(np.column_stack([np.full(len(dets), f), dets]))
            if tframes is not None and f % save_every == 0:
                k = f // save_every
                tframes[k] = np.round((np.nan_to_num(thermal.latest, nan=0.0) + 273.15) * 100).clip(0, 65535)
                tindex.append(f)
        crowd.events = []
        n = len(snap["pid"])
        if n:
            gts.append(np.column_stack([np.full(n, f), snap["pid"], snap["gid"], snap["pos"], snap["vel"],
                                        snap["browsing"], snap["height"], snap["pose"]]))
        if verbose and (f + 1) % max(1, n_frames // 10) == 0:
            print(f"  frame {f + 1}/{n_frames}  people={n}  elapsed={time.time() - t0:.0f}s", flush=True)

    points = np.concatenate(pts) if pts else np.zeros((0, 7))
    gt = np.concatenate(gts) if gts else np.zeros((0, 10))
    meta = {
        "config": cfg,
        "frame_rate_hz": fr,
        "n_frames": n_frames,
        "sensors_true": [p.as_dict() for p in poses],
        "sensors_calibrated": [p.as_dict() for p in calib],
    }
    if thermal is not None:
        meta["thermal"] = {
            "preset": asdict(thermal.preset),
            "focal_px": thermal.optics.f,
            "vfov_deg": thermal.optics.vfov_deg,
            "cameras_true": [p.as_dict() for p in thermal.poses],
            "cameras_calibrated": [p.as_dict() for p in thermal.calib],
            "frames_file": "thermal_frames.npy (uint16, centi-kelvin; [snapshot, camera, row, col])" if tframes is not None else None,
        }
        np.save(out / "thermal_dets.npy", (np.concatenate(tdets) if tdets else np.zeros((0, 6))).astype(np.float32))
        np.save(out / "thermal_captures.npy", (np.concatenate(tcaps) if tcaps else np.zeros((0, 2))).astype(np.int32))
        if tframes is not None:
            tframes.flush()
            np.save(out / "thermal_frames_index.npy", np.array(tindex, dtype=np.int32))
        thermal.close()
    save_run(out_dir, meta, points, gt)
    if verbose:
        msg = f"saved {len(points)} radar points from {len(radars)} radars"
        if thermal is not None:
            msg += f", {sum(len(d) for d in tdets)} thermal detections from {thermal.n} cameras"
        print(f"{msg}, {n_frames} frames -> {out_dir}")
