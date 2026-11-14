"""Thermal camera network: scene + free-running cameras + renderer + sensor model + edge detection."""
from __future__ import annotations

import numpy as np

from .radiometry import radiance, temperature
from .thermal_camera import Optics, SensorArray, get_preset
from .thermal_detect import DetectParams, ThermalDetector, merge_detections
from .thermal_scene import Environment, ThermalScene


class ThermalSystem:
    def __init__(self, cfg: dict, seats: list[dict], rng: np.random.Generator):
        from .simulate import build_poses, perturb  # local import: simulate imports this module

        tc = cfg["thermal"]
        self.cfg = tc
        self.preset = get_preset(tc.get("camera", "lepton31r"), tc.get("overrides"))
        self.optics = Optics(self.preset)
        self.poses = build_poses(tc)
        self.calib = perturb(self.poses, tc.get("calibration_error", {}), rng)
        self.n = len(self.poses)
        self.cam_xy = np.array([p.position[:2] for p in self.calib])
        self.env = Environment.from_dict(cfg.get("environment"))
        self.scene = ThermalScene(cfg["layout"], self.env, seats, rng)
        self.sensor = SensorArray(self.preset, self.n, rng)
        self.params = DetectParams.from_dict(tc.get("detection"))
        self.detector = ThermalDetector(self.optics, self.calib, self.params, self.preset.fps)
        self.phase = rng.uniform(0, 1, self.n)  # cameras free-run, not synchronised to each other
        self.renderer = None
        H, W = self.preset.height, self.preset.width
        self.latest = np.full((self.n, H, W), np.nan, dtype=np.float32)
        self.latest_dets = [np.zeros((0, 5)) for _ in range(self.n)]
        self._static_t = -1e9

    def _ensure_gpu(self, t: float):
        """Create GL resources lazily, in the thread that renders (GL contexts are per thread)."""
        if self.renderer is not None:
            return
        from .thermal_render import ThermalRenderer

        sc = self.scene
        self.renderer = ThermalRenderer(self.optics, self.poses, sc.V, sc.F, sc.human.n_verts, sc.human.F)
        self.renderer.set_static_radiance(sc.static_radiance(t))
        self._static_t = t
        # Empty-store reference ("night calibration"): the starting background for every camera.
        rad = self.renderer.render(range(self.n), background_rad=float(radiance(self.env.air(t))))
        self.detector.set_background(self.sensor.reference(temperature(rad), t))

    def capturing(self, t: float, dt: float) -> np.ndarray:
        fps = self.preset.fps
        now = np.floor((t + self.phase / fps) * fps)
        before = np.floor((t - dt + self.phase / fps) * fps)
        return np.nonzero(now > before)[0]

    def step(self, t: float, dt: float, snap: dict, events: list):
        """Advance one radar frame. Returns (cameras that captured, their frames degC, raw detections)."""
        self.scene.on_events(events)
        idx = self.capturing(t, dt)
        if len(idx) == 0:
            return idx, None, np.zeros((0, 5))
        self._ensure_gpu(t)
        if t - self._static_t > 1.0:  # room temperature drifts slowly
            self.renderer.set_static_radiance(self.scene.static_radiance(t))
            self._static_t = t
        pv, prad, pid = self.scene.people(snap, t)
        iv, irad = self.scene.imprint_quads(t)
        rad = self.renderer.render(idx, pv, prad, pid, iv, irad, background_rad=float(radiance(self.env.air(t))))
        frames = self.sensor.measure(temperature(rad), idx, t)
        self.latest[idx] = frames
        dets = self.detector.update(frames, idx)
        for c in idx:
            self.latest_dets[c] = dets[dets[:, 0] == c]
        return idx, frames, dets

    def merged_detections(self) -> np.ndarray:
        """Newest detections of every camera, de-duplicated across overlapping views."""
        return merge_detections(np.concatenate(self.latest_dets), self.cam_xy)

    def close(self):
        if self.renderer is not None:
            self.renderer.release()
            self.renderer = None
