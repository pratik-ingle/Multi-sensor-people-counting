"""Frame-by-frame simulation + fusion + tracking for the live viewer (radar, thermal, fused)."""
from __future__ import annotations

import base64
import io
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from .body import sample_scatterers
from .crowd import Crowd
from .fusion import plausible
from .geometry import sph_to_cart
from .layout import Box, Layout
from .plots import coverage_map, thermal_footprint
from .radar import Radar, RadarParams
from .simulate import build_poses, perturb
from .tracker import make_tracker

THERMAL_POINTS_PER_DET = 3
THERMAL_SNR_DB = 20.0
T_MIN, T_MAX = 20.0, 36.0  # display range of thermal images

# "Ironbow"-style palette used by most thermal cameras.
_STOPS = np.array([[0.0, 0, 0, 0], [0.15, 30, 0, 90], [0.35, 140, 0, 150], [0.55, 220, 50, 60],
                   [0.75, 250, 140, 0], [0.9, 255, 215, 40], [1.0, 255, 255, 230]])
_LUT = np.stack([np.interp(np.linspace(0, 1, 256), _STOPS[:, 0], _STOPS[:, k]) for k in (1, 2, 3)], 1).astype(np.uint8)


def colorize(T: np.ndarray, alpha: np.ndarray | None = None) -> np.ndarray:
    idx = np.clip((np.nan_to_num(T, nan=T_MIN) - T_MIN) / (T_MAX - T_MIN) * 255, 0, 255).astype(np.uint8)
    rgb = _LUT[idx]
    if alpha is None:
        return rgb
    return np.concatenate([rgb, (alpha * 255).astype(np.uint8)[..., None]], axis=-1)


def png_data_url(img: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG", compress_level=1)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def png_bytes(img: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def _box(b: Box) -> dict:
    return {"name": b.name, "x0": b.x0, "y0": b.y0, "x1": b.x1, "y1": b.y1}


def _r(a, nd=2):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


def _match(G: np.ndarray, T: np.ndarray, gate: float = 0.75):
    found = np.zeros(len(G), dtype=bool)
    real = np.zeros(len(T), dtype=bool)
    if len(G) and len(T):
        d = np.hypot(G[:, None, 0] - T[None, :, 0], G[:, None, 1] - T[None, :, 1])
        r, c = linear_sum_assignment(np.where(d < gate, d, 1e6))
        ok = d[r, c] < gate
        found[r[ok]] = True
        real[c[ok]] = True
    return found, real


class LiveSim:
    def __init__(self, cfg: dict, seed: int | None = None, warmup_s: float | None = None):
        self.rng = np.random.default_rng(cfg.get("seed", 0) if seed is None else seed)
        self.layout = Layout(cfg["layout"])
        self.crowd = Crowd(self.layout, cfg["crowd"], self.rng)
        self.params = RadarParams.from_dict(cfg.get("radar", {}))  # shared by all radars: live-tunable
        self.poses = build_poses(cfg["sensors"])
        self.calib = perturb(self.poses, cfg["sensors"].get("calibration_error", {}), self.rng)
        self.radars = [Radar(i, p, self.params) for i, p in enumerate(self.poses)]
        self.dt = 1.0 / float(cfg.get("frame_rate_hz", 10))
        self.thermal = None
        if cfg.get("thermal", {}).get("enabled", False):
            from .thermal_system import ThermalSystem
            self.thermal = ThermalSystem(cfg, self.crowd.seats, self.rng)
        warm = cfg.get("warmup_s", 0) if warmup_s is None else warmup_s
        for _ in range(int(warm / self.dt)):
            self.crowd.step(self.dt)
            if self.thermal is not None:
                self.thermal.scene.on_events(self.crowd.events)
            self.crowd.events = []
        lay = self.layout
        doors = tuple((b.x0 - 1.0, b.y0 - 1.0, b.x1 + 1.0, b.y1 + 1.5) for b in (lay.entrance, lay.exit))
        names = ["radar"] + (["thermal", "fused"] if self.thermal is not None else [])
        self.trackers = {n: make_tracker(n, self.dt, no_static_zones=doors) for n in names}
        self.frame = 0
        self.want_mosaic = True
        self._mosaic = self._mosaic_map() if self.thermal is not None else None

    # ---------------------------------------------------------------- thermal floor mosaic
    def _mosaic_map(self, cell: float = 0.12, z: float = 1.2):
        """For every floor cell: which camera sees it most directly, and through which pixel."""
        from .thermal_camera import world_to_cam

        th = self.thermal
        lay = self.layout
        xs = np.arange(cell / 2, lay.width, cell)
        ys = np.arange(cell / 2, lay.depth, cell)
        X, Y = np.meshgrid(xs, ys)
        P = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z)])
        W, H = th.optics.W, th.optics.H
        best = np.full(X.size, np.inf)
        cam = np.zeros(X.size, dtype=int)
        pix = np.zeros(X.size, dtype=int)
        for c, pose in enumerate(th.calib):
            px, py = th.optics.project(world_to_cam(pose, P))
            inside = np.isfinite(px) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
            d = np.hypot(P[:, 0] - pose.position[0], P[:, 1] - pose.position[1])
            better = inside & (d < best)
            best[better] = d[better]
            cam[better] = c
            col = np.clip(np.floor(np.nan_to_num(px)), 0, W - 1).astype(int)
            row = np.clip(H - 1 - np.floor(np.nan_to_num(py)), 0, H - 1).astype(int)
            pix[better] = (row * W + col)[better]
        return {"shape": X.shape, "cam": cam, "pix": pix, "valid": np.isfinite(best)}

    def mosaic_png(self) -> str:
        return png_data_url(self.mosaic_rgba())

    def mosaic_rgba(self) -> np.ndarray:
        """Floor-wide thermal image, RGBA, row 0 = far wall (max y), like the floor plan."""
        m = self._mosaic
        th = self.thermal
        flat = th.latest.reshape(th.n, -1)
        offset = np.zeros(th.n)
        if th.detector.bg is not None:  # cross-calibrate tiles: each unit has its own absolute error
            offset = np.median(th.detector.bg.reshape(th.n, -1), axis=1)
            offset -= np.median(offset)
        T = (flat[m["cam"], m["pix"]] - offset[m["cam"]]).reshape(m["shape"])
        alpha = (m["valid"].reshape(m["shape"]) & np.isfinite(T)).astype(float) * 0.92
        return colorize(T, alpha)[::-1]

    def camera_png(self, k: int, scale: int = 3) -> bytes:
        T = self.thermal.latest[k].copy()
        return png_bytes(np.repeat(np.repeat(colorize(T), scale, axis=0), scale, axis=1))

    # ---------------------------------------------------------------- layout for the page
    def layout_json(self) -> dict:
        lay = self.layout
        _, _, cov = coverage_map(lay, self.poses, self.params)
        out = {
            "width": lay.width, "depth": lay.depth, "dt": self.dt,
            "obstacles": [{**_box(o), "height": o.height, "reflective": o.reflective} for o in lay.obstacles],
            "zones": [_box(z.box) for z in lay.zones],
            "entrance": _box(lay.entrance), "exit": _box(lay.exit),
            "sensors": [p.position.tolist() for p in self.poses],
            "coverage": {"cell": 0.25, "nx": int(cov.shape[1]), "ny": int(cov.shape[0]),
                         "data": cov.ravel().astype(int).tolist()},
            "pipelines": list(self.trackers),
            "thermal": None,
        }
        if self.thermal is not None:
            th = self.thermal
            out["thermal"] = {
                "camera": th.preset.name, "res": [th.optics.W, th.optics.H], "hfov": th.preset.hfov_deg,
                "fps": th.preset.fps, "netd_k": th.preset.netd_k,
                "positions": [p.position.tolist() for p in th.poses],
                "footprints": [_r(thermal_footprint(th.optics, p, 1.4)[::8]) for p in th.poses],
                "t_range": [T_MIN, T_MAX],
                "palette": _LUT[::16].tolist(),
            }
        return out

    # ---------------------------------------------------------------- one frame
    def step(self) -> dict:
        t0 = time.perf_counter()
        lay = self.layout
        self.crowd.step(self.dt)
        snap = self.crowd.snapshot()
        sc = sample_scatterers(snap, self.rng)

        xyz, vel, sid, snr = [], [], [], []
        for radar in self.radars:
            m = radar.measure(sc, snap, lay.obstacles, self.rng)
            if len(m):
                ps = sph_to_cart(m[:, 0].astype(float), m[:, 1].astype(float), m[:, 2].astype(float))
                xyz.append(self.calib[radar.sid].to_world(ps))  # pipeline only knows calibrated poses
                vel.append(m[:, 3])
                snr.append(m[:, 4])
                sid.append(np.full(len(m), radar.sid))
        if xyz:
            xyz, vel, snr, sid = (np.concatenate(a) for a in (xyz, vel, snr, sid))
        else:
            xyz, vel, snr, sid = np.zeros((0, 3)), np.zeros(0), np.zeros(0), np.zeros(0, dtype=int)
        keep = plausible(xyz, lay)
        r_xy, r_snr = xyz[keep, :2], snr[keep].astype(float)

        merged = np.zeros((0, 5))
        n_captured = 0
        if self.thermal is not None:
            cams, _, _ = self.thermal.step(self.crowd.t, self.dt, snap, self.crowd.events)
            n_captured = len(cams)
            merged = self.thermal.merged_detections()
        self.crowd.events = []
        t_xy = np.repeat(merged[:, 1:3], THERMAL_POINTS_PER_DET, axis=0)
        t_snr = np.full(len(t_xy), THERMAL_SNR_DB)
        inputs = {"radar": (r_xy, r_snr), "thermal": (t_xy, t_snr),
                  "fused": (np.concatenate([r_xy, t_xy]), np.concatenate([r_snr, t_snr]))}

        G = snap["pos"]
        nz = len(lay.zones)
        zt = np.bincount(lay.zone_index(G[:, 0], G[:, 1]) + 1, minlength=nz + 1)[1:] if len(G) else np.zeros(nz)
        pipes = {}
        for name, tracker in self.trackers.items():
            tracks = tracker.step(*inputs[name])
            T = np.array([t.x for t in tracks]).reshape(-1, 4)
            found, real = _match(G, T)
            ze = np.bincount(lay.zone_index(T[:, 0], T[:, 1]) + 1, minlength=nz + 1)[1:] if len(T) else np.zeros(nz)
            pipes[name] = {"est": int(len(T)), "ze": [int(v) for v in ze], "d": found.astype(int).tolist(),
                           "tr": {"id": [t.tid for t in tracks], "x": _r(T[:, 0]), "y": _r(T[:, 1]),
                                  "vx": _r(T[:, 2]), "vy": _r(T[:, 3]), "ok": real.astype(int).tolist()}}

        self.frame += 1
        out = {
            "f": self.frame, "t": round(self.frame * self.dt, 2),
            "true": int(len(G)), "zt": [int(v) for v in zt], "pipes": pipes,
            "p": {"x": _r(xyz[:, 0]), "y": _r(xyz[:, 1]), "z": _r(xyz[:, 2]), "v": _r(vel),
                  "s": sid.astype(int).tolist(), "k": keep.astype(int).tolist()},
            "g": {"x": _r(G[:, 0]), "y": _r(G[:, 1]), "r": _r(snap["radius"]),
                  "b": snap["browsing"].astype(int).tolist(), "st": snap["pose"].astype(int).tolist()},
            "td": {"x": _r(merged[:, 1]), "y": _r(merged[:, 2])},
            "cams": n_captured,
        }
        if self.thermal is not None and self.want_mosaic and self.frame % 2 == 0:
            out["mosaic"] = self.mosaic_png()
        out["ms"] = round(1000 * (time.perf_counter() - t0), 1)
        return out

    def close(self):
        if self.thermal is not None:
            self.thermal.close()
