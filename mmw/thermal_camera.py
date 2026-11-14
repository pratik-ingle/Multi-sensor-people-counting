"""Thermal camera: lens geometry and microbolometer sensor model.

Presets follow public datasheets where available. Values marked 'est.' are engineering
estimates; replace them with measurements from your own units.

  lepton31r     FLIR Lepton 3.1R: 160x120, 95 deg wide lens with barrel distortion, <50 mK, 8.7 Hz
  lepton35      FLIR Lepton 3.5:  160x120, 57 deg lens, <50 mK, 8.7 Hz
  mlx90640_wide Melexis MLX90640 (110x75 deg): 32x24 thermopile, NETD 0.1 K @ 1 Hz (noise grows
                with refresh rate), no shutter, chess-pattern sub-page readout
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np
from scipy.ndimage import gaussian_filter

from .geometry import Pose


@dataclass
class ThermalPreset:
    name: str
    res: tuple = (160, 120)            # (width, height) pixels
    hfov_deg: float = 95.0
    lens: str = "stereographic"        # rectilinear | stereographic (mild barrel) | equidistant (strong barrel)
    fps: float = 8.7
    netd_k: float = 0.05               # temporal noise, 1 sigma
    psf_sigma_px: float = 0.5          # optics blur (est.)
    fpn_k: float = 0.05                # residual fixed-pattern noise right after a shutter event (est.)
    fpn_drift_k: float = 0.25          # fixed-pattern build-up until the next shutter event (est.)
    column_k: float = 0.03             # column noise (est.)
    offset_k: float = 1.0              # per-unit absolute radiometric error, 1 sigma (est.)
    gain_err: float = 0.01             # per-unit gain error (est.)
    vignette_k: float = 0.3            # edge offset from housing / narcissus (est.)
    ffc_period_s: float = 180.0        # flat-field correction (shutter) interval, 0 = no shutter
    ffc_duration_s: float = 0.5        # image freezes while the shutter is closed
    chess: bool = False                # MLX90640 chess-pattern: each frame updates half the pixels
    quant_k: float = 0.01              # TLinear output resolution

    @property
    def width(self) -> int:
        return int(self.res[0])

    @property
    def height(self) -> int:
        return int(self.res[1])


PRESETS = {
    "lepton31r": ThermalPreset("lepton31r"),
    "lepton35": ThermalPreset("lepton35", hfov_deg=57.0, lens="rectilinear"),
    "mlx90640_wide": ThermalPreset("mlx90640_wide", res=(32, 24), hfov_deg=110.0, lens="equidistant", fps=8.0,
                                   netd_k=0.28, psf_sigma_px=0.4, fpn_k=0.15, fpn_drift_k=0.1, column_k=0.0,
                                   offset_k=1.0, gain_err=0.02, vignette_k=0.6, ffc_period_s=0.0,
                                   ffc_duration_s=0.0, chess=True),
}


def get_preset(name: str, overrides: dict | None = None) -> ThermalPreset:
    base = PRESETS[name]
    if not overrides:
        return base
    known = {f.name for f in fields(ThermalPreset)}
    kw = {f.name: getattr(base, f.name) for f in fields(ThermalPreset)}
    kw.update({k: v for k, v in overrides.items() if k in known})
    return ThermalPreset(**kw)


_LENS = {
    "rectilinear": (np.tan, np.arctan),
    "stereographic": (lambda t: 2 * np.tan(t / 2), lambda r: 2 * np.arctan(r / 2)),
    "equidistant": (lambda t: t, lambda r: r),
}


class Optics:
    """Radially symmetric lens. Camera coords: x right, y up, z forward (optical axis).
    Pixel coords are continuous, x from the left edge, y_up from the bottom edge."""

    def __init__(self, preset: ThermalPreset):
        self.p = preset
        self.W, self.H = preset.width, preset.height
        self.g, self.ginv = _LENS[preset.lens]
        self.f = (self.W / 2) / self.g(np.radians(preset.hfov_deg / 2))

    def rays(self, x, y_up) -> np.ndarray:
        u = np.asarray(x, dtype=float) - self.W / 2
        v = np.asarray(y_up, dtype=float) - self.H / 2
        th = self.ginv(np.hypot(u, v) / self.f)
        ph = np.arctan2(v, u)
        return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=-1)

    def project(self, d: np.ndarray):
        """Camera-coordinate directions -> (x, y_up) pixel coords; NaN behind the camera."""
        n = np.linalg.norm(d, axis=-1)
        th = np.arccos(np.clip(d[..., 2] / np.maximum(n, 1e-12), -1, 1))
        ph = np.arctan2(d[..., 1], d[..., 0])
        rd = self.f * self.g(th)
        x = self.W / 2 + rd * np.cos(ph)
        y = self.H / 2 + rd * np.sin(ph)
        bad = th >= np.radians(89.0)
        return np.where(bad, np.nan, x), np.where(bad, np.nan, y)

    @property
    def vfov_deg(self) -> float:
        return float(2 * np.degrees(self.ginv((self.H / 2) / self.f)))


# Camera frame (x right, y up, z forward) vs the Pose sensor frame (x right, y boresight, z up).
def world_to_cam(pose: Pose, P: np.ndarray) -> np.ndarray:
    s = pose.to_sensor(P)
    return np.stack([s[..., 0], s[..., 2], s[..., 1]], axis=-1)


def cam_to_world_dir(pose: Pose, d: np.ndarray) -> np.ndarray:
    s = np.stack([d[..., 0], d[..., 2], d[..., 1]], axis=-1)
    return s @ pose.R.T


def view_matrix(pose: Pose) -> np.ndarray:
    """World -> OpenGL eye space (x right, y up, looking down -z)."""
    M = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])  # sensor -> GL eye
    A = M @ pose.R.T
    V = np.eye(4)
    V[:3, :3] = A
    V[:3, 3] = -A @ pose.position
    return V


def floor_plane_xy(optics: Optics, pose: Pose, z_plane: float):
    """World (x, y) where each pixel centre's ray meets the horizontal plane z = z_plane.
    Rows are top-down (row 0 = top of image). NaN where the ray never reaches the plane."""
    W, H = optics.W, optics.H
    xs = np.arange(W) + 0.5
    ys_up = (H - 1 - np.arange(H)) + 0.5
    X, Y = np.meshgrid(xs, ys_up)
    d = cam_to_world_dir(pose, optics.rays(X, Y))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (z_plane - pose.position[2]) / d[..., 2]
    t = np.where(t > 0, t, np.nan)
    return pose.position[0] + t * d[..., 0], pose.position[1] + t * d[..., 1]


@dataclass
class SensorArray:
    """Noise, non-uniformity, shutter and readout behaviour of N identical cameras."""
    preset: ThermalPreset
    n: int
    rng: np.random.Generator
    last: np.ndarray = field(init=False)

    def __post_init__(self):
        p, rng = self.preset, self.rng
        H, W = p.height, p.width
        self.fpn_resid = rng.normal(0, p.fpn_k, (self.n, H, W))
        drift = gaussian_filter(rng.normal(0, 1, (self.n, H, W)), sigma=(0, H / 10, W / 10))
        drift /= drift.std(axis=(1, 2), keepdims=True) + 1e-9
        self.fpn_drift = p.fpn_drift_k * drift + rng.normal(0, p.fpn_drift_k * 0.3, (self.n, H, W))
        self.columns = rng.normal(0, p.column_k, (self.n, 1, W))
        self.offset = rng.normal(0, p.offset_k, (self.n, 1, 1))
        self.gain = 1.0 + rng.normal(0, p.gain_err, (self.n, 1, 1))
        yy, xx = np.mgrid[0:H, 0:W]
        r2 = ((xx + 0.5 - W / 2) / (W / 2)) ** 2 + ((yy + 0.5 - H / 2) / (H / 2)) ** 2
        self.vignette = p.vignette_k * r2 / 2.0
        self.checker = ((xx + yy) % 2).astype(bool)
        self.subpage = np.zeros(self.n, dtype=int)
        self.ffc_phase = rng.uniform(0, max(p.ffc_period_s, 1e-6), self.n)
        self.last = np.full((self.n, H, W), np.nan, dtype=np.float32)

    def reference(self, t_true: np.ndarray, t: float) -> np.ndarray:
        """Noise-free reading of a static scene (an averaged empty-room capture) for all cameras."""
        p = self.preset
        drift = ((t + self.ffc_phase) % p.ffc_period_s) / p.ffc_period_s if p.ffc_period_s > 0 else np.ones(self.n)
        return (t_true * self.gain + self.offset + self.fpn_resid + drift[:, None, None] * self.fpn_drift
                + self.columns + self.vignette).astype(np.float32)

    def in_ffc(self, idx, t: float) -> np.ndarray:
        p = self.preset
        if p.ffc_period_s <= 0:
            return np.zeros(len(idx), dtype=bool)
        return ((t + self.ffc_phase[idx]) % p.ffc_period_s) < p.ffc_duration_s

    def measure(self, t_true: np.ndarray, idx: np.ndarray, t: float) -> np.ndarray:
        """t_true: (k, H, W) scene temperature after optics -> (k, H, W) camera output in degC."""
        p, rng = self.preset, self.rng
        k = len(idx)
        drift = ((t + self.ffc_phase[idx]) % p.ffc_period_s) / p.ffc_period_s if p.ffc_period_s > 0 else np.full(k, 1.0)
        out = (t_true * self.gain[idx] + self.offset[idx] + self.fpn_resid[idx]
               + drift[:, None, None] * self.fpn_drift[idx] + self.columns[idx] + self.vignette
               + rng.normal(0, p.netd_k, t_true.shape) + rng.normal(0, p.column_k * 0.5, (k, 1, p.width)))
        out = np.round(out / p.quant_k) * p.quant_k
        prev = self.last[idx]
        have_prev = ~np.isnan(prev[:, 0, 0])
        freeze = self.in_ffc(idx, t) & have_prev
        out[freeze] = prev[freeze]
        if p.chess:  # only one checkerboard sub-page is refreshed per frame
            mask = self.checker[None] == (self.subpage[idx] % 2).astype(bool)[:, None, None]
            out = np.where(mask | ~have_prev[:, None, None], out, prev)
            self.subpage[idx] += 1
        self.last[idx] = out
        return out.astype(np.float32)
