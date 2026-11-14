"""People detection on thermal frames, per camera (what would run on each camera's edge processor),
plus merging of detections from overlapping cameras."""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
from scipy.ndimage import gaussian_filter, label, maximum_filter

from .geometry import Pose
from .thermal_camera import Optics, cam_to_world_dir

DET_COLS = ["cam", "x", "y", "strength", "nadir_dist"]


@dataclass
class DetectParams:
    z_plane_m: float = 1.4       # detections are placed at this height (upper body)
    fg_k: float = 1.0            # pixel counts as 'warm' if this much above its background
    peak_k: float = 2.0          # a person candidate peaks at least this much above background
    peak_window_m: float = 0.40  # non-maximum suppression window at the detection plane
    min_area_m2: float = 0.04    # warm blob must be at least this big (rejects kettles, phones, noise)
    bin: int = 2                 # detect on binned frames (Lepton 160x120 -> 80x60)
    bg_alpha: float = 0.003      # background learning rate per frame (~40 s at 8.7 Hz)
    cold_alpha: float = 0.05     # faster where the scene got colder (a warm object left)
    absorb_s: float = 300.0      # warm for longer than this -> slowly absorbed into the background
    absorb_alpha: float = 0.02

    @classmethod
    def from_dict(cls, d: dict | None) -> "DetectParams":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


def bin_frames(F: np.ndarray, b: int) -> np.ndarray:
    if b == 1:
        return F
    k, H, W = F.shape
    return F[:, :H // b * b, :W // b * b].reshape(k, H // b, b, W // b, b).mean(axis=(2, 4))


def plane_map(optics: Optics, pose: Pose, z: float, b: int = 1):
    """World (x, y) where each (binned) pixel's ray meets the plane z. Rows top-down."""
    H, W = optics.H // b, optics.W // b
    xs = (np.arange(W) + 0.5) * b
    ys_up = optics.H - (np.arange(H) + 0.5) * b
    X, Y = np.meshgrid(xs, ys_up)
    d = cam_to_world_dir(pose, optics.rays(X, Y))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (z - pose.position[2]) / d[..., 2]
    t = np.where(t > 0, t, np.nan)
    return pose.position[0] + t * d[..., 0], pose.position[1] + t * d[..., 1]


class ThermalDetector:
    def __init__(self, optics: Optics, poses: list[Pose], params: DetectParams, fps: float):
        self.p = p = params
        b = p.bin
        self.n = len(poses)
        maps = [plane_map(optics, pose, p.z_plane_m, b) for pose in poses]
        self.mx = np.stack([m[0] for m in maps])
        self.my = np.stack([m[1] for m in maps])
        cam_xy = np.array([pose.position[:2] for pose in poses])
        self.nadir = np.hypot(self.mx - cam_xy[:, 0, None, None], self.my - cam_xy[:, 1, None, None])
        drop = float(np.mean([pose.position[2] for pose in poses])) - p.z_plane_m
        px_m = drop / optics.f * b                         # pixel footprint at the image centre
        self.win = max(3, int(round(p.peak_window_m / px_m)) | 1)
        self.sigma = max(0.6, self.win / 6.0)
        self.min_px = max(2, int(p.min_area_m2 / px_m ** 2))
        self.dt = 1.0 / fps
        self.bg = None
        self.fg_age = np.zeros(self.mx.shape, dtype=np.float32)
        self.struct = np.zeros((3, 3, 3), dtype=bool)
        self.struct[1] = True  # connect pixels within a camera only

    def set_background(self, reference: np.ndarray):
        self.bg = bin_frames(reference, self.p.bin).astype(np.float32)

    def update(self, frames: np.ndarray, idx) -> np.ndarray:
        """frames: (k, H, W) degC from cameras idx -> detections (m, 5) = cam, x, y, strength, nadir_dist."""
        p = self.p
        idx = np.asarray(idx)
        F = bin_frames(frames, p.bin)
        bg = self.bg[idx]
        D = F - bg
        warm = D > p.fg_k
        age = np.where(warm, self.fg_age[idx] + self.dt, 0.0)
        self.fg_age[idx] = age
        alpha = np.where(warm, np.where(age > p.absorb_s, p.absorb_alpha, 0.0),
                         np.where(D < -p.fg_k, p.cold_alpha, p.bg_alpha))
        self.bg[idx] = bg + alpha * (F - bg)

        Ds = gaussian_filter(np.clip(D, 0, None), sigma=(0, self.sigma, self.sigma))
        peaks = (Ds >= maximum_filter(Ds, size=(1, self.win, self.win))) & (Ds > p.peak_k) & warm
        if not peaks.any():
            return np.zeros((0, 5))
        lab, _ = label(warm, structure=self.struct)
        area = np.bincount(lab.ravel())
        k, r, c = np.nonzero(peaks)
        ok = area[lab[k, r, c]] >= self.min_px
        k, r, c = k[ok], r[ok], c[ok]
        cams = idx[k]
        x, y = self.mx[cams, r, c], self.my[cams, r, c]
        good = np.isfinite(x)
        return np.column_stack([cams, x, y, Ds[k, r, c], self.nadir[cams, r, c]])[good]


def replay_merged(dets: np.ndarray, caps: np.ndarray, n_frames: int, n_cams: int, cam_xy: np.ndarray):
    """Rebuild, for every radar frame, the newest detections of each camera (from its most recent
    capture) merged across cameras. dets rows: frame, cam, x, y, strength, nadir; caps rows: frame, cam."""
    from .io import frame_slices

    ds = frame_slices(dets[:, 0], n_frames)
    cs = frame_slices(caps[:, 0], n_frames)
    empty = np.zeros((0, 5))
    latest = [empty] * n_cams
    for f in range(n_frames):
        for c in caps[cs[f]:cs[f + 1], 1]:
            latest[int(c)] = empty
        rows = dets[ds[f]:ds[f + 1], 1:].astype(float)
        for c in np.unique(rows[:, 0]).astype(int):
            latest[c] = rows[rows[:, 0] == c]
        yield merge_detections(np.concatenate(latest), cam_xy)


def merge_detections(dets: np.ndarray, cam_xy: np.ndarray, across_m: float = 0.45, along_m: float = 0.3,
                     along_frac: float = 0.3, same_cam_m: float = 0.4) -> np.ndarray:
    """One detection per person across overlapping cameras.

    A peak is projected to an assumed height, so its error lies along the camera's radial line
    (a head seen obliquely lands ~20 % too far out, legs too close in) and is small across it.
    Detections are accepted nearest-to-nadir first (most accurate); a later detection is a
    duplicate if it lies within that radial error band of an accepted one."""
    if len(dets) == 0:
        return dets
    order = np.lexsort((-dets[:, 3], dets[:, 4]))
    kept = np.zeros((0, 5))
    for i in order:
        d = dets[i]
        if len(kept):
            v = kept[:, 1:3] - d[1:3]
            same = kept[:, 0] == d[0]
            radial = d[1:3] - cam_xy[int(d[0])]
            rho = float(np.hypot(*radial))
            u = radial / rho if rho > 1e-6 else np.array([1.0, 0.0])
            along = v @ u
            across = np.abs(v[:, 0] * u[1] - v[:, 1] * u[0])
            dup = ~same & (across < across_m) & (np.abs(along) < along_m + along_frac * rho)
            dup |= same & (np.hypot(v[:, 0], v[:, 1]) < same_cam_m)
            if dup.any():
                continue
        kept = np.vstack([kept, d])
    return kept
