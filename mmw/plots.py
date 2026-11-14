"""Figures: floor plan with sensor coverage, a live snapshot, counts over time, engagement heatmaps."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

from .geometry import Pose, cart_to_sph  # noqa: E402
from .layout import Layout  # noqa: E402

ZONE_COLORS = ["#2a9d8f", "#e76f51", "#8338ec", "#f4a261", "#3a86ff", "#6c757d", "#d62828", "#52b788"]
TRUE_COLOR = "#1d3557"
PIPE_COLORS = {"radar": "#d62828", "thermal": "#f77f00", "fused": "#2a9d8f"}


def _floor(ax, layout: Layout, labels=True):
    ax.add_patch(Rectangle((0, 0), layout.width, layout.depth, fill=False, lw=1.5, ec="k"))
    for o in layout.obstacles:
        ax.add_patch(Rectangle((o.x0, o.y0), o.x1 - o.x0, o.y1 - o.y0, fc="#9aa0a6" if o.reflective else "#cfd4da",
                               ec="#5f6368", lw=0.8, hatch="//" if o.reflective else None, zorder=2))
    for k, z in enumerate(layout.zones):
        b = z.box
        c = ZONE_COLORS[k % len(ZONE_COLORS)]
        ax.add_patch(Rectangle((b.x0, b.y0), b.x1 - b.x0, b.y1 - b.y0, fill=False, ec=c, lw=1.6, ls="--", zorder=3))
        if labels:
            ax.text(b.x0 + 0.15, b.y1 - 0.15, z.name, color=c, fontsize=8, va="top", zorder=4, weight="bold")
    for b, name in ((layout.entrance, "IN"), (layout.exit, "OUT")):
        ax.add_patch(Rectangle((b.x0, b.y0), b.x1 - b.x0, b.y1 - b.y0, fc="#ffd166", ec="none", alpha=0.7, zorder=1))
        ax.text((b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2, name, ha="center", va="center", fontsize=7, zorder=4)
    ax.set_xlim(-0.5, layout.width + 0.5)
    ax.set_ylim(-0.5, layout.depth + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def coverage_map(layout: Layout, poses: list[Pose], params, z=1.2, cell=0.25):
    xs = np.arange(cell / 2, layout.width, cell)
    ys = np.arange(cell / 2, layout.depth, cell)
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z)])
    cov = np.zeros(X.size, dtype=int)
    for pose in poses:
        r, az, el = cart_to_sph(pose.to_sensor(pts))
        cov += ((r < params.max_range) & (np.abs(az) <= np.radians(params.fov_az_deg))
                & (np.abs(el) <= np.radians(params.fov_el_deg))).astype(int)
    return xs, ys, cov.reshape(X.shape)


def thermal_footprint(optics, pose: Pose, z: float) -> np.ndarray:
    """Outline (K, 2) of the floor area a thermal camera sees at height z."""
    from .thermal_detect import plane_map

    mx, my = plane_map(optics, pose, z, 1)
    edge = np.concatenate([np.column_stack([mx[0], my[0]]), np.column_stack([mx[:, -1], my[:, -1]]),
                           np.column_stack([mx[-1, ::-1], my[-1, ::-1]]), np.column_stack([mx[::-1, 0], my[::-1, 0]])])
    return edge[np.isfinite(edge).all(1)]


def plot_layout(path, layout, poses, params, thermal_poses=None, thermal_optics=None, thermal_z=1.4):
    xs, ys, cov = coverage_map(layout, poses, params)
    fig, ax = plt.subplots(figsize=(11, 7.5))
    im = ax.imshow(cov, origin="lower", extent=[0, layout.width, 0, layout.depth], cmap="Blues",
                   vmin=0, vmax=max(cov.max(), 1), alpha=0.8, zorder=0)
    _floor(ax, layout)
    sp = np.array([p.position for p in poses])
    ax.scatter(sp[:, 0], sp[:, 1], marker="^", s=60, c="#d62828", ec="k", zorder=5, label=f"{len(poses)} radars")
    if thermal_poses:
        for p in thermal_poses:
            ax.add_patch(Polygon(thermal_footprint(thermal_optics, p, thermal_z), closed=True, fill=False,
                                 ec="#f77f00", lw=0.9, zorder=4))
        tp = np.array([p.position for p in thermal_poses])
        ax.scatter(tp[:, 0], tp[:, 1], marker="s", s=28, c="#f77f00", ec="k", zorder=5,
                   label=f"{len(thermal_poses)} thermal cameras (outline = view at {thermal_z} m)")
    fig.colorbar(im, ax=ax, shrink=0.7, label="radars covering a point at 1.2 m height")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Showroom layout and sensor coverage, {layout.width:.0f} x {layout.depth:.0f} m")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_snapshot(path, layout, xyz, gt_xy, tracks: dict, t_s, tdets=None):
    fig, ax = plt.subplots(figsize=(11, 7.5))
    _floor(ax, layout, labels=True)
    ax.scatter(xyz[:, 0], xyz[:, 1], s=3, c="#adb5bd", zorder=3, label=f"radar points ({len(xyz)})")
    if tdets is not None and len(tdets):
        ax.scatter(tdets[:, 1], tdets[:, 2], s=30, marker="+", c="#f77f00", lw=1.2, zorder=4,
                   label=f"thermal detections ({len(tdets)})")
    ax.scatter(gt_xy[:, 0], gt_xy[:, 1], s=60, facecolors="none", edgecolors=TRUE_COLOR, lw=1.4, zorder=5,
               label=f"true people ({len(gt_xy)})")
    for name, xy in tracks.items():
        ax.scatter(xy[:, 0], xy[:, 1], s=40, marker="x", c=PIPE_COLORS.get(name, "k"), lw=1.6, zorder=6,
                   label=f"{name} tracks ({len(xy)})")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Sensor data and tracks at t = {t_s:.1f} s")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_counts(path, t, gt_total, gt_zone, est: dict, layout):
    """est: name -> (total (F,), per_zone (F, Z))."""
    nz = len(layout.zones)
    rows = (nz + 2) // 3
    fig, axes = plt.subplots(1 + rows, 3, figsize=(14, 3 + 2.2 * rows), squeeze=False)
    gs = axes[0, 0].get_gridspec()
    for a in axes[0]:
        a.remove()
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(t, gt_total, c=TRUE_COLOR, lw=1.8, label="true")
    for name, (tot, _) in est.items():
        ax0.plot(t, tot, c=PIPE_COLORS.get(name, "k"), lw=1.0, label=name)
    ax0.set_title("People in the store")
    ax0.set_ylabel("count")
    ax0.legend(fontsize=8, ncol=4)
    for k, z in enumerate(layout.zones):
        a = axes[1 + k // 3, k % 3]
        a.plot(t, gt_zone[:, k], c=TRUE_COLOR, lw=1.4)
        for name, (_, zc) in est.items():
            a.plot(t, zc[:, k], c=PIPE_COLORS.get(name, "k"), lw=0.8)
        a.set_title(z.name, fontsize=9)
    for k in range(nz, 3 * rows):
        axes[1 + k // 3, k % 3].axis("off")
    for a in axes[-1]:
        a.set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_heatmaps(path, layout, H_true, H_est: dict):
    panels = [("true", H_true)] + list(H_est.items())
    vmax = np.percentile(np.concatenate([h.ravel() for _, h in panels]), 99.5)
    cols = 2
    rows = (len(panels) + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(16, 5.6 * rows), squeeze=False)
    for ax, (name, H) in zip(axes.ravel(), panels):
        im = ax.imshow(H, origin="lower", extent=[0, layout.width, 0, layout.depth], cmap="inferno",
                       vmin=0, vmax=vmax, zorder=0)
        _floor(ax, layout)
        ax.set_title(f"{name}: dwell (person-seconds per m²)")
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_zone_share(path, rows_by: dict):
    """rows_by: pipeline name -> zone_summary rows (each has share_true, share_est)."""
    first = next(iter(rows_by.values()))
    names = [r["zone"] for r in first]
    x = np.arange(len(names))
    n = 1 + len(rows_by)
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - 0.4 + w / 2, [100 * r["share_true"] for r in first], w, label="true", color=TRUE_COLOR)
    for k, (name, rows) in enumerate(rows_by.items(), start=1):
        ax.bar(x - 0.4 + w / 2 + k * w, [100 * r["share_est"] for r in rows], w, label=name,
               color=PIPE_COLORS.get(name, "k"))
    ax.set_xticks(x, names)
    ax.set_ylabel("% of in-zone person-time")
    ax.set_title("Where shoppers spend their time")
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_thermal_frames(path, frames: np.ndarray, cams: list[int], titles: list[str], vmin=20.0, vmax=36.0):
    """Raw camera outputs (degC) side by side."""
    n = len(cams)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4.1 * rows), squeeze=False)
    for ax, c, title in zip(axes.ravel(), cams, titles):
        im = ax.imshow(frames[c], cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.8, label="°C")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
