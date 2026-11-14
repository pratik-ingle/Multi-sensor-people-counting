"""Track people from radar, thermal cameras, and both fused; write per-frame positions/counts and
engagement analytics; score every pipeline against ground truth.

    .venv/bin/python scripts/run_baseline.py --run data/sim_run1

Outputs (in <run>/baseline/), for each pipeline <p> in radar / thermal / fused:
  tracks_<p>.csv        frame, time_s, track_id, x, y, vx, vy, zone   -- where every person is, each frame
  counts_<p>.csv        frame, time_s, total, <one column per zone>   -- how many, each frame
  zone_summary_<p>.csv  average occupancy and share of time per zone
  metrics.json          accuracy of every pipeline against simulation ground truth
  *.png                 coverage, snapshot, counts over time, heatmaps, zone shares, raw thermal frames
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mmw import analytics, plots  # noqa: E402
from mmw.fusion import plausible, sensor_poses, to_world  # noqa: E402
from mmw.geometry import Pose  # noqa: E402
from mmw.io import frame_slices, load_run  # noqa: E402
from mmw.layout import Layout  # noqa: E402
from mmw.radar import RadarParams  # noqa: E402
from mmw.tracker import make_tracker  # noqa: E402

THERMAL_POINTS_PER_DET = 3      # each merged thermal detection enters the tracker as 3 points ...
THERMAL_SNR_DB = 20.0           # ... weighted like a strong radar return


def run_tracker(name, n_frames, dt, inputs, doors):
    tracker = make_tracker(name, dt, no_static_zones=doors)
    rows, per_frame = [], []
    for f in range(n_frames):
        xy, snr = inputs(f)
        tracks = tracker.step(xy, snr)
        per_frame.append(np.array([t.x[:2] for t in tracks]).reshape(-1, 2))
        for tr in tracks:
            rows.append((f, tr.tid, tr.x[0], tr.x[1], tr.x[2], tr.x[3]))
    return np.array(rows, dtype=float).reshape(-1, 6), per_frame


def write_outputs(out, name, est, layout, n_frames, dt):
    names = [z.name for z in layout.zones]
    zone = layout.zone_index(est[:, 2], est[:, 3])
    with open(out / f"tracks_{name}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "time_s", "track_id", "x", "y", "vx", "vy", "zone"])
        for r, z in zip(est, zone):
            w.writerow([int(r[0]), f"{r[0] * dt:.2f}", int(r[1]), f"{r[2]:.2f}", f"{r[3]:.2f}",
                        f"{r[4]:.2f}", f"{r[5]:.2f}", names[z] if z >= 0 else ""])
    total, zc = analytics.counts(est[:, 0].astype(int), zone, n_frames, len(names))
    with open(out / f"counts_{name}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "time_s", "total"] + names)
        for f in range(n_frames):
            w.writerow([f, f"{f * dt:.2f}", int(total[f])] + [int(c) for c in zc[f]])
    return total, zc


def evaluate(est, total, zc, gt, gt_total, gt_zc, layout, n_frames, dt):
    gs = frame_slices(gt[:, 0], n_frames)
    es = frame_slices(est[:, 0], n_frames)
    m, detected = analytics.match_metrics(gs, gt[:, 3:5], es, est[:, 2:4], n_frames)
    zrows = analytics.zone_summary(gt_zc, zc, layout, dt, n_frames * dt)
    H_true = analytics.heatmap(gt[:, 3], gt[:, 4], layout, dt)
    H_est = analytics.heatmap(est[:, 2], est[:, 3], layout, dt)
    err = total.astype(float) - gt_total
    browsing = gt[:, 7] > 0.5
    seated = gt[:, 9] > 0.5 if gt.shape[1] > 9 else np.zeros(len(gt), dtype=bool)
    near = np.zeros(len(gt), dtype=bool)
    for f in range(n_frames):
        g = gt[gs[f]:gs[f + 1], 3:5]
        if len(g) > 1:
            d = np.hypot(g[:, None, 0] - g[None, :, 0], g[:, None, 1] - g[None, :, 1])
            np.fill_diagonal(d, np.inf)
            near[gs[f]:gs[f + 1]] = d.min(1) < 0.8
    metrics = {
        "avg_people_true": float(gt_total.mean()), "avg_people_est": float(total.mean()),
        "count_mae": float(np.abs(err).mean()), "count_bias": float(err.mean()),
        "count_mape_pct": float(100 * np.abs(err).mean() / max(gt_total.mean(), 1e-9)),
        "zone_count_mae": float(np.mean([r["count_mae"] for r in zrows])),
        "heatmap_correlation": float(np.corrcoef(H_true.ravel(), H_est.ravel())[0, 1]),
        **m,
        "recall_walking": float(detected[~browsing].mean()),
        "recall_browsing": float(detected[browsing].mean()),
        "recall_seated": float(detected[seated].mean()) if seated.any() else None,
        "recall_isolated": float(detected[~near].mean()),
        "recall_close_to_others": float(detected[near].mean()),
    }
    return metrics, zrows, H_true, H_est


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="data/sim_run1")
    ap.add_argument("--poses", default="calibrated", choices=["calibrated", "true"])
    args = ap.parse_args()
    run_dir = Path(args.run)
    out = run_dir / "baseline"
    out.mkdir(exist_ok=True)

    meta, points, gt = load_run(run_dir)
    cfg = meta["config"]
    layout = Layout(cfg["layout"])
    n_frames, fr = meta["n_frames"], meta["frame_rate_hz"]
    dt = 1.0 / fr
    t_axis = np.arange(n_frames) * dt
    doors = tuple((b.x0 - 1.0, b.y0 - 1.0, b.x1 + 1.0, b.y1 + 1.5) for b in (layout.entrance, layout.exit))

    # ---------------- radar points
    xyz = to_world(points, sensor_poses(meta, args.poses))
    ok = plausible(xyz, layout)
    points, xyz = points[ok], xyz[ok]
    ps = frame_slices(points[:, 0], n_frames)
    print(f"radar: {ok.mean():.0%} of points kept after height/room/furniture filtering")

    def radar_in(f):
        sl = slice(ps[f], ps[f + 1])
        return xyz[sl, :2], points[sl, 6].astype(float)

    pipelines = {"radar": radar_in}

    # ---------------- thermal detections
    thermal_merged = None
    if "thermal" in meta and (run_dir / "thermal_dets.npy").exists():
        from mmw.thermal_detect import replay_merged

        tdets = np.load(run_dir / "thermal_dets.npy")
        tcaps = np.load(run_dir / "thermal_captures.npy")
        cams = meta["thermal"][f"cameras_{args.poses}"]
        cam_xy = np.array([Pose.from_dict(d).position[:2] for d in cams])
        thermal_merged = list(replay_merged(tdets, tcaps, n_frames, len(cams), cam_xy))
        print(f"thermal: {np.mean([len(m) for m in thermal_merged]):.1f} merged detections per frame "
              f"from {len(cams)} cameras")

        def thermal_in(f):
            m = thermal_merged[f]
            return np.repeat(m[:, 1:3], THERMAL_POINTS_PER_DET, axis=0), np.full(len(m) * THERMAL_POINTS_PER_DET,
                                                                                THERMAL_SNR_DB)

        def fused_in(f):
            a, sa = radar_in(f)
            b, sb = thermal_in(f)
            return np.concatenate([a, b]), np.concatenate([sa, sb])

        pipelines["thermal"] = thermal_in
        pipelines["fused"] = fused_in

    # ---------------- track, write, evaluate
    snap_f = n_frames // 2
    results, snap_tracks = {}, {}
    nz = len(layout.zones)
    gt_total = gt_zc = None
    if gt is not None:
        gt_zone = layout.zone_index(gt[:, 3], gt[:, 4])
        gt_total, gt_zc = analytics.counts(gt[:, 0].astype(int), gt_zone, n_frames, nz)
    for name, inputs in pipelines.items():
        t0 = time.time()
        est, per_frame = run_tracker(name, n_frames, dt, inputs, doors)
        total, zc = write_outputs(out, name, est, layout, n_frames, dt)
        snap_tracks[name] = per_frame[snap_f]
        results[name] = {"est": est, "total": total, "zc": zc}
        print(f"{name}: tracked {n_frames} frames in {time.time() - t0:.1f}s")

    params = RadarParams.from_dict(cfg.get("radar", {}))
    th_poses = th_optics = None
    if "thermal" in meta:
        from mmw.thermal_camera import Optics, get_preset

        th_poses = [Pose.from_dict(d) for d in meta["thermal"]["cameras_true"]]
        th_optics = Optics(get_preset(meta["thermal"]["preset"]["name"],
                                      {k: v for k, v in meta["thermal"]["preset"].items() if k != "name"}))
    plots.plot_layout(out / "layout_coverage.png", layout, sensor_poses(meta, "true"), params, th_poses, th_optics)

    if gt is None:
        print(f"no ground truth (real recording); outputs in {out}")
        return

    all_metrics, zrows_by, H_est_by = {}, {}, {}
    for name, r in results.items():
        metrics, zrows, H_true, H_est = evaluate(r["est"], r["total"], r["zc"], gt, gt_total, gt_zc, layout,
                                                 n_frames, dt)
        all_metrics[name] = metrics
        zrows_by[name] = zrows
        H_est_by[name] = H_est
        with open(out / f"zone_summary_{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(zrows[0].keys()))
            w.writeheader()
            w.writerows(zrows)
    (out / "metrics.json").write_text(json.dumps(all_metrics, indent=2))

    gs = frame_slices(gt[:, 0], n_frames)
    sl = slice(ps[snap_f], ps[snap_f + 1])
    plots.plot_snapshot(out / "snapshot.png", layout, xyz[sl], gt[gs[snap_f]:gs[snap_f + 1], 3:5], snap_tracks,
                        snap_f * dt, thermal_merged[snap_f] if thermal_merged is not None else None)
    plots.plot_counts(out / "counts.png", t_axis, gt_total, gt_zc,
                      {k: (r["total"], r["zc"]) for k, r in results.items()}, layout)
    plots.plot_heatmaps(out / "heatmaps.png", layout, H_true, H_est_by)
    plots.plot_zone_share(out / "zone_share.png", zrows_by)
    if (run_dir / "thermal_frames.npy").exists():
        tf = np.load(run_dir / "thermal_frames.npy", mmap_mode="r")
        k = len(tf) // 2
        frames = tf[k].astype(np.float32) / 100.0 - 273.15
        cam_xy = np.array([p.position[:2] for p in th_poses])
        picks = [("sofas", (12.85, 7.55)), ("beds", (17.1, 13.45)), ("kitchens", (12.85, 16.4)),
                 ("lighting", (2.9, 10.5)), ("checkout", (21.35, 4.6)), ("entrance, sunlit floor", (2.9, 1.65))]
        cams = [int(np.argmin(np.hypot(cam_xy[:, 0] - x, cam_xy[:, 1] - y))) for _, (x, y) in picks]
        plots.plot_thermal_frames(out / "thermal_frames.png", frames, cams,
                                  [f"{n} (camera {c})" for (n, _), c in zip(picks, cams)])

    keys = ["avg_people_est", "count_mae", "count_bias", "precision", "recall", "mean_position_error_m",
            "recall_walking", "recall_browsing", "recall_seated", "recall_close_to_others", "heatmap_correlation",
            "zone_count_mae"]
    print(f"\ntrue average people in store: {gt_total.mean():.1f}")
    print(f"{'metric':<24}" + "".join(f"{n:>10}" for n in all_metrics))
    for k in keys:
        vals = [all_metrics[n][k] for n in all_metrics]
        print(f"{k:<24}" + "".join(f"{v:>10.2f}" if v is not None else f"{'-':>10}" for v in vals))
    print(f"\n{'zone share of time':<20}{'true':>8}" + "".join(f"{n:>9}" for n in zrows_by))
    for k in range(nz):
        print(f"{layout.zones[k].name:<20}{100 * zrows_by['radar'][k]['share_true']:>7.1f}%"
              + "".join(f"{100 * zrows_by[n][k]['share_est']:>8.1f}%" for n in zrows_by))
    print(f"outputs in {out}")


if __name__ == "__main__":
    main()
