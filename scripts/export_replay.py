"""Record the live simulator into static files so the viewer page can replay it without a server
(for example on GitHub Pages, which only hosts static files).

    .venv/bin/python scripts/export_replay.py --seconds 90 --out docs

Writes:
  <out>/index.html                 the viewer page (same file as web/live.html; it detects replay mode)
  <out>/replay/meta.json           layout, cameras, timing, accuracy summary
  <out>/replay/chunk_NNN.json      frames (positions in centimetres) in chunks
  <out>/replay/mosaic_NNN.jpg      floor-wide thermal image of each chunk, as a sprite sheet
  <out>/replay/cams/cam_NN.jpg     raw output of each thermal camera, as a sprite sheet (loaded on click)
"""
import argparse
import json
import shutil
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mmw.live import LiveSim, colorize  # noqa: E402

FLOOR_RGB = np.array([17, 22, 29], dtype=float)  # page floor colour (--floor)


def cm(a):
    return np.round(np.asarray(a, dtype=float) * 100).astype(int).tolist()


def compact(fr: dict, f: int, dt: float) -> dict:
    """Live frame -> replay frame: lengths in centimetres, no embedded image."""
    p, g, td = fr["p"], fr["g"], fr["td"]
    return {
        "f": f, "t": round(f * dt, 2), "true": fr["true"], "zt": fr["zt"], "cams": fr["cams"], "ms": fr["ms"],
        "p": {"x": cm(p["x"]), "y": cm(p["y"]), "z": cm(p["z"]), "v": cm(p["v"]), "s": p["s"], "k": p["k"]},
        "g": {"x": cm(g["x"]), "y": cm(g["y"]), "r": cm(g["r"]), "b": g["b"], "st": g["st"]},
        "td": {"x": cm(td["x"]), "y": cm(td["y"])},
        "pipes": {k: {"est": v["est"], "ze": v["ze"], "d": v["d"],
                      "tr": {"id": v["tr"]["id"], "x": cm(v["tr"]["x"]), "y": cm(v["tr"]["y"]),
                             "vx": cm(v["tr"]["vx"]), "vy": cm(v["tr"]["vy"]), "ok": v["tr"]["ok"]}}
                  for k, v in fr["pipes"].items()},
    }


def sprite(tiles: list[np.ndarray], cols: int) -> Image.Image:
    h, w = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
    return Image.fromarray(sheet)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "showroom.yaml"))
    ap.add_argument("--out", default=str(ROOT / "docs"))
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--warmup", type=float, default=240.0, help="crowd build-up before anything is sensed")
    ap.add_argument("--settle", type=float, default=20.0, help="sensing before recording, so tracks exist at t=0")
    ap.add_argument("--chunk", type=int, default=100, help="frames per chunk file")
    ap.add_argument("--mosaic-every", type=int, default=2, help="floor thermal image every N frames")
    ap.add_argument("--cam-fps", type=float, default=2.0, help="raw camera frames per second in the replay")
    ap.add_argument("--quality", type=int, default=82, help="JPEG quality")
    args = ap.parse_args()

    out = Path(args.out)
    rep = out / "replay"
    if rep.exists():
        shutil.rmtree(rep)
    (rep / "cams").mkdir(parents=True)

    cfg = yaml.safe_load(Path(args.config).read_text())
    sim = LiveSim(cfg, warmup_s=args.warmup)
    sim.want_mosaic = False
    fps = 1.0 / sim.dt
    th = sim.thermal
    for _ in range(int(args.settle * fps)):
        sim.step()

    n = int(args.seconds * fps)
    cam_every = max(1, int(round(fps / args.cam_fps)))
    frames, mosaics = [], []
    cams = [[] for _ in range(th.n)] if th else []
    t0 = time.time()
    for f in range(n):
        fr = sim.step()
        frames.append(compact(fr, f, sim.dt))
        if th is not None:
            if f % args.mosaic_every == 0:
                rgba = sim.mosaic_rgba().astype(float)
                a = rgba[..., 3:4] / 255.0
                mosaics.append(np.round(rgba[..., :3] * a + FLOOR_RGB * (1 - a)).astype(np.uint8))
            if f % cam_every == 0:
                for k in range(th.n):
                    cams[k].append(colorize(th.latest[k]))
        if (f + 1) % 100 == 0:
            print(f"  recorded {f + 1}/{n} frames ({time.time() - t0:.0f}s)", flush=True)

    n_chunks = (n + args.chunk - 1) // args.chunk
    per_chunk = args.chunk // args.mosaic_every
    mcols = 10
    for c in range(n_chunks):
        (rep / f"chunk_{c:03d}.json").write_text(
            json.dumps({"frames": frames[c * args.chunk:(c + 1) * args.chunk]}, separators=(",", ":")))
        if mosaics:
            sprite(mosaics[c * per_chunk:(c + 1) * per_chunk], mcols).save(
                rep / f"mosaic_{c:03d}.jpg", quality=args.quality)
    ccols = 10
    for k, tiles in enumerate(cams):
        sprite(tiles, ccols).save(rep / "cams" / f"cam_{k:02d}.jpg", quality=args.quality)

    summary = {}
    for name in sim.trackers:
        d = np.concatenate([np.array(fr["pipes"][name]["d"], dtype=float) for fr in frames])
        ok = np.concatenate([np.array(fr["pipes"][name]["tr"]["ok"], dtype=float) for fr in frames])
        err = np.array([fr["pipes"][name]["est"] - fr["true"] for fr in frames], dtype=float)
        summary[name] = {"count_mae": float(np.abs(err).mean()), "count_bias": float(err.mean()),
                         "recall": float(d.mean()) if len(d) else None,
                         "precision": float(ok.mean()) if len(ok) else None}
    meta = {
        "layout": sim.layout_json(),
        "fps": fps, "n_frames": n, "seconds": args.seconds, "chunk": args.chunk, "n_chunks": n_chunks,
        "mosaic_every": args.mosaic_every, "mosaic_tile": [int(mosaics[0].shape[1]), int(mosaics[0].shape[0])] if mosaics else None,
        "mosaic_cols": mcols, "cam_every": cam_every, "cam_tile": [int(th.optics.W), int(th.optics.H)] if th else None,
        "cam_cols": ccols, "cam_frames": len(cams[0]) if cams else 0,
        "avg_true": float(np.mean([fr["true"] for fr in frames])),
        "recorded": date.today().isoformat(), "summary": summary,
    }
    (rep / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))
    shutil.copy(ROOT / "web" / "live.html", out / "index.html")
    (out / ".nojekyll").touch()
    if th is not None:
        th.close()
    size = sum(p.stat().st_size for p in rep.rglob("*") if p.is_file()) / 1e6
    print(f"replay: {n} frames ({args.seconds:.0f} s), {size:.1f} MB -> {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
