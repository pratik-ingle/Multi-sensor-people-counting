"""Generate a synthetic multi-radar showroom recording.

    .venv/bin/python scripts/simulate.py --config configs/showroom.yaml --out data/sim_run1
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mmw.simulate import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/showroom.yaml")
    ap.add_argument("--out", default="data/sim_run1")
    ap.add_argument("--duration", type=float, help="override duration_s")
    ap.add_argument("--seed", type=int, help="override seed")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.duration is not None:
        cfg["duration_s"] = args.duration
    if args.seed is not None:
        cfg["seed"] = args.seed
    run(cfg, args.out)


if __name__ == "__main__":
    main()
