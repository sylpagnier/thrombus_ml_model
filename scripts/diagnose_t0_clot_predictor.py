"""Sweep T0 clot predictors: GT u,v,p + species (no spf.mu, no spf.sr sidecar).

Usage::

    python scripts/diagnose_t0_clot_predictor.py --anchor patient007
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.core_physics.t0_clot_predictor import sweep_t0_clot_predictors  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="T0 clot predictor sweep (no GT mu/sr)")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--times", default="", help="comma list; default ~10 evenly spaced + last")
    ap.add_argument("--out", default="outputs/biochem/clot_trigger/t0_clot_predictor_sweep.json")
    args = ap.parse_args()

    root = get_project_root()
    graph_path = root / "data/processed/graphs_biochem_anchors" / f"{args.anchor}.pt"
    if not graph_path.is_file():
        raise FileNotFoundError(graph_path)

    data = torch.load(graph_path, map_location="cpu", weights_only=False)
    n_steps = int(data.y.shape[0])
    if args.times.strip():
        times = [int(x.strip()) for x in args.times.split(",") if x.strip()]
    else:
        times = list(range(0, n_steps, max(1, n_steps // 10))) + [n_steps - 1]
        times = sorted({max(0, min(t, n_steps - 1)) for t in times})

    report = sweep_t0_clot_predictors(data, anchor=args.anchor, times=times)
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    best = report.best.get("overall", {})
    print(f"[OK] {args.anchor} -> {out_path}")
    print(
        f"[i] best overall: {best.get('predictor')} "
        f"gamma={best.get('gamma_mode')} "
        f"gscale={best.get('gamma_scale')} "
        f"poi={best.get('poiseuille_scale')} "
        f"mean_F1_growth={best.get('mean_f1_growth_times', float('nan')):.3f} "
        f"F1_last={best.get('f1_last', float('nan')):.3f}"
    )
    for key in ("species_hard", "species_soft", "mu_growth", "mu_growth_nucleation"):
        b = report.best.get(key, {})
        if b:
            print(
                f"[i] best {key}: F1_growth={b.get('mean_f1_growth_times', float('nan')):.3f} "
                f"F1_last={b.get('f1_last', float('nan')):.3f} "
                f"gamma={b.get('gamma_mode')} poi={b.get('poiseuille_scale')}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
