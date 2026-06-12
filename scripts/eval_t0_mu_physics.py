"""T0 mu physics eval: GT flow + species -> mu_pred vs GT spf.mu label.

Usage::

    python scripts/eval_t0_mu_physics.py --anchor patient007
    python scripts/eval_t0_mu_physics.py --anchor patient007 --gamma-mode max
    python scripts/eval_t0_mu_physics.py --sweep-gamma
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

from src.core_physics.t0_mu_physics import (  # noqa: E402
    eval_anchor_t0_mu,
    resolve_t0_gamma_mode,
    write_report,
)
from src.utils.paths import get_project_root  # noqa: E402


def _print_report(report) -> None:
    print(f"[OK] {report.anchor} gamma={report.physics['gamma_mode']} ratio_max={report.ratio_max:g}")
    print(f"[i] sidecar={report.gamma_sidecar} gates={report.pass_gates}")
    for row in report.times:
        print(
            f"  t={row['time']:2d} bulk_ratio={row['ratio_median_bulk']:.3f} "
            f"growth_ratio={row['ratio_median_growth']:.3f} "
            f"r_all={row['pearson_all']:.3f} r_growth={row['pearson_growth']:.3f} "
            f"logMAE_g={row['mu_log_mae_growth']:.3f} n_growth={row['n_growth']}"
        )
    print(f"[i] summary={report.summary}")


def main() -> int:
    ap = argparse.ArgumentParser(description="T0 mu physics eval (no GT mu input)")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--times", default="", help="comma times; default quartiles")
    ap.add_argument("--gamma-mode", default="auto", help="auto|comsol_sr|max|graph|kinematic")
    ap.add_argument("--no-hard-step", action="store_true")
    ap.add_argument("--ratio-max", type=float, default=0.0, help="0 = BiochemConfig default (80)")
    ap.add_argument("--out", default="outputs/biochem/clot_trigger/t0_mu_physics.json")
    ap.add_argument("--sweep-gamma", action="store_true", help="compare gamma modes")
    args = ap.parse_args()

    root = get_project_root()
    graph = root / args.anchor_dir / f"{args.anchor}.pt"
    if not graph.is_file():
        print(f"[ERR] missing graph {graph}", file=sys.stderr)
        return 1

    times = None
    if args.times.strip():
        times = [int(x.strip()) for x in args.times.split(",") if x.strip()]

    ratio_max = float(args.ratio_max) if float(args.ratio_max) > 0 else None
    hard_step = not args.no_hard_step

    if args.sweep_gamma:
        modes = ["comsol_sr", "max", "kinematic", "graph"]
        sweep_rows = []
        for mode in modes:
            if mode == "comsol_sr" and args.gamma_mode == "auto":
                gm = resolve_t0_gamma_mode(args.anchor, root=root)
                if gm != "comsol_sr":
                    continue
            gm = mode if mode != "comsol_sr" or resolve_t0_gamma_mode(args.anchor, root=root) == "comsol_sr" else None
            if gm is None and mode == "comsol_sr":
                continue
            report = eval_anchor_t0_mu(
                graph,
                times=times,
                gamma_mode=mode,
                hard_step=hard_step,
                ratio_max=ratio_max,
            )
            sweep_rows.append(report.to_dict())
            _print_report(report)
        out = root / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"anchor": args.anchor, "sweep": sweep_rows}, indent=2), encoding="utf-8")
        print(f"[save] {out}")
        return 0

    gamma_mode = None if args.gamma_mode == "auto" else args.gamma_mode
    report = eval_anchor_t0_mu(
        graph,
        times=times,
        gamma_mode=gamma_mode,
        hard_step=hard_step,
        ratio_max=ratio_max,
    )
    out = root / args.out
    write_report(report, out)
    _print_report(report)
    print(f"[save] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
