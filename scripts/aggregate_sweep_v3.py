"""Aggregate phase1 wall-gen sweep_v3 holdout metrics (patient020 gate)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _p020_or_mean(data: dict) -> dict[str, float]:
    simple = data.get("simple") or {}
    mean = dict(simple.get("mean") or {})
    per = dict(simple.get("per_anchor") or {})
    p020 = dict(per.get("patient020") or {})
    src = p020 if p020 else mean
    return {
        "deploy_clot_score": float(src.get("deploy_clot_score") or 0.0),
        "deploy_clot_f1": float(src.get("deploy_clot_f1") or 0.0),
        "deploy_clot_relaxed_prec": float(
            src.get("deploy_clot_relaxed_prec")
            or src.get("deploy_clot_offwall_relaxed_prec")
            or 0.0
        ),
        "deploy_clot_relaxed_rec": float(
            src.get("deploy_clot_relaxed_rec")
            or src.get("deploy_clot_offwall_relaxed_rec")
            or 0.0
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sweep-dir",
        default="outputs/biochem/eda/wall_gen_sweep_v3",
    )
    ap.add_argument(
        "--out-csv",
        default="",
        help="Optional CSV path (default: <sweep-dir>/phase1_sweep_results.csv)",
    )
    ap.add_argument(
        "--leg-glob",
        default="WG_*",
        help="Directory glob under sweep-dir (default: WG_*)",
    )
    args = ap.parse_args()
    sweep_dir = Path(args.sweep_dir)
    out_csv = Path(args.out_csv) if str(args.out_csv).strip() else (sweep_dir / "phase1_sweep_results.csv")

    rows: list[dict[str, object]] = []
    for leg_dir in sorted(sweep_dir.glob(str(args.leg_glob))):
        if not leg_dir.is_dir():
            continue
        eval_path = leg_dir / "eval_holdout_cold.json"
        if not eval_path.is_file():
            continue
        try:
            data = json.loads(eval_path.read_text(encoding="utf-8"))
            m = _p020_or_mean(data)
            rows.append({"leg": leg_dir.name, **m})
        except Exception as e:
            print(f"[WARN] parse failed {eval_path}: {e}", flush=True)

    if not rows:
        print(f"[WARN] no completed legs under {sweep_dir}", flush=True)
        return 0

    fields = [
        "leg",
        "deploy_clot_score",
        "deploy_clot_f1",
        "deploy_clot_relaxed_prec",
        "deploy_clot_relaxed_rec",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    best = max(rows, key=lambda r: float(r["deploy_clot_score"]))
    print(f"[OK] Wrote {len(rows)} legs -> {out_csv}", flush=True)
    for row in rows:
        print(
            f"  {row['leg']}: score={float(row['deploy_clot_score']):.4f} "
            f"f1={float(row['deploy_clot_f1']):.4f}",
            flush=True,
        )
    print(
        f"[i] winner_by_score={best['leg']} "
        f"(deploy_clot_score={float(best['deploy_clot_score']):.4f})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
