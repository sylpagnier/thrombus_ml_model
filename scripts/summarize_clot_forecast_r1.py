#!/usr/bin/env python3
"""Summarize R1 prong A/B/C/D multi-anchor eval + train logs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LADDER = ROOT / "outputs/biochem/clot_forecast_ladder"
PRONGS = ("r1_prong_a", "r1_prong_b", "r1_prong_c", "r1_prong_d")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _best_val_from_train_log(path: Path) -> dict | None:
    rows = _load_jsonl(path)
    if not rows:
        return None
    best = max(rows, key=lambda r: float(r.get("val_score", -999)))
    return {
        "epoch": best.get("epoch"),
        "val_f1": best.get("val", {}).get("clot_f1"),
        "val_logMAE": best.get("val", {}).get("mu_log_mae"),
        "val_score": best.get("val_score"),
        "pred_pos_frac": best.get("val", {}).get("pred_pos_frac"),
    }


def main() -> int:
    summary: dict = {"prongs": {}, "winner_backbone": None, "notes": []}
    best_mean = -1.0
    best_p007 = -1.0
    winner = None

    for leg in PRONGS:
        leg_dir = LADDER / leg
        eval_rows = _load_jsonl(leg_dir / "multi_anchor.jsonl")
        train_best = _best_val_from_train_log(leg_dir / "clot_phi_train_log.jsonl")
        if not eval_rows and train_best is None:
            continue
        f1s = [float(r["val"]["clot_f1"]) for r in eval_rows] if eval_rows else []
        p007 = next(
            (float(r["val"]["clot_f1"]) for r in eval_rows if r.get("anchor") == "patient007"),
            None,
        )
        row = {
            "leg": leg,
            "n_anchors": len(eval_rows),
            "mean_f1": sum(f1s) / len(f1s) if f1s else None,
            "min_f1": min(f1s) if f1s else None,
            "p007_f1": p007,
            "per_anchor": {
                r["anchor"]: {
                    "f1": r["val"]["clot_f1"],
                    "rec": r["val"]["clot_rec"],
                    "logMAE": r["val"]["mu_log_mae"],
                    "pred_pos_frac": r["val"]["pred_pos_frac"],
                    "score": r.get("val_score"),
                }
                for r in eval_rows
            },
            "train_best": train_best,
        }
        summary["prongs"][leg] = row
        if f1s and row["mean_f1"] is not None:
            if row["mean_f1"] > best_mean or (row["mean_f1"] == best_mean and (p007 or 0) > best_p007):
                best_mean = row["mean_f1"]
                best_p007 = p007 or 0.0
                winner = leg

    if winner:
        summary["winner_backbone"] = winner
    out = LADDER / "r1_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK]  wrote {out}")
    for leg, row in summary["prongs"].items():
        print(
            f"  {leg}: mean_f1={row['mean_f1']:.3f} min_f1={row['min_f1']:.3f} "
            f"p007={row['p007_f1']:.3f}"
            if row.get("mean_f1") is not None
            else f"  {leg}: (no eval)"
        )
    if winner:
        print(f"[i]  backbone winner: {winner}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
