"""Summarize quick time-context A/B eval outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _metrics(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        r
        for r in data.get("rows", [])
        if r.get("mode") == "deploy_frozen" and not r.get("error")
    ]
    p007 = next((r for r in rows if r.get("anchor") == "patient007"), rows[0] if rows else {})
    hold = [r for r in rows if r.get("anchor") != "patient007"]
    hold_mean = (
        sum(float(r.get("clot_f1_main", 0.0)) for r in hold) / max(len(hold), 1)
        if rows
        else 0.0
    )
    return {
        "deploy_clot_score": float(p007.get("clot_score_main", 0.0)),
        "clot_f1_main": float(p007.get("clot_f1_main", 0.0)),
        "holdout_mean_clot_f1_main": float(hold_mean),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-a", required=True)
    ap.add_argument("--eval-b", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    legs = {
        "A": _metrics(Path(args.eval_a)),
        "B": _metrics(Path(args.eval_b)),
    }
    winner = "B" if legs["B"]["deploy_clot_score"] > legs["A"]["deploy_clot_score"] else "A"
    summary = {
        "legs": legs,
        "winner": winner,
        "winner_key": "deploy_clot_score",
    }
    Path(args.out_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = [
        "# Time Context A/B (30m)",
        "",
        "| Leg | deploy_clot_score | p007 clot_f1 | holdout mean clot_f1 |",
        "|-----|-------------------|--------------|-----------------------|",
        f"| A | {legs['A']['deploy_clot_score']:.3f} | {legs['A']['clot_f1_main']:.3f} | {legs['A']['holdout_mean_clot_f1_main']:.3f} |",
        f"| B | {legs['B']['deploy_clot_score']:.3f} | {legs['B']['clot_f1_main']:.3f} | {legs['B']['holdout_mean_clot_f1_main']:.3f} |",
        "",
        f"Winner: {winner} (deploy_clot_score)",
    ]
    Path(args.out_md).write_text("\n".join(md) + "\n", encoding="utf-8")
    print("[OK] wrote", args.out_json)
    print("[OK] wrote", args.out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
