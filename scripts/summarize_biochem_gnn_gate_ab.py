"""Summarise gate A/B eval results and pick a winner.

Usage:
    python scripts/summarize_biochem_gnn_gate_ab.py \
        --eval-dirs '{"A": "outputs/.../baseline_sigmoid/eval", "B": "..."}' \
        --output outputs/.../gate_ab_summary.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any


GUIDING_KEYS = [
    "deploy_clot_score",
    "clot_guiding_main",
    "clot_f1_main",
    "clot_score_main",
    "clot_relaxed_f05_main",
]


def _load_eval_dir(eval_dir: str) -> dict[str, Any]:
    """Load deploy_ab_eval.json (preferred) or legacy per-anchor JSON dir."""
    p = pathlib.Path(eval_dir)
    ab = p / "deploy_ab_eval.json"
    if ab.is_file():
        data = json.loads(ab.read_text(encoding="utf-8"))
        rows = [
            r
            for r in data.get("rows", [])
            if r.get("mode") == "deploy_frozen" and not r.get("error")
        ]
        if not rows:
            return {}
        p007 = next((r for r in rows if r.get("anchor") == "patient007"), rows[0])
        hold = [r for r in rows if r.get("anchor") != "patient007"]
        out: dict[str, float] = {}
        for key, val in p007.items():
            if isinstance(val, (int, float)):
                out[key] = float(val)
        out["deploy_clot_score"] = float(
            p007.get("clot_score_main", p007.get("clot_guiding_main", p007.get("clot_f1_main", 0.0)))
        )
        if hold:
            for key in ("clot_f1_main", "clot_guiding_main", "clot_score_main", "clot_relaxed_f05_main"):
                out[f"holdout_mean_{key}"] = sum(float(r.get(key, 0.0)) for r in hold) / len(hold)
        return out

    results: dict[str, list[float]] = {}
    files = sorted(p.glob("*.json"))
    for fp in files:
        try:
            data = json.loads(fp.read_text())
        except Exception:
            continue
        for k, v in data.items():
            if isinstance(v, (int, float)):
                results.setdefault(k, []).append(float(v))
    if not results:
        return {}
    return {k: (sum(vs) / len(vs)) for k, vs in results.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dirs", required=True, help="JSON dict mapping leg key -> eval dir path")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    eval_dirs: dict[str, str] = json.loads(args.eval_dirs)

    leg_metrics: dict[str, dict[str, float]] = {}
    for leg, d in eval_dirs.items():
        metrics = _load_eval_dir(d)
        leg_metrics[leg] = metrics
        if not metrics:
            print(f"[WARN] leg {leg}: no eval JSONs found in {d}")

    # Pick winner by first available guiding key
    winner = None
    best_score = -1e9
    winner_key = None
    for gk in GUIDING_KEYS:
        scores = {lg: m.get(gk, None) for lg, m in leg_metrics.items() if m.get(gk) is not None}
        if not scores:
            continue
        winner_key = gk
        winner = max(scores, key=lambda k: scores[k])
        best_score = scores[winner]
        break

    summary = {
        "legs": leg_metrics,
        "winner": winner,
        "winner_key": winner_key,
        "winner_score": best_score,
    }

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print(f"[OK] summary -> {out}")
    if winner:
        print(f"[OK] winner: Leg {winner}  ({winner_key}={best_score:.4f})")
    else:
        print("[WARN] no guiding metric found; cannot pick winner")
    return 0


if __name__ == "__main__":
    sys.exit(main())
