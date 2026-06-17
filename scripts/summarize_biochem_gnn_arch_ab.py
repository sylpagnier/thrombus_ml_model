"""Summarize GraphSAGE vs GNODE pushforward A/B eval JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load(path: Path) -> dict:
    if not path.is_file():
        return {"error": f"missing: {path}", "rows": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(rows: list[dict], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r and r[key] == r[key]]
    return sum(vals) / len(vals) if vals else float("nan")


def _summarize_leg(payload: dict) -> dict:
    rows = list(payload.get("rows") or payload.get("results") or [])
    if not rows and isinstance(payload.get("by_anchor"), dict):
        rows = list(payload["by_anchor"].values())
    p007 = next((r for r in rows if r.get("anchor") == "patient007"), None)
    return {
        "n_anchors": len(rows),
        "mean_clot_guiding_main": _mean(rows, "clot_guiding_main"),
        "mean_clot_f1_main": _mean(rows, "clot_f1_main"),
        "mean_clot_score_main": _mean(rows, "clot_score_main"),
        "mean_clot_relaxed_f05_main": _mean(rows, "clot_relaxed_f05_main"),
        "p007_clot_guiding_main": float(p007["clot_guiding_main"]) if p007 else float("nan"),
        "p007_clot_f1_main": float(p007["clot_f1_main"]) if p007 else float("nan"),
        "p007_clot_score_main": float(p007.get("clot_score_main", p007.get("clot_guiding_main", float("nan"))))
        if p007
        else float("nan"),
        "p007_time_main": int(p007["time_main"]) if p007 else -1,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize biochem GNN arch A/B")
    ap.add_argument("--sage-eval", default="outputs/biochem/biochem_gnn/arch_ab/sage/eval_deploy_frozen.json")
    ap.add_argument("--gnode-eval", default="outputs/biochem/biochem_gnn/arch_ab/gnode/eval_deploy_frozen.json")
    ap.add_argument("--out", default="outputs/biochem/biochem_gnn/arch_ab/arch_ab_summary.json")
    args = ap.parse_args()

    sage = _summarize_leg(_load(Path(args.sage_eval)))
    gnode = _summarize_leg(_load(Path(args.gnode_eval)))

    score_sage = sage["mean_clot_guiding_main"]
    score_gnode = gnode["mean_clot_guiding_main"]
    p007_sage = sage["p007_clot_guiding_main"]
    p007_gnode = gnode["p007_clot_guiding_main"]

    if score_sage == score_sage and score_gnode == score_gnode:
        if abs(score_sage - score_gnode) < 1e-6:
            winner = "tie"
        else:
            winner = "sage" if score_sage > score_gnode else "gnode"
    else:
        winner = "unknown"

    out = {
        "winner_by_mean_guiding": winner,
        "delta_mean_guiding_sage_minus_gnode": score_sage - score_gnode
        if score_sage == score_sage and score_gnode == score_gnode
        else float("nan"),
        "delta_p007_guiding_sage_minus_gnode": p007_sage - p007_gnode
        if p007_sage == p007_sage and p007_gnode == p007_gnode
        else float("nan"),
        "sage": sage,
        "gnode": gnode,
        "sage_eval": str(args.sage_eval),
        "gnode_eval": str(args.gnode_eval),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"[OK] winner (mean guiding): {winner}", flush=True)
    print(
        f"[i] sage mean guiding={score_sage:.4f} p007={p007_sage:.4f} | "
        f"gnode mean guiding={score_gnode:.4f} p007={p007_gnode:.4f}",
        flush=True,
    )
    print(f"[save] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
