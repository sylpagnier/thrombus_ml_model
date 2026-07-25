"""Summarize 2h tile-mode explore: A vs UnionTile vs PerComponent.

Verdict focuses on whether per-clot-region tiles help hop_ge2 localization
vs the default union tile, without tanking clot F1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

KEYS = (
    "deploy_clot_f1",
    "deploy_clot_score",
    "deploy_clot_offwall_n_pred_hop_ge2",
    "deploy_clot_offwall_n_gt_hop_ge2",
    "deploy_clot_offwall_strict_f1_hop_ge2",
    "deploy_clot_offwall_strict_f1",
)


def _mean(report: dict) -> dict:
    simple = report.get("simple") or {}
    return dict(simple.get("mean") or {})


def _load_probe(path: Path, label: str) -> dict | None:
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    m = _mean(report)
    return {
        "label": label,
        "path": str(path),
        "mean": {k: float(m.get(k, 0.0) or 0.0) for k in KEYS},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-root",
        default="outputs/biochem/offwall_model/wc_v7_tile_cc_explore_2h",
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = Path(args.run_root)
    if not root.is_absolute():
        root = Path.cwd() / root

    arms = {}
    for name in ("A", "UnionTile", "PerComponent"):
        row = _load_probe(root / f"probe_{name}.json", name)
        if row is None:
            print(f"[WARN] missing probe_{name}.json", flush=True)
            continue
        arms[name] = row

    union = arms.get("UnionTile")
    cc = arms.get("PerComponent")
    a = arms.get("A")

    verdict = "incomplete"
    detail = {}
    if union and cc:
        d_hop = (
            cc["mean"]["deploy_clot_offwall_n_pred_hop_ge2"]
            - union["mean"]["deploy_clot_offwall_n_pred_hop_ge2"]
        )
        d_strict = (
            cc["mean"]["deploy_clot_offwall_strict_f1_hop_ge2"]
            - union["mean"]["deploy_clot_offwall_strict_f1_hop_ge2"]
        )
        d_clot = cc["mean"]["deploy_clot_f1"] - union["mean"]["deploy_clot_f1"]
        detail = {
            "d_hop_ge2_n_pred_CC_minus_Union": d_hop,
            "d_hop_ge2_strict_CC_minus_Union": d_strict,
            "d_clot_f1_CC_minus_Union": d_clot,
        }
        wall_ok = d_clot >= -0.02
        if wall_ok and d_strict > 0.01 and d_hop > 0.5:
            verdict = "per_component_helps"
        elif wall_ok and (d_strict > 0.01 or d_hop > 0.5):
            verdict = "per_component_weak_help"
        elif (not wall_ok) and (d_strict > 0.01 or d_hop > 0.5):
            verdict = "per_component_helps_lumen_hurts_wall"
        elif d_strict < -0.01 or d_hop < -0.5:
            verdict = "per_component_hurts"
        else:
            verdict = "null_or_tie"

    report = {
        "arms": arms,
        "a_clot_f1": None if a is None else a["mean"]["deploy_clot_f1"],
        "delta_CC_minus_Union": detail,
        "verdict": verdict,
    }

    print("=" * 72, flush=True)
    print("TILE CC EXPLORE 2H  (A / UnionTile / PerComponent)", flush=True)
    print("=" * 72, flush=True)
    hdr = f"{'arm':<14} {'clot_f1':>8} {'hop_ge2':>8} {'strict':>8}"
    print(hdr, flush=True)
    for name in ("A", "UnionTile", "PerComponent"):
        row = arms.get(name)
        if row is None:
            continue
        m = row["mean"]
        print(
            f"{name:<14} {m['deploy_clot_f1']:8.4f} "
            f"{m['deploy_clot_offwall_n_pred_hop_ge2']:8.1f} "
            f"{m['deploy_clot_offwall_strict_f1_hop_ge2']:8.4f}",
            flush=True,
        )
    print(f"[i] verdict={verdict} detail={detail}", flush=True)

    if args.out.strip():
        out = Path(args.out)
        if not out.is_absolute():
            out = Path.cwd() / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[save] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
