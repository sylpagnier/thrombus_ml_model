"""Rank precision-sweep legs by relaxed precision with a recall floor.

Objective per anchor (matches SPECIES_CONTINUOUS_CLOUT_SCORE=relaxed_prec_floor):
  rec <= 0           -> 0
  rec >= floor       -> relaxed_prec
  0 < rec < floor    -> relaxed_prec * rec / floor

Leg score = w_p007 * obj(p007) + (1 - w_p007) * mean(obj(holdout)).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _obj(prec: float, rec: float, floor: float) -> float:
    if rec <= 0.0:
        return 0.0
    if floor <= 0.0 or rec >= floor:
        return float(prec)
    return float(prec) * (rec / floor)


def _leg_metrics(eval_path: Path, floor: float, w_p007: float) -> dict:
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    rows = [r for r in data.get("rows", []) if r.get("mode") == "deploy_frozen" and not r.get("error")]
    if not rows:
        return {}
    p007 = next((r for r in rows if r.get("anchor") == "patient007"), rows[0])
    hold = [r for r in rows if r.get("anchor") != "patient007"]

    def rp(r):
        return float(r.get("clot_relaxed_prec_main", 0.0))

    def rr(r):
        return float(r.get("clot_relaxed_rec_main", 0.0))

    p007_obj = _obj(rp(p007), rr(p007), floor)
    hold_obj = sum(_obj(rp(r), rr(r), floor) for r in hold) / len(hold) if hold else p007_obj
    out = {
        "p007_relaxed_prec": rp(p007),
        "p007_relaxed_rec": rr(p007),
        "p007_relaxed_f05": float(p007.get("clot_relaxed_f05_main", 0.0)),
        "p007_f1": float(p007.get("clot_f1_main", 0.0)),
        "p007_obj": p007_obj,
        "holdout_relaxed_prec": sum(rp(r) for r in hold) / len(hold) if hold else rp(p007),
        "holdout_relaxed_rec": sum(rr(r) for r in hold) / len(hold) if hold else rr(p007),
        "holdout_obj": hold_obj,
        "score": w_p007 * p007_obj + (1.0 - w_p007) * hold_obj,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-root", required=True)
    ap.add_argument("--floor", type=float, default=0.30)
    ap.add_argument("--w-p007", type=float, default=0.5)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    root = Path(args.sweep_root)
    if not root.is_absolute():
        root = REPO / root

    legs = []
    baselines = []
    for leg_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        ev = leg_dir / "eval" / "deploy_ab_eval.json"
        if not ev.is_file():
            continue
        m = _leg_metrics(ev, args.floor, args.w_p007)
        if not m:
            continue
        cfg = {}
        cfg_path = leg_dir / "leg_config.json"
        if cfg_path.is_file():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        entry = {"label": leg_dir.name, "config": cfg, "metrics": m}
        if leg_dir.name.startswith("baseline"):
            baselines.append(entry)
        else:
            legs.append(entry)

    legs.sort(key=lambda r: float(r["metrics"].get("score", 0.0)), reverse=True)

    # Current best (locked) baseline, if evaluated, gates promotion.
    baseline = max(baselines, key=lambda r: float(r["metrics"].get("score", 0.0))) if baselines else None
    best = legs[0] if legs else None
    promote_ok = False
    if best is not None:
        if baseline is None:
            promote_ok = True
        else:
            promote_ok = float(best["metrics"]["score"]) > float(baseline["metrics"]["score"])
        best["beats_baseline"] = bool(promote_ok)
        if baseline is not None:
            best["delta_vs_baseline"] = float(best["metrics"]["score"]) - float(baseline["metrics"]["score"])

    summary = {"floor": args.floor, "w_p007": args.w_p007, "legs": legs,
               "baseline": baseline, "best": best, "promote_ok": promote_ok}
    out_json = Path(args.out_json)
    if not out_json.is_absolute():
        out_json = REPO / out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Species precision sweep (relaxed precision, recall floor)",
        "",
        f"Objective: relaxed precision with recall floor = {args.floor}; "
        f"leg score = {args.w_p007:.2f} p007 + {1 - args.w_p007:.2f} holdout.",
        "",
        "| Rank | Leg | score | p007 rprec | p007 rrec | p007 f05 | p007 f1 | hold rprec | hold rrec |",
        "|------|-----|-------|------------|-----------|----------|---------|------------|-----------|",
    ]
    for i, leg in enumerate(legs, start=1):
        m = leg["metrics"]
        lines.append(
            f"| {i} | {leg['label']} | {m['score']:.3f} | {m['p007_relaxed_prec']:.3f} | "
            f"{m['p007_relaxed_rec']:.3f} | {m['p007_relaxed_f05']:.3f} | {m['p007_f1']:.3f} | "
            f"{m['holdout_relaxed_prec']:.3f} | {m['holdout_relaxed_rec']:.3f} |"
        )
    if baseline is not None:
        m = baseline["metrics"]
        lines.append(
            f"| - | **{baseline['label']} (current best)** | {m['score']:.3f} | {m['p007_relaxed_prec']:.3f} | "
            f"{m['p007_relaxed_rec']:.3f} | {m['p007_relaxed_f05']:.3f} | {m['p007_f1']:.3f} | "
            f"{m['holdout_relaxed_prec']:.3f} | {m['holdout_relaxed_rec']:.3f} |"
        )
    if best is not None:
        b = best
        lines += ["", f"Best sweep leg: **{b['label']}** (score {b['metrics']['score']:.3f}, "
                  f"p007 relaxed precision {b['metrics']['p007_relaxed_prec']:.3f} "
                  f"@ recall {b['metrics']['p007_relaxed_rec']:.3f})"]
        if baseline is not None:
            verdict = "BEATS" if promote_ok else "does NOT beat"
            lines.append(
                f"Verdict: best **{verdict}** current locked baseline "
                f"(score {baseline['metrics']['score']:.3f}; delta "
                f"{best.get('delta_vs_baseline', 0.0):+.3f}). promote_ok={promote_ok}"
            )
        if b.get("config"):
            lines.append(f"Config: `{json.dumps(b['config'])}`")

    out_md = Path(args.out_md)
    if not out_md.is_absolute():
        out_md = REPO / out_md
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"best": summary["best"], "baseline": summary["baseline"],
                      "promote_ok": summary["promote_ok"]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
