"""Summarize flow-alignment + onset-loss ablation diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def _first_crossing(timeseries: dict, threshold: float) -> float:
    if not isinstance(timeseries, dict) or not timeseries:
        return float("nan")
    ts = sorted((int(k), float(v)) for k, v in timeseries.items())
    for t, v in ts:
        if v >= threshold:
            return float(t)
    return float(ts[-1][0])


def _metrics(eval_json: Path, onset_threshold: float) -> dict[str, float]:
    d = json.loads(eval_json.read_text(encoding="utf-8"))
    rows = [r for r in d.get("rows", []) if r.get("mode") == "deploy_frozen" and not r.get("error")]
    if not rows:
        return {
            "p007_f1_main": float("nan"),
            "holdout_mean_f1_main": float("nan"),
            "mean_onset_t_guiding": float("nan"),
            "mean_onset_t_f1": float("nan"),
        }
    p007 = next((r for r in rows if r.get("anchor") == "patient007"), rows[0])
    hold = [r for r in rows if r.get("anchor") != "patient007"]
    onset_g = [_first_crossing(r.get("clot_guiding", {}), onset_threshold) for r in rows]
    onset_f1 = [_first_crossing(r.get("clot_f1", {}), onset_threshold) for r in rows]
    return {
        "p007_f1_main": float(p007.get("clot_f1_main", 0.0)),
        "holdout_mean_f1_main": (
            mean(float(r.get("clot_f1_main", 0.0)) for r in hold) if hold else float("nan")
        ),
        "mean_onset_t_guiding": mean(onset_g),
        "mean_onset_t_f1": mean(onset_f1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--onset-threshold", type=float, default=0.2)
    args = ap.parse_args()

    run_root = Path(args.run_root)
    cond_dirs = sorted([p for p in run_root.iterdir() if p.is_dir()])
    conditions: dict[str, dict] = {}
    for cd in cond_dirs:
        eval_json = cd / "eval" / "deploy_ab_eval.json"
        meta_json = cd / "meta.json"
        if not eval_json.is_file():
            continue
        m = _metrics(eval_json, float(args.onset_threshold))
        meta = json.loads(meta_json.read_text(encoding="utf-8")) if meta_json.is_file() else {}
        conditions[cd.name] = {"meta": meta, **m}

    base = conditions.get("baseline", {})
    base_onset = float(base.get("mean_onset_t_guiding", float("nan")))
    ranking = []
    for name, c in conditions.items():
        onset = float(c.get("mean_onset_t_guiding", float("nan")))
        ranking.append(
            {
                "condition": name,
                "holdout_mean_f1_main": float(c.get("holdout_mean_f1_main", 0.0)),
                "p007_f1_main": float(c.get("p007_f1_main", 0.0)),
                "mean_onset_t_guiding": onset,
                "onset_delta_vs_baseline": onset - base_onset if base_onset == base_onset else float("nan"),
            }
        )
    ranking.sort(key=lambda r: r["holdout_mean_f1_main"], reverse=True)

    out_json = run_root / "phase2_diagnostics_summary.json"
    out_md = run_root / "phase2_diagnostics_report.md"
    payload = {
        "onset_threshold": args.onset_threshold,
        "conditions": conditions,
        "ranking": ranking,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Phase-2 Onset Diagnostics Report",
        "",
        f"Onset threshold: {args.onset_threshold}",
        "",
        "| Condition | holdout_mean_f1 | p007_f1 | mean_onset_t_guiding | onset_delta_vs_baseline |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in ranking:
        lines.append(
            f"| {r['condition']} | {r['holdout_mean_f1_main']:.3f} | {r['p007_f1_main']:.3f} | "
            f"{r['mean_onset_t_guiding']:.1f} | {r['onset_delta_vs_baseline']:+.1f} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {out_json}")
    print(f"[OK] wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
