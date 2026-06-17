"""Summarize onset diagnosis pack eval outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def _onset_time(row: dict, *, metric: str, threshold: float) -> float:
    vals = row.get(metric) or {}
    if not isinstance(vals, dict) or not vals:
        return float("nan")
    times = sorted(int(k) for k in vals.keys())
    for t in times:
        if float(vals[str(t)]) >= threshold:
            return float(t)
    return float(times[-1])


def _summarize_condition(path: Path, *, onset_threshold: float) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [r for r in payload.get("rows", []) if not r.get("error")]
    out = {
        "rows": len(rows),
        "p007_f1_main": float("nan"),
        "holdout_mean_f1_main": float("nan"),
        "mean_onset_t_guiding": float("nan"),
        "mean_onset_t_f1": float("nan"),
    }
    if not rows:
        return out
    p007 = next((r for r in rows if r.get("anchor") == "patient007"), rows[0])
    hold = [r for r in rows if r.get("anchor") != "patient007"]
    out["p007_f1_main"] = float(p007.get("clot_f1_main", 0.0))
    out["holdout_mean_f1_main"] = (
        mean(float(r.get("clot_f1_main", 0.0)) for r in hold) if hold else float("nan")
    )
    onset_g = [_onset_time(r, metric="clot_guiding", threshold=onset_threshold) for r in rows]
    onset_f = [_onset_time(r, metric="clot_f1", threshold=onset_threshold) for r in rows]
    out["mean_onset_t_guiding"] = mean(onset_g)
    out["mean_onset_t_f1"] = mean(onset_f)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--onset-threshold", type=float, default=0.2)
    args = ap.parse_args()

    run_root = Path(args.run_root)
    eval_files = sorted(run_root.glob("*/eval/deploy_ab_eval.json"))
    summary: dict[str, dict] = {}
    for fp in eval_files:
        key = fp.parent.parent.name
        summary[key] = _summarize_condition(fp, onset_threshold=float(args.onset_threshold))

    # ranking: higher guiding quality proxy + closer onset to baseline
    base = summary.get("baseline", {})
    base_onset = float(base.get("mean_onset_t_guiding", float("nan")))
    ranked = []
    for k, v in summary.items():
        onset = float(v.get("mean_onset_t_guiding", float("nan")))
        onset_delta = onset - base_onset if base_onset == base_onset else float("nan")
        score = float(v.get("holdout_mean_f1_main", 0.0))
        ranked.append(
            {
                "condition": k,
                "holdout_mean_f1_main": score,
                "mean_onset_t_guiding": onset,
                "onset_delta_vs_baseline": onset_delta,
            }
        )
    ranked.sort(key=lambda r: r["holdout_mean_f1_main"], reverse=True)

    out_json = run_root / "onset_diagnosis_summary.json"
    out_md = run_root / "onset_diagnosis_report.md"
    payload = {"conditions": summary, "ranking": ranked, "onset_threshold": args.onset_threshold}
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Onset Diagnosis Pack Report",
        "",
        f"Onset threshold: guiding/f1 >= {args.onset_threshold}",
        "",
        "| Condition | holdout_mean_f1 | p007_f1 | mean_onset_t_guiding | onset_delta_vs_baseline |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in ranked:
        c = r["condition"]
        m = summary[c]
        lines.append(
            f"| {c} | {float(m.get('holdout_mean_f1_main', 0.0)):.3f} | "
            f"{float(m.get('p007_f1_main', 0.0)):.3f} | "
            f"{float(m.get('mean_onset_t_guiding', 0.0)):.1f} | "
            f"{float(r.get('onset_delta_vs_baseline', 0.0)):+.1f} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {out_json}")
    print(f"[OK] wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
