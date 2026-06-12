"""Quick analysis of clot_prior_rule_sweep.json."""
import json
import statistics
from pathlib import Path

d = json.loads(Path("outputs/biochem/diagnostics/clot_prior_rule_sweep.json").read_text())
summary = {r["rule"]: r for r in d["summary"]}
rows = d["per_anchor"]

compare = [
    "prior_p0.80",
    "prior_p0.85",
    "prior_p0.85|t0_strip",
    "prior_p0.90",
    "prior_p0.80|t0_strip",
    "prior_p0.80|flux_stream_top10",
]
print("=== Mean across 6 anchors ===")
print(f"{'rule':<50} {'F1':>6} {'prec':>6} {'rec':>6} {'pred+':>6} {'gt+':>6}")
for name in compare:
    if name not in summary:
        print(f"{name}: NOT IN SWEEP")
        continue
    r = summary[name]
    print(
        f"{r['rule']:<50} {r['mean_band_f1']:>6.3f} {r['mean_band_prec']:>6.3f} "
        f"{r['mean_band_rec']:>6.3f} {r['mean_pred_pos_frac']:>6.3f} {r['mean_gt_pos_frac']:>6.3f}"
    )

best = "prior_p0.80"
print("\n=== Per-anchor: best vs prior_p0.85|t0_strip ===")
for rule in (best, "prior_p0.85|t0_strip", "prior_p0.85"):
    sub = [r for r in rows if r["rule"] == rule]
    print(f"--- {rule} ---")
    for r in sorted(sub, key=lambda x: x["anchor"]):
        print(
            f"  {r['anchor']:<12} F1={r['band_f1']:.3f} prec={r['band_prec']:.3f} "
            f"rec={r['band_rec']:.3f} pred+={r['band_pred_pos_frac']:.3f} gt+={r['band_gt_pos_frac']:.3f}"
        )

best_rows = [r for r in rows if r["rule"] == best]
f1s = [r["band_f1"] for r in best_rows]
print(f"\nbest F1 range: {min(f1s):.3f}-{max(f1s):.3f} stdev={statistics.pstdev(f1s):.3f}")
top_f1 = summary[best]["mean_band_f1"]
close = sum(1 for r in d["summary"] if r["mean_band_f1"] >= top_f1 - 0.01)
print(f"rules within 0.01 of best mean F1: {close} / {len(d['summary'])}")
