"""Pick best rule from clot_prior_rule_sweep.json."""
import json
from pathlib import Path

d = json.loads(Path("outputs/biochem/diagnostics/clot_prior_rule_sweep.json").read_text())
summary = sorted(d["summary"], key=lambda r: (-r["score"], -r["mean_band_f1"], -r["mean_band_prec"]))
print("TOP 15 by score:")
hdr = f"{'rule':<55} {'F1':>6} {'prec':>6} {'rec':>6} {'pred+':>6} {'score':>6}"
print(hdr)
for r in summary[:15]:
    print(
        f"{r['rule']:<55} {r['mean_band_f1']:>6.3f} {r['mean_band_prec']:>6.3f} "
        f"{r['mean_band_rec']:>6.3f} {r['mean_pred_pos_frac']:>6.3f} {r['score']:>6.3f}"
    )

rows = d["per_anchor"]
p7 = sorted([r for r in rows if r["anchor"] == "patient007"], key=lambda r: (-r["band_f1"], -r["band_prec"]))
print("\nTOP 10 patient007 by F1:")
for r in p7[:10]:
    print(
        f"  {r['rule']:<50} F1={r['band_f1']:.3f} prec={r['band_prec']:.3f} "
        f"rec={r['band_rec']:.3f} pred+={r['band_pred_pos_frac']:.3f}"
    )

stag = sorted([r for r in summary if "flux_stag" in r["rule"]], key=lambda r: -r["score"])[:8]
print("\nBest flux_stag rules (mean):")
for r in stag:
    print(f"  {r['rule']:<50} F1={r['mean_band_f1']:.3f} pred+={r['mean_pred_pos_frac']:.3f}")

winner = summary[0]
print(f"\nWINNER: {winner['rule']} score={winner['score']:.3f}")
