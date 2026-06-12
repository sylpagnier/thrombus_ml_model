"""Extract per-anchor metrics for selected rules from refined sweep JSON."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
path = REPO / "outputs/biochem/diagnostics/clot_prior_rule_sweep_refined.json"
data = json.loads(path.read_text())
rules = [
    "prior_p0.80",
    "prior_p0.80|flux_stag_top20|tie_dx_hop",
]
for rule in rules:
    rows = [r for r in data["per_anchor"] if r["rule"] == rule]
    if not rows:
        print(f"{rule}: NO ROWS")
        continue
    mf1 = sum(r["band_f1"] for r in rows) / len(rows)
    print(f"=== {rule} n={len(rows)} mean_F1={mf1:.3f}")
    for r in sorted(rows, key=lambda x: x["anchor"]):
        print(
            f"  {r['anchor']}: F1={r['band_f1']:.3f} prec={r['band_prec']:.3f} "
            f"rec={r['band_rec']:.3f} pred+={r['band_pred_pos_frac']:.3f} flag={r['n_flag']}"
        )
