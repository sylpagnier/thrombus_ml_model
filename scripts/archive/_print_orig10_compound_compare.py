"""Print quantitative A/B/C tables for orig10 compound run."""
from __future__ import annotations

import json
from pathlib import Path

root = Path("outputs/biochem/offwall_model/wc_v7_compound_abc_orig10_9h")
files = {
    "A": root / "eval_A_canonical.json",
    "B": root / "eval_B_compound_frontier.json",
    "C": root / "eval_C_compound_wall_prec.json",
}
keys = [
    "deploy_mat_f1",
    "deploy_clot_f1",
    "deploy_clot_score",
    "deploy_clot_offwall_relaxed_f1",
    "deploy_clot_offwall_strict_f1",
    "deploy_clot_offwall_n_pred",
    "deploy_clot_offwall_n_gt",
]
reports = {k: json.loads(p.read_text(encoding="utf-8")) for k, p in files.items()}
mean = {k: reports[k]["simple"]["mean"] for k in reports}
per = {k: reports[k]["simple"]["per_anchor"] for k in reports}

print("=== COHORT MEAN ===")
print(f"{'metric':<34} {'A':>8} {'B':>8} {'C':>8} {'B-A':>8} {'C-A':>8}")
for key in keys:
    a, b, c = mean["A"][key], mean["B"][key], mean["C"][key]
    print(f"{key:<34} {a:8.4f} {b:8.4f} {c:8.4f} {b-a:+8.4f} {c-a:+8.4f}")

sample = ["patient001", "patient007", "patient004"]
print("\n=== SAMPLE ANCHORS (viz trio) ===")
print(
    f"{'anchor':<12} {'arm':<3} {'clot_f1':>8} {'score':>8} "
    f"{'ow_rel':>8} {'ow_str':>8} {'n_pred':>7} {'n_gt':>7}"
)
for anc in sample:
    for arm in "ABC":
        m = per[arm][anc]
        print(
            f"{anc:<12} {arm:<3} {m['deploy_clot_f1']:8.4f} {m['deploy_clot_score']:8.4f} "
            f"{m['deploy_clot_offwall_relaxed_f1']:8.4f} {m['deploy_clot_offwall_strict_f1']:8.4f} "
            f"{m['deploy_clot_offwall_n_pred']:7.1f} {m['deploy_clot_offwall_n_gt']:7.1f}"
        )
    print()

print("=== COHORT WINNERS ===")
for key in keys:
    vals = {a: mean[a][key] for a in "ABC"}
    best = max(vals, key=vals.get)
    print(f"{key}: {best} ({vals[best]:.4f})")
