"""Summarize WC_v7 canonical vs compound growth A/B/C eval JSONs.

Usage::

    python scripts/summarize_wc_v7_compound_ab.py \\
        --arm-a outputs/.../eval_A_canonical.json \\
        --arm-b outputs/.../eval_B_compound_frontier.json \\
        --arm-c outputs/.../eval_C_compound_wall_prec.json \\
        --out   outputs/.../compare_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = (
    "deploy_mat_f1",
    "deploy_clot_f1",
    "deploy_clot_score",
    "deploy_clot_offwall_relaxed_f1",
    "deploy_clot_offwall_strict_f1",
    "deploy_clot_offwall_n_pred",
    "deploy_clot_offwall_n_gt",
)


def _mean(report: dict) -> dict:
    simple = report.get("simple") or {}
    return dict(simple.get("mean") or {})


def _load_arm(path: str, label: str) -> dict | None:
    if not path.strip():
        return None
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)
    report = json.loads(p.read_text(encoding="utf-8"))
    m = _mean(report)
    return {
        "label": label,
        "path": str(p),
        "two_model": report.get("two_model"),
        "mean": {k: m.get(k, 0.0) for k in METRICS},
        "_raw_mean": m,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize WC_v7 compound A/B/C")
    ap.add_argument("--arm-a", required=True, help="Canonical-only eval JSON")
    ap.add_argument("--arm-b", default="", help="Compound frontier-route eval JSON")
    ap.add_argument("--arm-c", default="", help="Best-practice wall-route+prec eval JSON")
    ap.add_argument("--arm-b2", default="", help="Deprecated alias; use --arm-c")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    arm_c_path = args.arm_c.strip() or args.arm_b2.strip()
    a = _load_arm(args.arm_a, "canonical_WC_v7")
    assert a is not None
    b = _load_arm(args.arm_b, "compound_frontier") if args.arm_b.strip() else None
    c = _load_arm(arm_c_path, "compound_wall_prec") if arm_c_path else None
    if b is None and c is None:
        raise SystemExit("Provide at least one of --arm-b or --arm-c")

    ma = a["_raw_mean"]
    summary: dict = {
        "arm_a": {k: v for k, v in a.items() if k != "_raw_mean"},
    }
    if b is not None:
        mb = b["_raw_mean"]
        summary["arm_b"] = {k: v for k, v in b.items() if k != "_raw_mean"}
        summary["delta_B_minus_A"] = {k: mb.get(k, 0.0) - ma.get(k, 0.0) for k in METRICS}
    if c is not None:
        mc = c["_raw_mean"]
        summary["arm_c"] = {k: v for k, v in c.items() if k != "_raw_mean"}
        summary["delta_C_minus_A"] = {k: mc.get(k, 0.0) - ma.get(k, 0.0) for k in METRICS}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 88)
    print("WC_v7 CANONICAL vs COMPOUND GROWTH A/B/C")
    print("=" * 88)
    header = f"{'metric':<32} {'A canon':>10}"
    if b is not None:
        header += f" {'B front':>10} {'dB-A':>10}"
    if c is not None:
        header += f" {'C wall+':>10} {'dC-A':>10}"
    print(header)
    for k in METRICS:
        va = ma.get(k, 0.0)
        row = f"{k:<32} {va:10.4f}"
        if b is not None:
            vb = b["_raw_mean"].get(k, 0.0)
            row += f" {vb:10.4f} {vb - va:+10.4f}"
        if c is not None:
            vc = c["_raw_mean"].get(k, 0.0)
            row += f" {vc:10.4f} {vc - va:+10.4f}"
        print(row)
    print(f"[save] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
