"""Summarize lumen recall-limit 2h: A vs Prec8hRef vs RecallPush on probe anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

KEYS = (
    "deploy_clot_f1",
    "deploy_clot_offwall_n_pred_hop_ge2",
    "deploy_clot_offwall_n_gt_hop_ge2",
    "deploy_clot_offwall_strict_f1_hop_ge2",
)


def _load(path: Path, label: str) -> dict | None:
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    simple = report.get("simple") or {}
    mean = dict(simple.get("mean") or {})
    per = dict(simple.get("per_anchor") or {})
    return {
        "label": label,
        "path": str(path),
        "mean": {k: float(mean.get(k, 0.0) or 0.0) for k in KEYS},
        "per_anchor": {
            a: {k: float((m or {}).get(k, 0.0) or 0.0) for k in KEYS}
            for a, m in per.items()
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-root",
        default="outputs/biochem/offwall_model/wc_v7_lumen_recall_limit_2h",
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = Path(args.run_root)
    if not root.is_absolute():
        root = Path.cwd() / root

    arms = {}
    for name in ("A", "Prec8hRef", "RecallPush", "Open001"):
        row = _load(root / f"probe_{name}.json", name)
        if row is None:
            print(f"[WARN] missing probe_{name}.json", flush=True)
            continue
        arms[name] = row

    # Alias: open001_1h launcher writes probe_Open001.json
    if "Open001" in arms and "RecallPush" not in arms:
        arms["RecallPush"] = dict(arms["Open001"])
        arms["RecallPush"]["label"] = "RecallPush"
    a = arms.get("A")
    prec = arms.get("Prec8hRef")
    rec = arms.get("RecallPush")

    def _ge2(arm: dict | None, anc: str) -> tuple[float, float, float]:
        if arm is None:
            return 0.0, 0.0, 0.0
        m = (arm.get("per_anchor") or {}).get(anc) or {}
        return (
            float(m.get("deploy_clot_offwall_n_gt_hop_ge2", 0) or 0),
            float(m.get("deploy_clot_offwall_n_pred_hop_ge2", 0) or 0),
            float(m.get("deploy_clot_offwall_strict_f1_hop_ge2", 0) or 0),
        )

    # Limit-analysis verdict focused on 001 opening + 007 recall
    gt001, pr_prec_001, st_prec_001 = _ge2(prec, "patient001")
    _, pr_rec_001, st_rec_001 = _ge2(rec, "patient001")
    gt007, pr_prec_007, st_prec_007 = _ge2(prec, "patient007")
    _, pr_rec_007, st_rec_007 = _ge2(rec, "patient007")
    _, pr_rec_004, _ = _ge2(rec, "patient004")
    _, pr_rec_008, _ = _ge2(rec, "patient008")

    opened_001 = pr_rec_001 > 0.5 and (prec is None or pr_prec_001 <= 0.5)
    recall_up_007 = pr_rec_007 > (pr_prec_007 + 2.0)
    spray = (pr_rec_004 > 2.0) or (pr_rec_008 > 5.0)

    if opened_001 and recall_up_007 and not spray:
        verdict = "capacity_yes_tuning_headroom"
    elif opened_001 or recall_up_007:
        verdict = "partial_capacity_mixed_spray" if spray else "partial_capacity"
    elif spray and not opened_001:
        verdict = "recall_push_sprays_no_001"
    else:
        verdict = "null_architecture_suspect"

    print("=" * 78, flush=True)
    print("LUMEN RECALL LIMIT 2H", flush=True)
    print("=" * 78, flush=True)
    print(f"{'arm':<12} {'clot_f1':>8} {'ge2_n':>8} {'ge2_gt':>8} {'strict':>8}", flush=True)
    for name, row in arms.items():
        m = row["mean"]
        print(
            f"{name:<12} {m['deploy_clot_f1']:8.3f} "
            f"{m['deploy_clot_offwall_n_pred_hop_ge2']:8.1f} "
            f"{m['deploy_clot_offwall_n_gt_hop_ge2']:8.1f} "
            f"{m['deploy_clot_offwall_strict_f1_hop_ge2']:8.3f}",
            flush=True,
        )
    print("\nPer-anchor hop_ge2 pred (Prec8hRef -> RecallPush):", flush=True)
    for anc in ("patient001", "patient007", "patient004", "patient008"):
        _, pp, sp = _ge2(prec, anc)
        _, pr, sr = _ge2(rec, anc)
        gt, _, _ = _ge2(a if a else prec, anc)
        if gt == 0 and a:
            gt, _, _ = _ge2(a, anc)
        print(
            f"  {anc}: gt={gt:.0f}  prec={pp:.0f} (s={sp:.3f})  recall={pr:.0f} (s={sr:.3f})",
            flush=True,
        )
    print(
        f"[i] opened_001={opened_001} recall_up_007={recall_up_007} spray={spray} "
        f"-> verdict={verdict}",
        flush=True,
    )

    report = {
        "arms": {k: {"mean": v["mean"], "per_anchor": v["per_anchor"]} for k, v in arms.items()},
        "gates": {
            "opened_001": opened_001,
            "recall_up_007": recall_up_007,
            "spray_004_or_008": spray,
            "prec_001": pr_prec_001,
            "rec_001": pr_rec_001,
            "prec_007": pr_prec_007,
            "rec_007": pr_rec_007,
        },
        "verdict": verdict,
    }
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
