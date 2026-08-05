"""Summarize go_wc_v7_open001_6h ladder probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _pa(probe: dict, anchor: str) -> dict:
    return ((probe.get("simple") or {}).get("per_anchor") or {}).get(anchor) or {}


def _load(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="outputs/biochem/offwall_model/wc_v7_open001_6h")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    root = Path(args.run_root)
    if not root.is_absolute():
        root = Path.cwd() / root

    arms = [
        "A",
        "A_Solo001_Freeze",
        "B_Solo001_Unfreeze",
        "C_Solo001_CC",
        "C2_Teachers_Recall",
        "D_Orig10_Band",
    ]
    rows = []
    for arm in arms:
        probe = _load(root / f"probe_{arm}.json")
        if probe is None:
            continue
        a001 = _pa(probe, "patient001")
        a007 = _pa(probe, "patient007")
        anchors = list(((probe.get("simple") or {}).get("per_anchor") or {}).keys())
        ge2_sum = 0.0
        ge2_gt = 0.0
        f1s = []
        for anc in anchors:
            r = _pa(probe, anc)
            ge2_sum += float(r.get("deploy_clot_offwall_n_pred_hop_ge2", 0) or 0)
            ge2_gt += float(r.get("deploy_clot_offwall_n_gt_hop_ge2", 0) or 0)
            f1s.append(float(r.get("deploy_clot_f1", 0) or 0))
        rows.append(
            {
                "arm": arm,
                "n_anchors": len(anchors),
                "mean_clot_f1": sum(f1s) / max(len(f1s), 1),
                "001_clot_f1": float(a001.get("deploy_clot_f1", 0) or 0),
                "001_hop_ge2": float(a001.get("deploy_clot_offwall_n_pred_hop_ge2", 0) or 0),
                "001_hop_ge2_gt": float(a001.get("deploy_clot_offwall_n_gt_hop_ge2", 0) or 0),
                "007_hop_ge2": float(a007.get("deploy_clot_offwall_n_pred_hop_ge2", 0) or 0),
                "cohort_hop_ge2_pred": ge2_sum,
                "cohort_hop_ge2_gt": ge2_gt,
            }
        )

    opened = any(r["001_hop_ge2"] > 0.5 for r in rows if r["arm"] != "A")
    best = None
    for r in rows:
        if r["arm"] == "A":
            continue
        if r["001_hop_ge2"] > 0.5:
            if best is None or r["001_hop_ge2"] > best["001_hop_ge2"]:
                best = r
            elif (
                best is not None
                and r["001_hop_ge2"] == best["001_hop_ge2"]
                and r["mean_clot_f1"] > best["mean_clot_f1"]
            ):
                best = r

    if opened and best is not None and best["mean_clot_f1"] >= 0.70:
        verdict = "pass_001_open_healthy_compound"
    elif opened:
        verdict = "pass_001_open_wall_check"
    else:
        verdict = "still_closed_001"

    state = _load(root / "open001_6h_state.json") or {}
    out = {
        "verdict": verdict,
        "opened_001": opened,
        "best_arm": None if best is None else best["arm"],
        "rows": rows,
        "state": state,
    }
    out_path = Path(args.out) if args.out else root / "compare_open001_6h.json"
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"[i] verdict={verdict} opened_001={opened} best={out['best_arm']}", flush=True)
    print(
        f"{'arm':<22} {'f1':>6} {'001_ge2':>8} {'007_ge2':>8} {'cohort_ge2':>10}",
        flush=True,
    )
    for r in rows:
        print(
            f"{r['arm']:<22} {r['mean_clot_f1']:6.3f} {r['001_hop_ge2']:8.1f} "
            f"{r['007_hop_ge2']:8.1f} {r['cohort_hop_ge2_pred']:10.1f}",
            flush=True,
        )
    print(f"[save] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
