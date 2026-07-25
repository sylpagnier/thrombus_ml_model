"""Summarize WC_v7 off-wall 2h limit-analysis probes -> limit_2h_summary.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _metrics(blob: dict) -> dict:
    """Accept probe_firewall style or eval_mat_growth simple.per_anchor / mean."""
    if "metrics" in blob and isinstance(blob["metrics"], dict):
        return blob["metrics"]
    if "simple" in blob and isinstance(blob["simple"], dict):
        pa = blob["simple"].get("per_anchor") or {}
        if isinstance(pa, dict) and pa:
            # single-anchor probe: take first
            row = next(iter(pa.values()))
            if isinstance(row, dict):
                return row
        return blob["simple"].get("mean") or {}
    if "clot" in blob:
        return blob["clot"]
    return blob


def _f(m: dict, *keys: str, default: float = 0.0) -> float:
    for k in keys:
        if k in m and m[k] is not None:
            try:
                return float(m[k])
            except (TypeError, ValueError):
                pass
    return float(default)


def classify(arm_id: str, hop_ge2: float, strict: float, a_hop: float) -> str:
    if arm_id == "A":
        return "baseline"
    if hop_ge2 > a_hop + 0.5 and strict > 0.05:
        return "signal"
    if hop_ge2 > a_hop + 0.5:
        return "weak_volume"
    return "null"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-root",
        default="outputs/biochem/offwall_model/wc_v7_offwall_limit_2h",
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    root = Path(args.run_root)
    if not root.is_absolute():
        root = Path.cwd() / root

    arm_order = ["A", "LumenPush", "FrontierPush", "SkipHopSpec", "BlindSat"]
    rows = []
    a_hop = 0.0
    for arm in arm_order:
        path = root / f"probe_{arm}.json"
        if not path.is_file():
            # also accept eval_*.json
            alt = root / arm / "probe.json"
            path = alt if alt.is_file() else path
        if not path.is_file():
            print(f"[WARN] missing probe for {arm}: {path}", flush=True)
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        # probe_firewall multi-mode: take compound wall or first matching
        if "modes" in raw and isinstance(raw["modes"], list):
            mode = None
            for m in raw["modes"]:
                lab = str(m.get("label") or "")
                if arm == "A" and "wall_alone" in lab:
                    mode = m
                    break
                if arm != "A" and ("compound_wall" in lab or "compound" in lab):
                    mode = m
                    break
            if mode is None and raw["modes"]:
                mode = raw["modes"][-1] if arm != "A" else raw["modes"][0]
            m = _metrics(mode or {})
        else:
            m = _metrics(raw)

        hop = _f(m, "deploy_clot_offwall_n_pred_hop_ge2", "offwall_n_pred_hop_ge2")
        if arm == "A":
            a_hop = hop
        row = {
            "arm": arm,
            "path": str(path),
            "deploy_clot_f1": _f(m, "deploy_clot_f1"),
            "deploy_mat_f1": _f(m, "deploy_mat_f1"),
            "deploy_clot_offwall_n_pred": _f(m, "deploy_clot_offwall_n_pred"),
            "deploy_clot_offwall_n_gt": _f(m, "deploy_clot_offwall_n_gt"),
            "deploy_clot_offwall_n_pred_hop_ge2": hop,
            "deploy_clot_offwall_n_gt_hop_ge2": _f(
                m, "deploy_clot_offwall_n_gt_hop_ge2", "offwall_n_gt_hop_ge2"
            ),
            "deploy_clot_offwall_strict_f1_hop_ge2": _f(
                m, "deploy_clot_offwall_strict_f1_hop_ge2", "offwall_strict_f1_hop_ge2"
            ),
            "deploy_clot_offwall_strict_f1": _f(m, "deploy_clot_offwall_strict_f1"),
        }
        row["verdict"] = classify(
            arm,
            row["deploy_clot_offwall_n_pred_hop_ge2"],
            row["deploy_clot_offwall_strict_f1_hop_ge2"],
            a_hop,
        )
        rows.append(row)

    any_signal = any(r["verdict"] == "signal" for r in rows if r["arm"] != "A")
    any_weak = any(r["verdict"] == "weak_volume" for r in rows if r["arm"] != "A")
    if any_signal:
        next_step = "scale_winning_arm_to_orig10_track1"
    elif any_weak:
        next_step = "keep_inductive_bias_add_localization"
    else:
        next_step = "stop_tuning_design_nonlocal_or_remesh"

    report = {
        "run_root": str(root),
        "arms": rows,
        "any_signal": any_signal,
        "any_weak_volume": any_weak,
        "next_step": next_step,
    }
    out = Path(args.out) if args.out.strip() else root / "limit_2h_summary.json"
    if not out.is_absolute():
        out = Path.cwd() / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 72, flush=True)
    print("OFF-WALL LIMIT 2H SUMMARY", flush=True)
    print("=" * 72, flush=True)
    print(
        f"{'arm':<14} {'clot_f1':>8} {'hop_ge2':>8} {'strict_ge2':>10} {'verdict':<16}",
        flush=True,
    )
    for r in rows:
        print(
            f"{r['arm']:<14} {r['deploy_clot_f1']:8.3f} "
            f"{r['deploy_clot_offwall_n_pred_hop_ge2']:8.1f} "
            f"{r['deploy_clot_offwall_strict_f1_hop_ge2']:10.3f} "
            f"{r['verdict']:<16}",
            flush=True,
        )
    print(f"[i] next_step={next_step}", flush=True)
    print(f"[save] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
