"""Summarize crack-001 ladder: which hypothesis opened patient001 lumen."""

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

CRACK_ARMS = ("Solo001_Freeze", "Solo001_Unfreeze", "Solo001_CC")
REF_ARMS = ("A", "Prec8hRef")


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


def _ge2(arm: dict | None, anc: str) -> tuple[float, float, float]:
    if arm is None:
        return 0.0, 0.0, 0.0
    m = (arm.get("per_anchor") or {}).get(anc) or {}
    return (
        float(m.get("deploy_clot_offwall_n_gt_hop_ge2", 0) or 0),
        float(m.get("deploy_clot_offwall_n_pred_hop_ge2", 0) or 0),
        float(m.get("deploy_clot_offwall_strict_f1_hop_ge2", 0) or 0),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-root",
        default="outputs/biochem/offwall_model/wc_v7_crack_001_3h",
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = Path(args.run_root)
    if not root.is_absolute():
        root = Path.cwd() / root

    arms: dict[str, dict] = {}
    for name in REF_ARMS + CRACK_ARMS:
        row = _load(root / f"probe_{name}.json", name)
        if row is None:
            print(f"[WARN] missing probe_{name}.json", flush=True)
            continue
        arms[name] = row

    def opened(name: str) -> bool:
        _, n_pred, _ = _ge2(arms.get(name), "patient001")
        return n_pred > 0.5

    first_open = next((a for a in CRACK_ARMS if opened(a)), None)
    # Spray: check worst among crack arms that actually ran (not just first_open).
    spray = False
    spray_n004 = 0.0
    spray_n008 = 0.0
    for name in CRACK_ARMS:
        if name not in arms:
            continue
        _, n004, _ = _ge2(arms.get(name), "patient004")
        _, n008, _ = _ge2(arms.get(name), "patient008")
        spray_n004 = max(spray_n004, n004)
        spray_n008 = max(spray_n008, n008)
        if n004 > 2.0 or n008 > 5.0:
            spray = True

    if first_open == "Solo001_Freeze":
        verdict = "opened_by_competition_fix"
    elif first_open == "Solo001_Unfreeze":
        verdict = "opened_by_unfreeze"
    elif first_open == "Solo001_CC":
        verdict = "opened_by_cc_tiles"
    elif any(a in arms for a in CRACK_ARMS):
        verdict = "still_closed_architecture_suspect"
    else:
        verdict = "incomplete_missing_crack_probes"

    print("=" * 78, flush=True)
    print("CRACK 001 LUMEN LOCK", flush=True)
    print("=" * 78, flush=True)
    print(f"{'arm':<18} {'clot_f1':>8} {'ge2_n':>8} {'ge2_gt':>8} {'strict':>8}", flush=True)
    for name in list(REF_ARMS) + list(CRACK_ARMS):
        row = arms.get(name)
        if row is None:
            continue
        m = row["mean"]
        print(
            f"{name:<18} {m['deploy_clot_f1']:8.3f} "
            f"{m['deploy_clot_offwall_n_pred_hop_ge2']:8.1f} "
            f"{m['deploy_clot_offwall_n_gt_hop_ge2']:8.1f} "
            f"{m['deploy_clot_offwall_strict_f1_hop_ge2']:8.3f}",
            flush=True,
        )

    print("\nPer-anchor hop_ge2 pred (patient001 gate):", flush=True)
    for anc in ("patient001", "patient007", "patient004", "patient008"):
        bits = []
        gt = 0.0
        for name in list(REF_ARMS) + list(CRACK_ARMS):
            g, n, s = _ge2(arms.get(name), anc)
            if g > 0:
                gt = g
            if name in arms:
                bits.append(f"{name}={n:.0f}(s={s:.3f})")
        print(f"  {anc}: gt={gt:.0f}  " + "  ".join(bits), flush=True)

    print(
        f"[i] first_open={first_open} spray={spray} -> verdict={verdict}",
        flush=True,
    )

    report = {
        "arms": {k: {"mean": v["mean"], "per_anchor": v["per_anchor"]} for k, v in arms.items()},
        "gates": {
            "first_open_arm": first_open,
            "opened_Solo001_Freeze": opened("Solo001_Freeze"),
            "opened_Solo001_Unfreeze": opened("Solo001_Unfreeze"),
            "opened_Solo001_CC": opened("Solo001_CC"),
            "spray_004_or_008": spray,
            "max_hop_ge2_pred_004": spray_n004,
            "max_hop_ge2_pred_008": spray_n008,
        },
        "verdict": verdict,
        "hypotheses": {
            "H1_competition": "Solo001_Freeze opens 001 -> 007/010 mix stole gradients",
            "H2_backbone_lock": "only Unfreeze opens -> WC_v7 features need update",
            "H3_tile_density": "only CC opens -> union undersamples compact 001 lumen",
            "architecture": "all closed -> inductive bias / route / IC limit",
        },
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
