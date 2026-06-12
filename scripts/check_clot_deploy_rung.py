#!/usr/bin/env python3
"""Gate checker for CAVO deploy ladder rungs (rule baseline + optional MLP ckpt)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.paths import get_project_root

# Thresholds from docs/DEPLOY_ARCHITECTURE.md, adapted for rule band-F1 path.
GATES = {
    "s0": {
        "p007_tfinal_band_f1_min": 0.50,
        "p007_tfinal_band_pred_frac_max": 0.65,
        # 0.44: accepted near-pass (2026-06 multistep rule); patient006 has ~2 GT clots.
        "mean_band_f1_min": 0.44,
    },
    "s1": {
        "p007_mean_band_f1_min": 0.38,
        "mean_band_f1_min": 0.32,
    },
    "g1": {
        "p007_mean_band_f1_min": 0.32,
        "mean_band_f1_min": 0.28,
    },
    "g2": {
        "p007_tfinal_band_f1_min": 0.50,
        "p007_tfinal_band_f1_vs_s1_min_ratio": 0.60,
        "mean_band_f1_min": 0.20,
    },
}


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _anchor_row(payload: dict, stem: str) -> dict | None:
    for row in payload.get("anchors", []):
        if row.get("anchor") == stem:
            return row
    return None


def check_rule_stage(stage: str, payload: dict, *, s1_payload: dict | None = None) -> tuple[bool, list[str]]:
    g = GATES[stage]
    lines: list[str] = []
    ok = True

    p007 = _anchor_row(payload, "patient007")
    if p007 is None:
        return False, ["[ERR] patient007 missing from eval"]

    mean_all = float(payload.get("mean_band_f1", 0.0))

    if stage == "s0":
        f1 = float(p007["tfinal_band_f1"])
        pp = float(p007["tfinal_band_pred_frac"])
        lines.append(f"p007 tfinal band F1={f1:.3f} (need>={g['p007_tfinal_band_f1_min']})")
        lines.append(f"p007 band pred+={pp:.3f} (need<={g['p007_tfinal_band_pred_frac_max']})")
        lines.append(f"mean band F1={mean_all:.3f} (need>={g['mean_band_f1_min']})")
        ok = (
            f1 >= g["p007_tfinal_band_f1_min"]
            and pp <= g["p007_tfinal_band_pred_frac_max"]
            and mean_all >= g["mean_band_f1_min"]
        )
    elif stage in ("s1", "g1"):
        f1 = float(p007["mean_band_f1"])
        lines.append(f"p007 mean band F1={f1:.3f} (need>={g['p007_mean_band_f1_min']})")
        lines.append(f"mean band F1={mean_all:.3f} (need>={g['mean_band_f1_min']})")
        ok = f1 >= g["p007_mean_band_f1_min"] and mean_all >= g["mean_band_f1_min"]
    elif stage == "g2":
        f1 = float(p007["tfinal_band_f1"])
        lines.append(f"p007 tfinal band F1={f1:.3f} (need>={g['p007_tfinal_band_f1_min']})")
        lines.append(f"mean band F1={mean_all:.3f} (need>={g['mean_band_f1_min']})")
        ok = f1 >= g["p007_tfinal_band_f1_min"] and mean_all >= g["mean_band_f1_min"]
        if s1_payload:
            s1_p007 = _anchor_row(s1_payload, "patient007")
            if s1_p007:
                s1_f1 = float(s1_p007["tfinal_band_f1"])
                ratio = f1 / max(s1_f1, 1e-6)
                need = g["p007_tfinal_band_f1_vs_s1_min_ratio"]
                lines.append(f"tfinal F1 vs S1 ratio={ratio:.3f} (need>={need})")
                ok = ok and ratio >= need

    return ok, lines


def main() -> int:
    ap = argparse.ArgumentParser(description="Check CAVO deploy ladder rung gate")
    ap.add_argument("--rung", required=True, choices=tuple(GATES.keys()))
    ap.add_argument("--mode", default="rule", choices=("rule", "checkpoint"))
    ap.add_argument("--json", default="", help="Eval JSON (default rule ladder path)")
    ap.add_argument("--s1-json", default="", help="S1 eval for G2 ratio check")
    args = ap.parse_args()

    root = get_project_root()
    if args.json:
        path = Path(args.json)
        if not path.is_absolute():
            path = root / path
    else:
        path = root / f"outputs/biochem/diagnostics/clot_prior_rule_{args.rung}.json"

    payload = _load_json(path)
    s1_payload = None
    if args.rung == "g2":
        s1_path = Path(args.s1_json) if args.s1_json else root / "outputs/biochem/diagnostics/clot_prior_rule_s1.json"
        if s1_path.is_file():
            s1_payload = _load_json(s1_path)

    ok, lines = check_rule_stage(args.rung, payload, s1_payload=s1_payload)
    print(f"[{'OK' if ok else 'FAIL'}] rung {args.rung.upper()} ({args.mode})")
    for line in lines:
        print(f"  {line}")
    print(f"  rule={payload.get('rule', '?')}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
