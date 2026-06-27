"""Summarize the A/B/C/D precision ladder (deploy clot F1, deploy_frozen).

Reads each leg's ``<leg>/eval/deploy_ab_eval.json`` (from eval_biochem_gnn_deploy_ab.py) and prints
a per-leg table for F1, precision, recall, guiding. The whole point is precision: a lever "wins"
only if it raises F1 (precision up) WITHOUT collapsing recall.

NOTE on columns: the default go_species_abcd recipe trains on ALL 6 anchors (moves234 convention,
so ``score_clot_w`` can distinguish leg B/D selection from A/C). Hence "p007" and "rest" are both
in-sample -- this is a controlled *relative* lever comparison, not a generalization claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LEG_ORDER = ["A_baseline", "B_footprint_sup", "C_geom_feats", "D_both"]


def _load_leg(leg_dir: Path) -> list[dict] | None:
    p = leg_dir / "eval" / "deploy_ab_eval.json"
    if not p.is_file():
        return None
    payload = json.loads(p.read_text(encoding="utf-8"))
    return [r for r in (payload.get("rows") or []) if r.get("mode") == "deploy_frozen"]


def _agg(rows: list[dict], val_anchor: str, key: str) -> tuple[float, float]:
    """Return (val-anchor value, mean over the other in-train anchors) for a clot_*_main key."""
    p007 = next((float(r[key]) for r in rows if r["anchor"] == val_anchor and r.get(key) == r.get(key)), float("nan"))
    hold = [float(r[key]) for r in rows if r["anchor"] != val_anchor and r.get(key) == r.get(key)]
    hmean = sum(hold) / len(hold) if hold else float("nan")
    return p007, hmean


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="outputs/biochem/biochem_gnn/abcd_precision")
    ap.add_argument("--val-anchor", default="patient007")
    args = ap.parse_args()

    root = Path(args.run_root)
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()

    legs = {}
    for leg in LEG_ORDER:
        rows = _load_leg(root / leg)
        if rows:
            legs[leg] = rows
    for d in sorted(root.glob("*")):
        if d.is_dir() and d.name not in legs:
            rows = _load_leg(d)
            if rows:
                legs[d.name] = rows

    if not legs:
        print(f"[ERR] no leg eval json found under {root}", flush=True)
        return 1

    va = args.val_anchor
    print("\n==================== A/B/C/D PRECISION LADDER (deploy clot F1) ====================", flush=True)
    hdr = f"  {'leg':<18}{'F1 p007':>9}{'F1 rest':>9}{'prec rest':>11}{'rec rest':>10}{'guid p007':>11}"
    print(hdr, flush=True)
    print("  " + "-" * (len(hdr) - 2), flush=True)

    base_hold = None
    ordered = [l for l in LEG_ORDER if l in legs] + [l for l in legs if l not in LEG_ORDER]
    rowvals = {}
    for leg in ordered:
        rows = legs[leg]
        f1_p, f1_h = _agg(rows, va, "clot_f1_main")
        pr_p, pr_h = _agg(rows, va, "clot_prec_main")
        rc_p, rc_h = _agg(rows, va, "clot_rec_main")
        g_p, g_h = _agg(rows, va, "clot_guiding_main")
        rowvals[leg] = (f1_p, f1_h, pr_h, rc_h, g_p)
        if leg == "A_baseline":
            base_hold = f1_h
        print(f"  {leg:<18}{f1_p:>9.3f}{f1_h:>9.3f}{pr_h:>11.3f}{rc_h:>10.3f}{g_p:>11.3f}", flush=True)

    print("  " + "-" * (len(hdr) - 2), flush=True)
    if base_hold is not None and base_hold == base_hold:
        print(f"\n  delta rest-mean F1 vs A_baseline ({base_hold:.3f}):", flush=True)
        for leg in ordered:
            if leg == "A_baseline":
                continue
            f1_h = rowvals[leg][1]
            if f1_h == f1_h:
                print(f"    {leg:<18} {f1_h - base_hold:+.3f}", flush=True)

    print("\n  READ:", flush=True)
    print("    B>A (rec held) -> footprint FP-supervision is a real precision lever.", flush=True)
    print("    C>A            -> non-flow geometry discriminates wall FP from TP.", flush=True)
    print("    D ~ max(B,C)   -> levers stack; else they overlap / interfere.", flush=True)
    print("    all ~= A       -> precision is Mat-capacity bound (epochs/data/scope), not these.", flush=True)
    print("==================================================================================", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
