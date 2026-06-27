"""Consolidate the clot-flow gate ladder (per-rung JSON from compare_coupled_mat_rollout).

Reads every ``<rung>.json`` written by ``go_clot_flow_gate_ladder.ps1`` and prints one table plus
the decisive contrasts (leash bite, static vs dynamic ceiling, corrector realized, z_kin channel).
ASCII only (Windows PowerShell console).

    python scripts/summarize_clot_flow_gate.py --dir outputs/biochem/corrector_coupling/gate_ladder
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# rung filename stem -> (order, label, which F1 is the headline: "coupled" or "baseline")
RUNGS = {
    "ref": (0, "ref baseline (flow-active, kine)", "baseline"),
    "ablate": (1, "leash check: flow ABLATED", "baseline"),
    "gt_static": (2, "#5  static GT ceiling", "coupled"),
    "gt_dynamic": (3, "#5c dynamic GT ceiling", "coupled"),
    "oracle_mu": (4, "#5b oracle-mu (true clot loc)", "coupled"),
    "corrector": (5, "#6a corrector, frozen z_kin", "coupled"),
    "corrector_resolve": (6, "#6b corrector + z_kin re-solve", "coupled"),
}


def _load(d: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for stem in RUNGS:
        p = d / f"{stem}.json"
        if p.is_file():
            try:
                out[stem] = json.loads(p.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] could not read {p}: {exc}")
    return out


def _headline(stem: str, r: dict) -> float:
    which = RUNGS[stem][2]
    return float(r.get("t_last_coupled_f1" if which == "coupled" else "t_last_baseline_f1", 0.0))


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize the clot-flow gate ladder JSONs.")
    ap.add_argument("--dir", required=True, help="directory of per-rung <rung>.json files")
    args = ap.parse_args()
    d = Path(args.dir)
    runs = _load(d)
    if not runs:
        print(f"[ERR] no rung JSONs found under {d}")
        return 1

    ckpt = next(iter(runs.values())).get("species_ckpt", "?")
    print("\n==================== CLOT-FLOW GATE LADDER ====================")
    print(f"  ckpt: {ckpt}")
    print(f"  dir : {d}")
    print(f"  {'rung':<34} | {'headline F1':>11} | {'mode':>16} | {'max|div|':>9} | {'zkin':>7}")
    print("  " + "-" * 88)
    ordered = sorted(runs.items(), key=lambda kv: RUNGS[kv[0]][0])
    vals: dict[str, float] = {}
    for stem, r in ordered:
        label = RUNGS[stem][1]
        f1 = _headline(stem, r)
        vals[stem] = f1
        cpl = r.get("coupling", {}) or {}
        mode = str(cpl.get("final_mode", "-"))
        mdiv = float(cpl.get("max_abs_diversion_nd", 0.0) or 0.0)
        zkin = "resolved" if cpl.get("kine_resolved") else "frozen"
        print(f"  {label:<34} | {f1:>11.3f} | {mode:>16} | {mdiv:>9.2e} | {zkin:>7}")
    print("  " + "-" * 88)

    # Decisive contrasts (only print the ones we have data for).
    ref = vals.get("ref")
    print("\n  CONTRASTS (vs ref baseline):")
    if ref is not None and "ablate" in vals:
        bite = ref - vals["ablate"]
        tag = "leash BIT (model reads flow)" if bite > 0.02 else "leash WEAK (flow ignored)"
        print(f"    leash bite (ref - ablate)        = {bite:+.3f}   [{tag}]")
    if ref is not None and "gt_static" in vals:
        print(f"    static ceiling  (#5  - ref)      = {vals['gt_static'] - ref:+.3f}")
    if ref is not None and "gt_dynamic" in vals:
        print(f"    dynamic ceiling (#5c - ref)      = {vals['gt_dynamic'] - ref:+.3f}")
    if "gt_dynamic" in vals and "gt_static" in vals:
        head = vals["gt_dynamic"] - vals["gt_static"]
        tag = "TEMPORAL SIGNAL REAL" if head > 0.02 else "no temporal headroom"
        print(f"    Trap C headroom (#5c - #5)       = {head:+.3f}   [{tag}]")
    if ref is not None and "oracle_mu" in vals:
        print(f"    oracle-mu       (#5b - ref)      = {vals['oracle_mu'] - ref:+.3f}")
    if ref is not None and "corrector" in vals:
        print(f"    corrector real  (#6a - ref)      = {vals['corrector'] - ref:+.3f}")
    if "corrector" in vals and "corrector_resolve" in vals:
        print(f"    z_kin re-solve  (#6b - #6a)      = {vals['corrector_resolve'] - vals['corrector']:+.3f}")
    if "gt_static" in vals and "oracle_mu" in vals and "corrector" in vals:
        print("\n  READ: gt-flow >> oracle-mu ~ corrector  -> corrector is the bottleneck (not the teacher).")
        print("        gt-flow ~ oracle-mu but corrector << -> predicted clot location is the bottleneck.")
    print("===============================================================\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
