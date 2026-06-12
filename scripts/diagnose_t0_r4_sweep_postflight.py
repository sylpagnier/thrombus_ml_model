"""Aggregate sweep leg diagnostics + preflight into morning triage report.

Usage::

    python scripts/diagnose_t0_r4_sweep_postflight.py
    python scripts/diagnose_t0_r4_sweep_postflight.py --sweep-dir outputs/biochem/sweep_t0_r4_arch_6h
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.core_physics.t0_r4_sweep import DEFAULT_SWEEP_ORDER, RECIPES
from src.utils.paths import get_project_root


def _load(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> int:
    ap = argparse.ArgumentParser(description="T0 Rung4 sweep postflight aggregate")
    ap.add_argument("--sweep-dir", default="outputs/biochem/sweep_t0_r4_arch_6h")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = get_project_root()
    sweep_dir = root / args.sweep_dir
    preflight = _load(sweep_dir / "preflight.json")
    summary = _load(sweep_dir / "summary.json")

    legs: list[dict] = []
    for leg_id in DEFAULT_SWEEP_ORDER:
        leg_dir = sweep_dir / leg_id
        if not leg_dir.is_dir():
            continue
        diag = _load(leg_dir / f"diagnostic_{args.anchor}.json")
        eval_j = _load(leg_dir / f"eval_{args.anchor}.json")
        if diag is None and eval_j is None and leg_id != "ref_s0":
            continue
        row: dict = {"leg_id": leg_id}
        if leg_id in RECIPES:
            row["family"] = RECIPES[leg_id].family
            row["hypothesis"] = RECIPES[leg_id].hypothesis
        if diag:
            row.update({
                "verdict": diag.get("verdict"),
                "final_f1_delta": diag.get("vs_s0", {}).get("final_f1_delta"),
                "fi_mae_improve": diag.get("vs_s0", {}).get("final_fi_mae_improve"),
                "mat_mae_improve": diag.get("vs_s0", {}).get("final_mat_mae_improve"),
                "identical_to_s0": diag.get("vs_s0", {}).get("identical_to_s0"),
                "fn_fixed_final": diag.get("timeline", [{}])[-1].get("localization", {}).get("fn_fixed"),
                "fp_fixed_final": diag.get("timeline", [{}])[-1].get("localization", {}).get("fp_fixed"),
            })
        if eval_j:
            health = eval_j.get("rollout_health", {})
            clot = eval_j.get("sweep_leg", eval_j.get("rung4_step", {})).get("clot_nucleation", [])
            row["eval_f1"] = float(clot[-1]["clot_f1"]) if clot else None
            row["health_pass"] = health.get("health_pass")
            row["wall_carpet"] = health.get("wall_carpet")
        legs.append(row)

    s0_f1 = None
    if preflight:
        s0_f1 = preflight.get("targets", {}).get("s0_baseline_f1")
    if s0_f1 is None and summary:
        s0_f1 = summary.get("s0_baseline_f1", 0.408)

    def _score(r: dict) -> float:
        return float(r.get("eval_f1") or r.get("final_f1_delta", -999) + (s0_f1 or 0))

    legs.sort(key=_score, reverse=True)

    promising = [
        r for r in legs
        if r.get("leg_id") not in ("ref_s0", "smoke_s4")
        and not r.get("identical_to_s0")
        and float(r.get("final_f1_delta") or -1) > 0.005
        and not r.get("wall_carpet")
    ]
    species_only = [
        r for r in legs
        if r.get("identical_to_s0") is False
        and float(r.get("final_f1_delta") or 0) <= 0.005
        and (float(r.get("fi_mae_improve") or 0) > 1e-5 or float(r.get("mat_mae_improve") or 0) > 1e-5)
    ]
    inert = [r for r in legs if r.get("identical_to_s0")]

    report = {
        "anchor": args.anchor,
        "s0_baseline_f1": s0_f1,
        "preflight_oracle": (preflight or {}).get("oracle_f1_final_t"),
        "preflight_targets": (preflight or {}).get("targets"),
        "n_legs_diagnosed": len(legs),
        "promising_legs": promising,
        "species_moved_commits_stuck": species_only,
        "inert_legs": [r["leg_id"] for r in inert],
        "legs_ranked": legs,
    }

    out = Path(args.out) if args.out else sweep_dir / "diagnostics_summary.json"
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[OK] diagnostics_summary -> {out}", flush=True)
    if preflight and preflight.get("oracle_f1_final_t"):
        o = preflight["oracle_f1_final_t"]
        print(
            f"[i] oracle ceilings: gate_fn_fp={o.get('gate_fn_fp', 0):.3f} "
            f"species_fn_fp={o.get('species_fn_fp', 0):.3f}",
            flush=True,
        )
    print(f"[i] inert (=s0) legs: {len(inert)}", flush=True)
    print(f"[i] species moved / commits stuck: {len(species_only)}", flush=True)
    print(f"[i] promising (F1 delta > 0.005): {len(promising)}", flush=True)
    for r in promising[:5]:
        print(
            f"    {r.get('leg_id')}: F1 d={float(r.get('final_f1_delta', 0)):+.3f} "
            f"verdict={r.get('verdict')}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
