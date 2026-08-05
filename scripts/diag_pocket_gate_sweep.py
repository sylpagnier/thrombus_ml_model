"""Free percentile sweep for the pocket gate (docs/WALL_MODEL_PLAN.md s4 Step 1).

One rollout per anchor, many re-grades: ``grade_deploy_clot_series`` reads
``CLOT_POCKET_GATE_PCT`` fresh on every call (src/evaluation/pocket_gate.py), so the
percentile sweep costs nothing beyond the single closed-loop rollout per anchor -- the same
trick ``diag_gelation_beta_margin.py`` uses for the beta curve.

Run this on TRAINING vessels only to pick a percentile. s2.4 fitted a global threshold
(0.12) directly on patient020, the primary holdout, and found the optimum sharp (F1 0.876
at thr 0.12 -> 0.635 at 0.15) -- fitting a threshold on the test set proves nothing.  Once a
percentile looks good here, apply it ONCE to patient020 / 043 / 044 via
``scripts/eval_mat_growth_simple.py --pocket-gate-pct <pct>``.

    python scripts/diag_pocket_gate_sweep.py \
        --ckpt outputs/biochem/eda/wall_gen_clotrich_nplus/WG_clotrich_nplus/best.pth \
        --anchors patient021,patient032,patient035,patient037
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.eval_mat_growth_simple import _apply_ckpt_recipe, _load_static  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.species_deploy_rollout import reset_species_rollout_flow_cache  # noqa: E402
from src.core_physics.species_pushforward_continuous import (  # noqa: E402
    clear_offwall_model_cache,
    deploy_species_rollout_series,
    grade_deploy_clot_series,
    load_continuous_bundle,
)
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.utils.kinematics_inference import (  # noqa: E402
    load_kinematics_predictor,
    resolve_kinematics_checkpoint,
)
from src.utils.paths import get_project_root  # noqa: E402

ANCHOR_DIR = get_project_root() / "data/processed/graphs_biochem_anchors"
DEFAULT_PCTS = "5,10,15,20,25,30,40,50"


def _grade(data, series, static, phys, bio, device, pct: float | None) -> dict:
    if pct is None:
        os.environ.pop("CLOT_POCKET_GATE_PCT", None)
    else:
        os.environ["CLOT_POCKET_GATE_PCT"] = str(pct)
    try:
        return grade_deploy_clot_series(
            data, series, static, phys, bio, device,
            time_index=None, flow_source="kinematics", gelation_beta=None,
        )
    finally:
        os.environ.pop("CLOT_POCKET_GATE_PCT", None)


def main() -> int:
    ap = argparse.ArgumentParser(description="Pocket-gate percentile sweep from one rollout per anchor")
    ap.add_argument("--ckpt", required=True, help="Wall-model checkpoint to diagnose")
    ap.add_argument("--anchors", required=True, help="Comma list, e.g. patient021,patient032,patient035,patient037")
    ap.add_argument("--pcts", default=DEFAULT_PCTS, help=f"Percentile grid (default {DEFAULT_PCTS})")
    ap.add_argument("--out", default="outputs/biochem/eda/pocket_gate_sweep/diag.json")
    args = ap.parse_args()

    root = get_project_root()
    device = require_cuda_device()
    anchors = [a.strip() for a in args.anchors.split(",") if a.strip()]
    pcts = [float(p) for p in args.pcts.split(",") if p.strip()]
    ckpt = Path(args.ckpt)
    if not ckpt.is_absolute():
        ckpt = root / ckpt
    if not ckpt.is_file():
        raise FileNotFoundError(f"--ckpt not found: {ckpt}")

    clear_offwall_model_cache()
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    meta = dict(payload.get("meta") or {})
    _apply_ckpt_recipe(meta, label="pocket_gate_sweep", ckpt_path=ckpt)
    bundle = load_continuous_bundle(ckpt, device=device, quiet=True)
    if bundle is None:
        raise FileNotFoundError(f"could not load continuous bundle: {ckpt}")
    model = bundle.model
    wall_hops = int(meta.get("wall_hops", 3))
    kine = load_kinematics_predictor(
        str(resolve_kinematics_checkpoint()), device, phys_cfg=PhysicsConfig(phase="kinematics")
    )
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")

    report: dict = {"ckpt": str(ckpt), "pcts": pcts, "per_anchor": {}}

    for anc in anchors:
        print(f"\n=== {anc} ===", flush=True)
        reset_species_rollout_flow_cache()
        data = torch.load(ANCHOR_DIR / f"{anc}.pt", map_location=device, weights_only=False)
        static = _load_static(data, device, kine, wall_hops, anc)
        static["n_times"] = int(data.y.shape[0])

        print("  rolling out (this is the expensive part, once)...", flush=True)
        series, data = deploy_species_rollout_series(
            model, data, static, phys, bio, device, flow_source="kinematics", gelation_beta=None,
        )

        rows = []
        print(f"  {'pct':>6} {'f1':>7} {'prec':>7} {'rec':>7} {'mass':>7} {'ncomp':>6} {'kept':>5} {'fp':>7} {'fn':>7}", flush=True)

        base = _grade(data, series, static, phys, bio, device, None)
        base_row = {
            "pct": None,
            "deploy_clot_f1": base.get("deploy_clot_f1", 0.0),
            "deploy_clot_prec": base.get("deploy_clot_prec", 0.0),
            "deploy_clot_rec": base.get("deploy_clot_rec", 0.0),
            "deploy_clot_mass_ratio": base.get("deploy_clot_mass_ratio", 0.0),
            "deploy_clot_fp": base.get("deploy_clot_fp", 0.0),
            "deploy_clot_fn": base.get("deploy_clot_fn", 0.0),
        }
        rows.append(base_row)
        print(
            f"  {'off':>6} {base_row['deploy_clot_f1']:7.4f} {base_row['deploy_clot_prec']:7.4f} "
            f"{base_row['deploy_clot_rec']:7.4f} {base_row['deploy_clot_mass_ratio']:7.3f} "
            f"{'--':>6} {'--':>5} {base_row['deploy_clot_fp']:7.0f} {base_row['deploy_clot_fn']:7.0f}",
            flush=True,
        )

        for pct in pcts:
            m = _grade(data, series, static, phys, bio, device, pct)
            row = {
                "pct": pct,
                "deploy_clot_f1": m.get("deploy_clot_f1", 0.0),
                "deploy_clot_prec": m.get("deploy_clot_prec", 0.0),
                "deploy_clot_rec": m.get("deploy_clot_rec", 0.0),
                "deploy_clot_mass_ratio": m.get("deploy_clot_mass_ratio", 0.0),
                "deploy_pocket_gate_ncomp_total": m.get("deploy_pocket_gate_ncomp_total", 0.0),
                "deploy_pocket_gate_ncomp_kept": m.get("deploy_pocket_gate_ncomp_kept", 0.0),
                "deploy_clot_fp": m.get("deploy_clot_fp", 0.0),
                "deploy_clot_fn": m.get("deploy_clot_fn", 0.0),
            }
            rows.append(row)
            print(
                f"  {pct:6.1f} {row['deploy_clot_f1']:7.4f} {row['deploy_clot_prec']:7.4f} "
                f"{row['deploy_clot_rec']:7.4f} {row['deploy_clot_mass_ratio']:7.3f} "
                f"{row['deploy_pocket_gate_ncomp_total']:6.0f} {row['deploy_pocket_gate_ncomp_kept']:5.0f} "
                f"{row['deploy_clot_fp']:7.0f} {row['deploy_clot_fn']:7.0f}",
                flush=True,
            )

        best = max(rows, key=lambda r: r["deploy_clot_f1"])
        print(f"  best: pct={best['pct']} f1={best['deploy_clot_f1']:.4f}", flush=True)
        report["per_anchor"][anc] = {"curve": rows, "best_pct": best["pct"], "best_f1": best["deploy_clot_f1"]}

        clear_offwall_model_cache()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Mean F1 per percentile across anchors -- what should actually drive the pick.
    all_pcts = [None, *pcts]
    print(f"\n{'=== mean across anchors ===':>30}", flush=True)
    print(f"  {'pct':>6} {'mean_f1':>8}", flush=True)
    mean_curve = []
    for pct in all_pcts:
        vals = []
        for anc in anchors:
            for row in report["per_anchor"][anc]["curve"]:
                if row["pct"] == pct:
                    vals.append(row["deploy_clot_f1"])
        mean_f1 = sum(vals) / max(len(vals), 1)
        mean_curve.append({"pct": pct, "mean_f1": mean_f1})
        label = "off" if pct is None else f"{pct:.1f}"
        print(f"  {label:>6} {mean_f1:8.4f}", flush=True)
    report["mean_curve"] = mean_curve
    best_mean = max(mean_curve, key=lambda r: r["mean_f1"])
    print(f"\n=> best mean percentile: {best_mean['pct']} (f1={best_mean['mean_f1']:.4f})", flush=True)
    report["best_mean_pct"] = best_mean["pct"]

    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[save] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
