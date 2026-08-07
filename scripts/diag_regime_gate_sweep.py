r"""What does the s10.4 regime router actually buy? (docs/WALL_MODEL_PLAN.md s10.5)

s2.7 measured the pocket gate as a real-gain-but-not-free-lunch: +0.451 on patient021,
+0.314 on 035, ~flat on 032, and a genuine -0.113 loss on 037 that no threshold fixes. It
concluded a global percentile was "the ceiling of what a flow-only post-process can do,"
because you cannot tell in advance which vessels it harms.

s10.4 says you can, from t=0 flow alone. This measures the payoff directly: ONE rollout per
anchor, re-graded three ways --

    gate OFF          | gate ON (global)   | gate ON + regime routing (skip on inverted)

-- reusing the single-rollout-many-grades trick (grade_deploy_clot_series re-reads
CLOT_POCKET_GATE_PCT and CLOT_POCKET_GATE_REGIME_ROUTE fresh on every call), so the sweep
costs nothing beyond one closed-loop rollout per vessel.

    python scripts/diag_regime_gate_sweep.py --anchors patient021,patient037,patient035,patient032
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
from src.evaluation.canonical_clot_eval import canonical_grade_series  # noqa: E402
from src.evaluation.clot_relaxed_metrics import scoring_fingerprint  # noqa: E402
from src.evaluation.pocket_gate import DEFAULT_REGIME_BAND_SPEED_THRESH  # noqa: E402
from src.utils.kinematics_inference import (  # noqa: E402
    load_kinematics_predictor,
    resolve_kinematics_checkpoint,
)
from src.utils.paths import get_project_root  # noqa: E402

ANCHOR_DIR = get_project_root() / "data/processed/graphs_biochem_anchors"
DEFAULT_CKPT = "outputs/biochem/eda/wall_gen_clotrich_nplus/WG_clotrich_nplus/best.pth"


def _grade(data, series, static, phys, bio, device, pct, route) -> dict:
    for k, v in (("CLOT_POCKET_GATE_PCT", pct), ("CLOT_POCKET_GATE_REGIME_ROUTE", route)):
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)
    try:
        # Canonical protocol, same as eval_mat_growth_simple.py. Calling
        # grade_deploy_clot_series directly here silently used a DIFFERENT protocol and made
        # deploy_clot_score incomparable across tools (WALL_MODEL_PLAN.md 20.1).
        return canonical_grade_series(
            data, series, static, phys, bio, device,
            time_index=None, flow_source="kinematics", gelation_beta=None,
        )
    finally:
        os.environ.pop("CLOT_POCKET_GATE_PCT", None)
        os.environ.pop("CLOT_POCKET_GATE_REGIME_ROUTE", None)


def main() -> int:
    ap = argparse.ArgumentParser(description="Gate OFF vs global vs regime-routed")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--pct", type=float, default=25.0)
    ap.add_argument("--route-thresh", type=float, default=DEFAULT_REGIME_BAND_SPEED_THRESH)
    ap.add_argument("--out", default="outputs/biochem/eda/clot_physics/regime_gate_sweep.json")
    args = ap.parse_args()

    root = get_project_root()
    device = require_cuda_device()
    anchors = [a.strip() for a in args.anchors.split(",") if a.strip()]
    ckpt = Path(args.ckpt)
    if not ckpt.is_absolute():
        ckpt = root / ckpt

    clear_offwall_model_cache()
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    meta = dict(payload.get("meta") or {})
    _apply_ckpt_recipe(meta, label="regime_gate_sweep", ckpt_path=ckpt)
    print(f"[i] SCORING FINGERPRINT {scoring_fingerprint()}", flush=True)
    bundle = load_continuous_bundle(ckpt, device=device, quiet=True)
    model = bundle.model
    wall_hops = int(meta.get("wall_hops", 3))
    kine = load_kinematics_predictor(
        str(resolve_kinematics_checkpoint()), device, phys_cfg=PhysicsConfig(phase="kinematics")
    )
    phys, bio = PhysicsConfig(phase="biochem"), BiochemConfig(phase="biochem")

    rows = []
    for anc in anchors:
        print(f"\n=== {anc} ===", flush=True)
        reset_species_rollout_flow_cache()
        data = torch.load(ANCHOR_DIR / f"{anc}.pt", map_location=device, weights_only=False)
        static = _load_static(data, device, kine, wall_hops, anc)
        static["n_times"] = int(data.y.shape[0])
        print("  rolling out (once)...", flush=True)
        series, data = deploy_species_rollout_series(
            model, data, static, phys, bio, device, flow_source="kinematics", gelation_beta=None,
        )
        off = _grade(data, series, static, phys, bio, device, None, None)
        glob = _grade(data, series, static, phys, bio, device, args.pct, None)
        route = _grade(data, series, static, phys, bio, device, args.pct, args.route_thresh)
        r = {
            "anchor": anc,
            # PRIMARY metric = deploy_clot_score (relaxed_prec_floor). F1 kept alongside.
            "score_off": off.get("deploy_clot_score", 0.0),
            "score_global": glob.get("deploy_clot_score", 0.0),
            "score_routed": route.get("deploy_clot_score", 0.0),
            "f1_off": off.get("deploy_clot_f1", 0.0),
            "f1_global": glob.get("deploy_clot_f1", 0.0),
            "f1_routed": route.get("deploy_clot_f1", 0.0),
            "mass_off": off.get("deploy_clot_mass_ratio", 0.0),
            "mass_global": glob.get("deploy_clot_mass_ratio", 0.0),
            "mass_routed": route.get("deploy_clot_mass_ratio", 0.0),
            "band_q25": route.get("deploy_regime_band_speed_q25", float("nan")),
            "inverted": route.get("deploy_regime_inverted", 0.0),
            "skipped": route.get("deploy_pocket_gate_skipped_inverted", 0.0),
        }
        rows.append(r)
        print(f"  band_q25={r['band_q25']:.4f} inverted={bool(r['inverted'])}  "
              f"SCORE: off={r['score_off']:.4f} global={r['score_global']:.4f} routed={r['score_routed']:.4f}  "
              f"| F1: {r['f1_off']:.4f}/{r['f1_global']:.4f}/{r['f1_routed']:.4f}", flush=True)
        clear_offwall_model_cache()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\n{'='*92}\n=== Gate OFF vs GLOBAL vs REGIME-ROUTED (pct={args.pct}, route>={args.route_thresh}) ===\n{'='*92}")
    print(f"  {'anchor':>13} {'regime':>9} {'off':>7} {'global':>7} {'routed':>7} "
          f"{'glob-off':>9} {'rout-off':>9} {'rout-glob':>10}")
    n_saved = 0
    for r in rows:
        reg = "INVERTED" if r["inverted"] else "normal"
        dg, dr, drg = r["score_global"] - r["score_off"], r["score_routed"] - r["score_off"], r["score_routed"] - r["score_global"]
        if drg > 1e-9:
            n_saved += 1
        print(f"  {r['anchor']:>13} {reg:>9} {r['score_off']:7.4f} {r['score_global']:7.4f} {r['score_routed']:7.4f} "
              f"{dg:+9.4f} {dr:+9.4f} {drg:+10.4f}")
    import numpy as np
    off = np.array([r["score_off"] for r in rows])
    gl = np.array([r["score_global"] for r in rows])
    ro = np.array([r["score_routed"] for r in rows])
    f_off = np.array([r["f1_off"] for r in rows])
    f_ro = np.array([r["f1_routed"] for r in rows])
    print(f"\n  mean F1:  off={off.mean():.4f}   global={gl.mean():.4f}   routed={ro.mean():.4f}")
    print(f"  worst-vessel delta vs off:  global={np.min(gl-off):+.4f}   routed={np.min(ro-off):+.4f}"
          f"   <- s2.7's minimax concern")
    print(f"  vessels where routing beat the global gate: {n_saved}/{len(rows)}")

    p = Path(args.out)
    if not p.is_absolute():
        p = root / p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"pct": args.pct, "route_thresh": args.route_thresh, "rows": rows}, indent=2), encoding="utf-8")
    print(f"\n[save] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
