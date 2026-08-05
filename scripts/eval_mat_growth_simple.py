"""Eval Mat-growth-simple ckpt vs triangle6_wall3hop baseline (analytical gelation clot).

Metrics per anchor @ deploy horizon (pred kine):
  * deploy_mat_f1   - closed-loop Mat active F1 on wall+3hop band
  * deploy_clot_*   - analytical mu1(Mat) gelation + nucleation trigger

Usage::

    python scripts/eval_mat_growth_simple.py
    python scripts/eval_mat_growth_simple.py --ckpt outputs/biochem/biochem_gnn/mat_growth_simple/best.pth
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

from src.biochem_gnn.mat_growth_simple import (  # noqa: E402
    BASELINE_COMPARE_ID,
)
from src.biochem_gnn.config import apply_deploy_env, global_ckpt_path  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.species_deploy_rollout import reset_species_rollout_flow_cache  # noqa: E402
from src.core_physics.species_pushforward_continuous import (  # noqa: E402
    clear_offwall_model_cache,
    discover_biochem_anchors,
    deploy_eval_time_index,
    eval_deploy_clot_f1,
    eval_full_rollout_fimat_f1,
    load_continuous_bundle,
    train_deploy_eval_flow_source,
)
from src.core_physics.species_gnn_clot_rollout import (  # noqa: E402
    load_species_gnn_rollout_bundle,
    rollout_species_gnn_phi_trajectory,
    species_gnn_static_from_band_dict,
)
from src.evaluation.canonical_clot_eval import canonical_deploy_clot_metrics  # noqa: E402
from src.evaluation.seed_growth_diagnostics import (  # noqa: E402
    format_seed_growth_panel,
    seed_growth_diagnostic_panel,
)
from src.evaluation.clot_timeline_metrics import eval_clot_timeline_on_grid  # noqa: E402
from src.core_physics.species_pushforward_gnn import build_band_base_features  # noqa: E402
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.utils.kinematics_inference import (  # noqa: E402
    load_kinematics_predictor,
    predict_kinematics_and_latent,
    resolve_kinematics_checkpoint,
)
from src.utils.paths import get_project_root  # noqa: E402

ANCHOR_DIR = get_project_root() / "data/processed/graphs_biochem_anchors"
DEFAULT_BASELINE_JSON = (
    get_project_root()
    / "outputs/biochem/biochem_gnn/baselines"
    / BASELINE_COMPARE_ID
    / "baseline.json"
)

# Canonical promoted model (WC_v7_clot_phi_mse, 2026-07-19).
# Prefer locked/ (manifest source of truth), then mat_canonical_deploy alias.
LOCKED_CANONICAL_CKPT = get_project_root() / "outputs/biochem/biochem_gnn/locked/species_gnn_best.pth"
MAT_CANONICAL_CKPT = get_project_root() / "outputs/biochem/biochem_gnn/mat_canonical_deploy/species/best.pth"


def _resolve_baseline_ckpt(explicit: str) -> Path:
    """Return the best available baseline checkpoint.

    Priority:
      1. Explicit --baseline-ckpt arg (if provided)
      2. locked/species_gnn_best.pth           (canonical WC_v7_clot_phi_mse)
      3. mat_canonical_deploy/species/best.pth (synced alias)
      4. global_ckpt_path()                     (species/best.pth fallback)
    """
    if explicit.strip():
        return Path(explicit.strip())
    if LOCKED_CANONICAL_CKPT.is_file():
        return LOCKED_CANONICAL_CKPT
    if MAT_CANONICAL_CKPT.is_file():
        return MAT_CANONICAL_CKPT
    return global_ckpt_path()


def _load_static(data, device, kine_model, wall_hops: int, anchor: str) -> dict:
    """One joint GINO-DEQ solve per vessel; bake u0_pred + z_kin into pack features."""
    from src.utils.kinematics_inference import predict_kinematics_and_latent
    from src.utils.paths import get_project_root
    from pathlib import Path
    
    kine_stem = Path(kine_model.config.ckpt_path).stem if getattr(kine_model, "config", None) and hasattr(kine_model.config, "ckpt_path") else "deploy"
    cache_dir = get_project_root() / ".cache" / "kinematics_t0" / kine_stem

    with torch.no_grad():
        pred_uv, z_kin = predict_kinematics_and_latent(
            kine_model, 
            data,
            disk_cache_dir=cache_dir,
            disk_cache_key=anchor.strip()
        )
    data.u0_pred = pred_uv[:, 0].detach().to(device="cpu").clone()
    data.v0_pred = pred_uv[:, 1].detach().to(device="cpu").clone()
    return build_band_base_features(
        data, kine_model, device, wall_hops=wall_hops, z_kin_override=z_kin
    )


# Script-lifetime typed config contexts (entered by _apply_ckpt_recipe / two-model bind).
_EVAL_PF_CM = None
_EVAL_RT_CM = None
# Explicit --gelation-beta, re-applied after every recipe rebind (see _apply_ckpt_recipe).
_EVAL_GELATION_BETA = ""


def _bind_eval_typed_configs(pf, rt) -> None:
    """Keep PushforwardConfig / BiochemRuntimeConfig active for the eval process."""
    global _EVAL_PF_CM, _EVAL_RT_CM
    from src.architecture.pushforward_config import use_pushforward_config
    from src.architecture.runtime_config import use_biochem_runtime

    if _EVAL_PF_CM is not None:
        try:
            _EVAL_PF_CM.__exit__(None, None, None)
        except Exception:
            pass
        _EVAL_PF_CM = None
    if _EVAL_RT_CM is not None:
        try:
            _EVAL_RT_CM.__exit__(None, None, None)
        except Exception:
            pass
        _EVAL_RT_CM = None
    _EVAL_PF_CM = use_pushforward_config(pf)
    _EVAL_PF_CM.__enter__()
    _EVAL_RT_CM = use_biochem_runtime(rt)
    _EVAL_RT_CM.__enter__()


def _apply_ckpt_recipe(
    meta: dict,
    *,
    label: str,
    ckpt_path: Path | str | None = None,
    pf_overrides: dict[str, object] | None = None,
) -> None:
    """Bind typed train/deploy configs from checkpoint meta (architecture + runtime).

    Architecture / rollout policy go through PushforwardConfig and BiochemRuntimeConfig.
    Only residual unknown / IO env keys are written to os.environ.
    """
    from dataclasses import replace

    from src.architecture.pushforward_config import (
        PushforwardConfig,
        split_legacy_env_overrides,
    )
    from src.architecture.runtime_config import (
        BiochemRuntimeConfig,
        split_legacy_runtime_env,
    )
    from src.biochem_gnn.config import GLOBAL_TRAIN_RECIPE

    scope = meta.get("pushforward_species_scope") or meta.get("species_scope")
    recipe_env: dict[str, str] = dict(GLOBAL_TRAIN_RECIPE)
    if label == "mat_growth_simple" or scope == "mat":
        from src.biochem_gnn.mat_growth_simple import MAT_GROWTH_SIMPLE_RECIPE

        recipe_env.update({k: str(v) for k, v in MAT_GROWTH_SIMPLE_RECIPE.items()})
    # Deploy eval must not inherit training GT flow-feat source.
    recipe_env.pop("SPECIES_FLOW_FEATS_SOURCE", None)

    residual_env: dict[str, str] = {}
    overrides = meta.get("env_overrides")
    if isinstance(overrides, dict) and overrides:
        recipe_env = {**recipe_env, **{k: str(v) for k, v in overrides.items()}}
        recipe_env.pop("SPECIES_FLOW_FEATS_SOURCE", None)
    elif ckpt_path is not None:
        path_s = str(ckpt_path).replace("\\", "/")
        if "mat_growth_ladder/" in path_s:
            parts = path_s.split("mat_growth_ladder/")
            if len(parts) > 1:
                leg = parts[1].split("/")[0]
                if leg:
                    try:
                        from src.biochem_gnn.mat_growth_simple import mat_growth_leg_spec

                        spec = mat_growth_leg_spec(leg)
                        recipe_env.update({k: str(v) for k, v in spec.env_overrides.items()})
                        # Typed leg knobs applied via merged meta below.
                        meta = {
                            **meta,
                            "config_kwargs": {
                                **dict(spec.config_kwargs),
                                **dict(meta.get("config_kwargs") or {}),
                            },
                            "runtime_kwargs": {
                                **dict(spec.runtime_kwargs),
                                **dict(meta.get("runtime_kwargs") or {}),
                            },
                        }
                    except Exception as e:
                        print(f"[WARN] Failed to apply leg typed config for {leg} from path: {e}")

    merged_meta = {**meta, "env_overrides": recipe_env if recipe_env else meta.get("env_overrides")}
    pf = PushforwardConfig.from_meta(merged_meta)
    # Deploy-faithful: auto flow-feat source (kine + optional coupling).
    pf = replace(pf, flow_feats_source="auto")
    # Deploy-only sparse commitment overrides.
    if pf_overrides:
        pf = replace(pf, **pf_overrides)

    rt = BiochemRuntimeConfig.from_meta(merged_meta).with_overrides(
        deploy_faithful=True,
        rollout_vel_source="kinematics",
        rollout_pin_other="rest",
        rollout_ic_source="resting",
        # Fair cold-deploy A/B: never inherit a train-time GT/coupling-off crutch.
        corrector_coupling=True,
        closed_loop_coupling=True,
        train_deploy_eval_flow="auto",
    )
    # Preserve CLI --gelation-beta across recipe rebind (ckpt meta must not override it).
    if _EVAL_GELATION_BETA:
        rt = rt.with_overrides(beta_override=_EVAL_GELATION_BETA)
    # Preserve CLI two-model / off-wall bind across recipe rebind.
    prev_rt = None
    try:
        from src.architecture.runtime_config import get_active_runtime

        prev_rt = get_active_runtime()
    except Exception:
        pass
    if prev_rt is not None and prev_rt.offwall.two_model_mode:
        rt = rt.with_overrides(
            two_model_mode=True,
            offwall_model_ckpt=prev_rt.offwall.offwall_model_ckpt,
            two_model_route=prev_rt.offwall.two_model_route,
            two_model_frontier_hops=prev_rt.offwall.two_model_frontier_hops,
            frontier_hops_map=prev_rt.offwall.frontier_hops_map,
            frontier_hops_anchor=prev_rt.offwall.frontier_hops_anchor,
        )
    _bind_eval_typed_configs(pf, rt)

    # Residual unknown env only (not architecture / OffwallConfig / RolloutDeployConfig keys).
    if recipe_env:
        _cfg, rem1 = split_legacy_env_overrides(recipe_env)
        _rt, rem2 = split_legacy_runtime_env(rem1)
        residual_env.update(rem2)
    for k, v in residual_env.items():
        os.environ[k] = str(v)
    os.environ.pop("SPECIES_FLOW_FEATS_SOURCE", None)


def _eval_ckpt(
    ckpt_path: Path,
    anchors: list[str],
    device: torch.device,
    *,
    label: str,
    cheap_val: bool = False,
    pf_overrides: dict[str, object] | None = None,
) -> dict:
    clear_offwall_model_cache()
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = dict(payload.get("meta") or {})
    _apply_ckpt_recipe(meta, label=label, ckpt_path=ckpt_path, pf_overrides=pf_overrides)
    bundle = load_continuous_bundle(ckpt_path, device=device, quiet=True)
    if bundle is None:
        raise FileNotFoundError(f"could not load continuous bundle: {ckpt_path}")
    model = bundle.model
    wall_hops = int(meta.get("wall_hops", 3))
    kine = load_kinematics_predictor(
        str(resolve_kinematics_checkpoint()),
        device,
        phys_cfg=PhysicsConfig(phase="kinematics"),
    )
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    flow_eval = "kinematics"
    # Load timeline bundle once per ckpt (not once per vessel).
    gnn_bundle = load_species_gnn_rollout_bundle(ckpt_path, device=device, quiet=True)
    per: dict[str, dict] = {}
    from src.architecture.pushforward_config import get_active_config
    from src.architecture.runtime_config import get_active_runtime

    hops_map_raw = ""
    rt_active = get_active_runtime()
    if rt_active is not None:
        hops_map_raw = str(rt_active.offwall.frontier_hops_map or "")
    if not hops_map_raw:
        hops_map_raw = os.environ.get("SPECIES_TWO_MODEL_FRONTIER_HOPS_MAP", "")
    for anc in anchors:
        print(f"    - {anc}...", flush=True)
        if hops_map_raw:
            rt0 = get_active_runtime()
            pf0 = get_active_config()
            if rt0 is not None and pf0 is not None:
                _bind_eval_typed_configs(
                    pf0,
                    rt0.with_overrides(
                        frontier_hops_anchor=anc,
                        frontier_hops_map=hops_map_raw,
                    ),
                )
            else:
                os.environ["SPECIES_TWO_MODEL_FRONTIER_HOPS_ANCHOR"] = anc
        reset_species_rollout_flow_cache()
        data = torch.load(ANCHOR_DIR / f"{anc}.pt", map_location=device, weights_only=False)
        static = _load_static(data, device, kine, wall_hops, anc)
        static["n_times"] = int(data.y.shape[0])
        t_eval = deploy_eval_time_index(int(data.y.shape[0]))
        mat_m = eval_full_rollout_fimat_f1(
            model, data, static, device, time_index=t_eval
        )
        # Shared protocol with training (see src/evaluation/canonical_clot_eval.py).
        clot_m = canonical_deploy_clot_metrics(
            model,
            data,
            static,
            phys,
            bio,
            device,
            time_index=t_eval,
            flow_source=flow_eval,
        )

        timeline_summary: dict[str, float] = {}
        if not cheap_val:
            try:
                if gnn_bundle is not None:
                    # Reuse the same band static / u0_pred (no second DEQ + ckpt reload).
                    gnn_static = species_gnn_static_from_band_dict(
                        static, data, device=device, wall_hops=wall_hops
                    )
                    phi_traj = rollout_species_gnn_phi_trajectory(
                        data,
                        gnn_bundle,
                        gnn_static,
                        phys_cfg=phys,
                        bio_cfg=bio,
                        device=device,
                        flow_source=flow_eval,
                    )
                    tl = eval_clot_timeline_on_grid(phi_traj, data, phys, device, max_frames=10)
                    timeline_summary = dict(tl.get("summary") or {})
            except Exception as exc:
                print(f"[WARN] clot timeline metrics skipped for {anc}: {exc}", flush=True)

        per[anc] = {
            "t_eval": int(t_eval),
            "deploy_mat_f1": float(mat_m["deploy_mat_f1"]),
            "deploy_fi_f1": float(mat_m.get("deploy_fi_f1", 0.0)),
            "mat_seed_prec": float(mat_m.get("mat_seed_prec", 0.0)),
            "mat_seed_count": float(mat_m.get("mat_seed_count", 0.0)),
            "mat_front_prec": float(mat_m.get("mat_front_prec", 0.0)),
            "mat_front_speed_ratio": float(mat_m.get("mat_front_speed_ratio", 0.0)),
            "mat_overpaint_frac": float(mat_m.get("mat_overpaint_frac", 0.0)),
            "mat_overpaint_per_gt": float(mat_m.get("mat_overpaint_per_gt", 0.0)),
            "deploy_clot_f1": float(clot_m["deploy_clot_f1"]),
            "deploy_clot_score": float(clot_m.get("deploy_clot_score", 0.0)),
            "deploy_clot_relaxed_prec": float(clot_m.get("deploy_clot_relaxed_prec", 0.0)),
            "deploy_clot_relaxed_rec": float(clot_m.get("deploy_clot_relaxed_rec", 0.0)),
            "deploy_clot_offwall_relaxed_f1": float(clot_m.get("deploy_clot_offwall_relaxed_f1", 0.0)),
            "deploy_clot_offwall_strict_f1": float(clot_m.get("deploy_clot_offwall_strict_f1", 0.0)),
            "deploy_clot_offwall_n_pred": float(clot_m.get("deploy_clot_offwall_n_pred", 0.0)),
            "deploy_clot_offwall_n_gt": float(clot_m.get("deploy_clot_offwall_n_gt", 0.0)),
            "deploy_clot_offwall_n_pred_hop2": float(clot_m.get("deploy_clot_offwall_n_pred_hop2", 0.0)),
            "deploy_clot_offwall_n_pred_hop3": float(clot_m.get("deploy_clot_offwall_n_pred_hop3", 0.0)),
            "deploy_clot_offwall_n_pred_hop_ge2": float(clot_m.get("deploy_clot_offwall_n_pred_hop_ge2", 0.0)),
            "deploy_clot_offwall_n_gt_hop_ge2": float(clot_m.get("deploy_clot_offwall_n_gt_hop_ge2", 0.0)),
            "deploy_clot_offwall_strict_f1_hop2": float(clot_m.get("deploy_clot_offwall_strict_f1_hop2", 0.0)),
            "deploy_clot_offwall_strict_f1_hop_ge2": float(clot_m.get("deploy_clot_offwall_strict_f1_hop_ge2", 0.0)),
            "deploy_wall_score": float(clot_m.get("deploy_wall_score", 0.0)),
            "deploy_wall_strict_f1": float(clot_m.get("deploy_wall_strict_f1", 0.0)),
            "deploy_clot_mass_ratio": float(clot_m.get("deploy_clot_mass_ratio", 0.0)),
            "deploy_clot_empty_gt_score": float(clot_m.get("deploy_clot_empty_gt_score", 0.0)),
            "deploy_clot_fp": float(clot_m.get("deploy_clot_fp", 0.0)),
            "deploy_clot_fn": float(clot_m.get("deploy_clot_fn", 0.0)),
            "deploy_gelation_beta": float(clot_m.get("deploy_gelation_beta", 1.0)),
            "deploy_pocket_gate_pct": float(clot_m.get("deploy_pocket_gate_pct", 0.0)),
            "deploy_pocket_gate_thresh": float(clot_m.get("deploy_pocket_gate_thresh", 0.0)),
            "deploy_pocket_gate_ncomp_total": float(clot_m.get("deploy_pocket_gate_ncomp_total", 0.0)),
            "deploy_pocket_gate_ncomp_kept": float(clot_m.get("deploy_pocket_gate_ncomp_kept", 0.0)),
            "deploy_pocket_gate_dropped_nodes": float(clot_m.get("deploy_pocket_gate_dropped_nodes", 0.0)),
            **{k: float(v) for k, v in timeline_summary.items()},
        }
        panel = seed_growth_diagnostic_panel(per[anc])
        per[anc]["diagnostic"] = panel
        print(format_seed_growth_panel(panel, label=anc), flush=True)
    keys = (
        "deploy_mat_f1",
        "deploy_clot_f1",
        "deploy_clot_score",
        "deploy_clot_offwall_relaxed_f1",
        "deploy_clot_offwall_strict_f1",
        "deploy_clot_offwall_n_pred",
        "deploy_clot_offwall_n_gt",
        "deploy_clot_offwall_n_pred_hop2",
        "deploy_clot_offwall_n_pred_hop3",
        "deploy_clot_offwall_n_pred_hop_ge2",
        "deploy_clot_offwall_n_gt_hop_ge2",
        "deploy_clot_offwall_strict_f1_hop2",
        "deploy_clot_offwall_strict_f1_hop_ge2",
        "deploy_wall_score",
        "deploy_wall_strict_f1",
        "deploy_clot_mass_ratio",
        "deploy_clot_empty_gt_score",
        "deploy_clot_fp",
        "deploy_clot_fn",
        "deploy_gelation_beta",
        "deploy_pocket_gate_pct",
        "deploy_pocket_gate_thresh",
        "deploy_pocket_gate_ncomp_total",
        "deploy_pocket_gate_ncomp_kept",
        "deploy_pocket_gate_dropped_nodes",
        "mat_seed_prec",
        "mat_seed_count",
        "mat_front_prec",
        "mat_front_speed_ratio",
        "mat_overpaint_frac",
        "mat_overpaint_per_gt",
        "clot_fp_median",
        "clot_fp_p90",
        "clot_fp_max",
        "clot_fn_median",
        "clot_err_median",
        "clot_err_p90",
        "clot_fp_early_mean",
    )
    mean = {k: sum(per[a].get(k, 0.0) for a in anchors) / max(len(anchors), 1) for k in keys}
    mean_panel = seed_growth_diagnostic_panel(mean)
    print(format_seed_growth_panel(mean_panel, label="mean"), flush=True)
    return {
        "label": label,
        "ckpt": str(ckpt_path),
        "per_anchor": per,
        "mean": mean,
        "diagnostic": mean_panel,
        "meta": meta,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Mat-growth-simple vs canonical baseline eval")
    ap.add_argument("--ckpt", default="", help="Mat-only simple ckpt (default: mat_growth_simple/best.pth)")
    ap.add_argument("--baseline-ckpt", default="",
                    help="Baseline ckpt (default: locked/species_gnn_best.pth = WC_v7_clot_phi_mse, "
                         "then mat_canonical_deploy /, then species/best.pth)")
    ap.add_argument("--baseline-json", default=str(DEFAULT_BASELINE_JSON))
    ap.add_argument("--anchors", default="", help="Comma list (default: all anchors on disk)")
    ap.add_argument("--out", default="outputs/biochem/biochem_gnn/mat_growth_simple/compare.json")
    ap.add_argument(
        "--offwall-ckpt",
        default="",
        help="Optional growth specialist ckpt; enables SPECIES_TWO_MODEL_MODE=1 during primary eval",
    )
    ap.add_argument(
        "--two-model-route",
        default="",
        choices=("", "wall", "frontier", "frontier_offwall", "frontier_lumen_only"),
        help="Routing for two-model blend (default: frontier when --offwall-ckpt is set)",
    )
    ap.add_argument(
        "--two-model-frontier-hops",
        type=float,
        default=2.0,
        help="Hops around committed Mat for frontier routing (0.5=tight off-wall shell)",
    )
    ap.add_argument(
        "--two-model-frontier-hops-map",
        default="",
        help="Per-anchor hops map, e.g. patient010:0.5,patient007:1,default:1",
    )
    ap.add_argument(
        "--deploy-frontier-hops",
        type=int,
        default=None,
        help="Deploy-time sparse commitment: restrict growth to k-hop predicted frontier (PushforwardConfig.frontier_hops).",
    )
    ap.add_argument(
        "--deploy-nucleation-topk",
        type=float,
        default=None,
        help="Deploy-time sparse commitment: top-k gate-logit fraction allowed to nucleate at t0 (PushforwardConfig.nucleation_topk).",
    )
    ap.add_argument(
        "--deploy-gate-temp",
        type=float,
        default=None,
        help="Deploy-time sparse commitment: sigmoid temperature for spatial gate sharpness (PushforwardConfig.gate_temp).",
    )
    ap.add_argument(
        "--deploy-mat-commit-thresh",
        type=float,
        default=None,
        help="Deploy-time sparse commitment: accumulated log-Mat threshold for committed support (PushforwardConfig.mat_commit_thresh).",
    )
    ap.add_argument(
        "--deploy-clot-trigger-commit-thresh",
        type=float,
        default=None,
        help="Deploy-time clot-trigger projection threshold (CLOT_TRIGGER_COMMIT_THRESH).",
    )
    ap.add_argument(
        "--pocket-gate-pct",
        type=float,
        default=None,
        help="Deploy post-process (docs/WALL_MODEL_PLAN.md s4 Step 1): drop predicted clot "
             "components whose min hop-2 speed is at/above this percentile of the vessel's "
             "own wall-node hop-2 speed distribution. Unset = gate off (unchanged behaviour). "
             "No retraining -- re-grades the same rollout (CLOT_POCKET_GATE_PCT).",
    )
    ap.add_argument(
        "--gelation-beta",
        type=float,
        default=None,
        help="Gelation readout gain applied to the graded clot label (valid [0.1, 2.0]). "
             "Unset = 1.0 = historical grading. Drives both the closed-loop mu_eff and the "
             "final readout, so each value needs its own rollout.",
    )
    ap.add_argument(
        "--no-baseline",
        action="store_true",
        help="Only eval --ckpt (skip second baseline pass; for A/B arm scripts)",
    )
    ap.add_argument(
        "--mat-leg",
        default="",
        help="Force mat-growth leg env before eval (e.g. WC_v7_clot_phi_mse)",
    )
    ap.add_argument(
        "--wall-floor-json",
        default="",
        help="Wall-alone eval JSON for compound gate guardrail (mean deploy_clot_f1)",
    )
    ap.add_argument(
        "--cheap-val",
        action="store_true",
        help="Skip heavy timeline grid interpolation (fast evaluation)",
    )
    ap.add_argument(
        "--list-legs",
        default="",
        help="Print matching ladder leg codes and exit",
    )
    args = ap.parse_args()

    if args.list_legs.strip():
        try:
            from src.biochem_gnn.mat_growth_simple import LADDER_LEG_ORDER
            filter_str = args.list_legs.strip().lower()
            for leg in LADDER_LEG_ORDER:
                if filter_str in leg.lower():
                    print(leg)
            return 0
        except Exception as e:
            print(f"[ERROR] Failed to list legs: {e}")
            return 1

    global _EVAL_GELATION_BETA
    if args.gelation_beta is not None:
        beta = float(args.gelation_beta)
        if not (0.1 <= beta <= 2.0):
            print(f"[ERROR] --gelation-beta {beta} outside valid range [0.1, 2.0]")
            return 1
        _EVAL_GELATION_BETA = f"{beta:.6g}"
        # Env is the fallback path for any rebind that drops the typed field.
        os.environ["SPECIES_GELATION_BETA_OVERRIDE"] = _EVAL_GELATION_BETA
        print(f"[i] gelation readout beta = {_EVAL_GELATION_BETA} (default grading is 1.0)", flush=True)

    root = get_project_root()
    device = require_cuda_device()
    anchors = (
        [a.strip() for a in args.anchors.split(",") if a.strip()]
        if args.anchors.strip()
        else discover_biochem_anchors(ANCHOR_DIR)
    )
    simple_ckpt = Path(args.ckpt) if args.ckpt.strip() else root / "outputs/biochem/biochem_gnn/mat_growth_simple/best.pth"
    if not simple_ckpt.is_absolute():
        simple_ckpt = root / simple_ckpt
    baseline_ckpt = _resolve_baseline_ckpt(args.baseline_ckpt)
    baseline_label = str(baseline_ckpt.relative_to(root)) if baseline_ckpt.is_relative_to(root) else str(baseline_ckpt)

    report: dict = {
        "anchors": anchors,
        "baseline_id": BASELINE_COMPARE_ID,
        "baseline_json": str(args.baseline_json),
        "baseline_ckpt": str(baseline_ckpt),
    }
    if Path(args.baseline_json).is_file():
        report["baseline_recorded"] = json.loads(Path(args.baseline_json).read_text(encoding="utf-8"))

    orig_env = dict(os.environ)

    if args.mat_leg.strip():
        from src.biochem_gnn.mat_growth_simple import apply_mat_growth_leg_env

        apply_mat_growth_leg_env(args.mat_leg.strip(), force=True)
        print(f"[i] Forced mat-leg env: {args.mat_leg.strip()}", flush=True)

    pf_overrides: dict[str, object] = {}
    if args.deploy_frontier_hops is not None:
        pf_overrides["frontier_hops"] = int(args.deploy_frontier_hops)
    if args.deploy_nucleation_topk is not None:
        pf_overrides["nucleation_topk"] = float(args.deploy_nucleation_topk)
    if args.deploy_gate_temp is not None:
        pf_overrides["gate_temp"] = float(args.deploy_gate_temp)
    if args.deploy_mat_commit_thresh is not None:
        pf_overrides["mat_commit_thresh"] = float(args.deploy_mat_commit_thresh)
    if args.deploy_clot_trigger_commit_thresh is not None:
        os.environ["CLOT_TRIGGER_COMMIT_THRESH"] = str(float(args.deploy_clot_trigger_commit_thresh))
    if args.pocket_gate_pct is not None:
        pct = float(args.pocket_gate_pct)
        if not (0.0 <= pct <= 100.0):
            print(f"[ERROR] --pocket-gate-pct {pct} outside [0, 100]")
            return 1
        os.environ["CLOT_POCKET_GATE_PCT"] = str(pct)
        print(f"[i] pocket gate percentile = {pct} (unset = gate off)", flush=True)

    report["deploy_overrides"] = {
        "frontier_hops": pf_overrides.get("frontier_hops"),
        "nucleation_topk": pf_overrides.get("nucleation_topk"),
        "gate_temp": pf_overrides.get("gate_temp"),
        "mat_commit_thresh": pf_overrides.get("mat_commit_thresh"),
        "clot_trigger_commit_thresh": (
            float(args.deploy_clot_trigger_commit_thresh)
            if args.deploy_clot_trigger_commit_thresh is not None
            else None
        ),
        "gelation_beta": float(args.gelation_beta) if args.gelation_beta is not None else None,
        "pocket_gate_pct": float(args.pocket_gate_pct) if args.pocket_gate_pct is not None else None,
    }

    offwall_raw = args.offwall_ckpt.strip()
    if offwall_raw:
        offwall_path = Path(offwall_raw)
        if not offwall_path.is_absolute():
            offwall_path = root / offwall_path
        if not offwall_path.is_file():
            raise FileNotFoundError(f"--offwall-ckpt not found: {offwall_path}")
        route = args.two_model_route.strip() or "frontier"
        hops_map = args.two_model_frontier_hops_map.strip()
        from src.architecture.pushforward_config import PushforwardConfig, get_active_config
        from src.architecture.runtime_config import BiochemRuntimeConfig, get_active_runtime

        pf0 = get_active_config() or PushforwardConfig()
        rt0 = get_active_runtime() or BiochemRuntimeConfig()
        _bind_eval_typed_configs(
            pf0,
            rt0.with_overrides(
                two_model_mode=True,
                offwall_model_ckpt=str(offwall_path).replace("\\", "/"),
                two_model_route=route,
                two_model_frontier_hops=float(args.two_model_frontier_hops),
                frontier_hops_map=hops_map,
            ),
        )
        report["two_model"] = {
            "enabled": True,
            "offwall_ckpt": str(offwall_path),
            "route": route,
            "frontier_hops": float(args.two_model_frontier_hops),
            "frontier_hops_map": hops_map or None,
        }
        print(
            f"[i] two-model ON route={route} frontier_hops={args.two_model_frontier_hops} "
            f"growth={offwall_path}",
            flush=True,
        )
    else:
        from src.architecture.pushforward_config import PushforwardConfig, get_active_config
        from src.architecture.runtime_config import BiochemRuntimeConfig, get_active_runtime

        pf0 = get_active_config() or PushforwardConfig()
        rt0 = get_active_runtime() or BiochemRuntimeConfig()
        _bind_eval_typed_configs(pf0, rt0.with_overrides(two_model_mode=False, offwall_model_ckpt=""))
        report["two_model"] = {"enabled": False}

    print(f"[i] eval leg: {simple_ckpt}", flush=True)
    report["simple"] = _eval_ckpt(
        simple_ckpt,
        [anc.strip() for anc in args.anchors.split(",") if anc.strip()],
        device,
        label="mat_growth_simple",
        cheap_val=getattr(args, "cheap_val", False),
        pf_overrides=pf_overrides or None,
    )

    # Clean up environment overrides and empty CUDA cache to prevent memory accumulation and paging hangs
    os.environ.clear()
    os.environ.update(orig_env)
    clear_offwall_model_cache()
    import gc
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if args.no_baseline:
        report["baseline"] = None
        report["delta_simple_minus_baseline"] = None
        out = Path(args.out)
        if not out.is_absolute():
            out = root / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        mean = report["simple"]["mean"]
        print(f"\n[OK] primary-only eval -> {out}", flush=True)
        for k in (
            "deploy_mat_f1",
            "deploy_clot_f1",
            "deploy_clot_score",
            "deploy_wall_score",
            "deploy_wall_strict_f1",
            "deploy_clot_mass_ratio",
            "deploy_clot_empty_gt_score",
            "deploy_clot_fp",
            "deploy_clot_fn",
            "deploy_clot_offwall_relaxed_f1",
            "deploy_clot_offwall_strict_f1",
            "deploy_clot_offwall_n_pred",
            "deploy_clot_offwall_n_gt",
            "deploy_clot_offwall_n_pred_hop_ge2",
            "deploy_clot_offwall_n_gt_hop_ge2",
            "deploy_clot_offwall_strict_f1_hop_ge2",
            "mat_seed_prec",
            "mat_seed_count",
            "mat_front_speed_ratio",
            "mat_overpaint_frac",
        ):
            print(f"  {k}: {mean.get(k, 0.0):.4f}", flush=True)
        diag = report["simple"].get("diagnostic") or seed_growth_diagnostic_panel(mean)
        print(format_seed_growth_panel(diag, label="summary"), flush=True)
        if report.get("two_model", {}).get("enabled"):
            from src.evaluation.compound_deploy_gates import (  # noqa: E402
                format_gate_summary,
                gate_compound_eval_report,
            )

            wall_floor_f1 = None
            if args.wall_floor_json.strip():
                wf = Path(args.wall_floor_json)
                if not wf.is_absolute():
                    wf = root / wf
                if wf.is_file():
                    wrep = json.loads(wf.read_text(encoding="utf-8"))
                    ws = wrep.get("simple") or wrep
                    wall_floor_f1 = float((ws.get("mean") or {}).get("deploy_clot_f1", 0) or 0)
            gates = gate_compound_eval_report(report, wall_floor_f1=wall_floor_f1)
            report["compound_gates"] = gates
            out.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"[i] compound gates: {format_gate_summary(gates)}", flush=True)
        return 0

    print(f"[i] eval canonical baseline: {baseline_ckpt}", flush=True)
    report["baseline"] = _eval_ckpt(
        baseline_ckpt,
        anchors,
        device,
        label=baseline_label,
    )

    delta = {
        k: report["simple"]["mean"][k] - report["baseline"]["mean"][k]
        for k in report["simple"]["mean"]
    }
    report["delta_simple_minus_baseline"] = delta

    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n==================== MAT GROWTH NEW LEG vs CANONICAL ({baseline_ckpt.name}) ====================", flush=True)
    print(
        f"{'metric':<30} {'baseline':>10} {'simple':>10} {'delta':>10}",
        flush=True,
    )
    for k in (
        "deploy_mat_f1",
        "deploy_clot_f1",
        "deploy_clot_score",
        "deploy_wall_score",
        "deploy_wall_strict_f1",
        "deploy_clot_mass_ratio",
        "deploy_clot_empty_gt_score",
        "deploy_clot_offwall_relaxed_f1",
        "deploy_clot_offwall_strict_f1",
        "deploy_clot_offwall_n_pred",
        "deploy_clot_offwall_n_gt",
    ):
        b = report["baseline"]["mean"].get(k, 0.0)
        s = report["simple"]["mean"].get(k, 0.0)
        d = delta.get(k, 0.0)
        print(f"{k:<30} {b:10.3f} {s:10.3f} {d:+10.3f}", flush=True)
    print(f"[save] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
