"""Clot trigger stack (T0-T6): flow -> species -> phi commit (+ optional mu->kine loop).

Star 1 (T1): GT flow + GT species -> hybrid trigger (physics gelation + learned MLP).
Star 2: pred flow + GT species. Star 3: pred flow + pred species (biochem teacher).
Star 5: retrain deploy teacher (pred kine + FI/Mat). Star 6: T4/T5 + mu->kine coupling.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.config import BiochemConfig, PhysicsConfig
from src.utils import species_channels as sc
from src.core_physics.clot_phi_simple import (
    ClotPhiStepBatch,
    build_clot_phi_step,
    cap_mu_eff_si,
    clot_phi_forward_apply_region,
    clot_phi_hybrid_enabled,
    clot_phi_model_uses_mpnn,
    log_blend_mu_eff_si,
    mu_eff_from_delta_log_si,
    physics_mu_eff_si,
    physics_phi_from_mu,
)
from src.utils.paths import get_project_root


class ClotTriggerStar(str, Enum):
    T0_ORACLE = "t0"  # physics-only eval, no train
    T1_GT_INPUTS = "t1"  # GT flow + GT species, train hybrid trigger
    T2_PRED_FLOW = "t2"  # pred kine + GT species
    T3_DUMPED_SPECIES = "t3"  # pred kine + cached teacher species dump
    T4_LIVE_TEACHER = "t4"  # pred kine + live GraphSAGE species rollout (frozen global ckpt)
    T5_DEPLOY_TEACHER = "t5"  # retrain teacher (pred kine, FI/Mat) + pred-flow species dump
    T6_COUPLED = "t6"  # T4/T5 stack + phi/mu -> GINO-DEQ feedback each macro step


@dataclass
class ClotTriggerPaths:
    out_dir: Path
    ckpt_name: str = "clot_trigger_t1_best.pth"
    log_name: str = "clot_trigger_t1_train_log.jsonl"

    @property
    def ckpt_path(self) -> Path:
        return self.out_dir / self.ckpt_name


def trigger_paths_for_star(star: ClotTriggerStar) -> ClotTriggerPaths:
    root = get_project_root() / "outputs" / "biochem" / "clot_trigger"
    leg = star.value
    names = {
        ClotTriggerStar.T1_GT_INPUTS: "clot_trigger_t1_best.pth",
        ClotTriggerStar.T2_PRED_FLOW: "clot_trigger_t2_best.pth",
        ClotTriggerStar.T3_DUMPED_SPECIES: "clot_trigger_t1_best.pth",
        ClotTriggerStar.T4_LIVE_TEACHER: "clot_trigger_t1_best.pth",
        ClotTriggerStar.T5_DEPLOY_TEACHER: "clot_trigger_t1_best.pth",
        ClotTriggerStar.T6_COUPLED: "clot_trigger_t1_best.pth",
    }
    return ClotTriggerPaths(
        out_dir=root / leg,
        ckpt_name=names.get(star, f"clot_trigger_{leg}_best.pth"),
        log_name=f"clot_trigger_{leg}_train_log.jsonl",
    )


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def physics_blend_alpha() -> float:
    return max(0.0, min(float(os.environ.get("CLOT_PHI_PHYSICS_BLEND_ALPHA", "0.55") or "0.55"), 1.0))


def physics_blend_enabled() -> bool:
    return _env_bool("CLOT_PHI_PHYSICS_BLEND", False)


def _model_logits(
    model: nn.Module, feats: torch.Tensor, edge_index: torch.Tensor | None
) -> torch.Tensor:
    if clot_phi_model_uses_mpnn(model):
        if edge_index is None:
            raise ValueError("mpnn model requires edge_index")
        return model.forward_logits(feats, edge_index)
    return model.forward_logits(feats)


def _model_delta_log_mu(
    model: nn.Module, feats: torch.Tensor, edge_index: torch.Tensor | None
) -> torch.Tensor:
    if clot_phi_model_uses_mpnn(model):
        if edge_index is None:
            raise ValueError("mpnn model requires edge_index")
        return model.forward_delta_log_mu(feats, edge_index)
    return model.forward_delta_log_mu(feats)


def forward_physics_trigger_phi(
    step: ClotPhiStepBatch,
    data,
    *,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    species_log1p: torch.Tensor | None = None,
    use_soft: bool = True,
    apply_region: bool = True,
    time_index: int | None = None,
    mu_anchor_si: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Explicit Mat/FI gelation trigger (no learned head)."""
    sp = species_log1p if species_log1p is not None else step.species_log_gt
    mu_phys = cap_mu_eff_si(
        physics_mu_eff_si(
            step.mu_c_si,
            sp,
            bio_cfg,
            device=device,
            data=data,
            u_nd=step.u_flow_nd,
            v_nd=step.v_flow_nd,
            phys_cfg=phys_cfg,
            time_index=time_index,
        )
    )
    region = step.region if apply_region else None
    phi_phys = physics_phi_from_mu(
        mu_phys,
        step.mu_c_si,
        region,
        phys_cfg,
        soft=use_soft,
        mu_anchor_si=mu_anchor_si,
    )
    return phi_phys.reshape(-1), mu_phys.reshape(-1)


def forward_clot_trigger_hybrid(
    model: nn.Module,
    step: ClotPhiStepBatch,
    data,
    *,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    edge_index: torch.Tensor | None = None,
    species_log1p: torch.Tensor | None = None,
    use_soft: bool = True,
    time_index: int | None = None,
    mu_anchor_si: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Hybrid trigger forward: physics gelation + learned phi (blended)."""
    sp = species_log1p if species_log1p is not None else step.species_log_gt
    phi_phys, mu_phys = forward_physics_trigger_phi(
        step,
        data,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        device=device,
        species_log1p=sp,
        use_soft=use_soft,
        apply_region=clot_phi_forward_apply_region() and not _env_bool("CLOT_TRIGGER_NUCLEATION", True),
        time_index=time_index,
        mu_anchor_si=mu_anchor_si,
    )
    logits = _model_logits(model, step.features, edge_index)
    phi_ml = torch.sigmoid(logits)
    if clot_phi_hybrid_enabled() and hasattr(model, "forward_delta_log_mu"):
        mu_ml = mu_eff_from_delta_log_si(
            step.mu_c_si, _model_delta_log_mu(model, step.features, edge_index)
        )
    else:
        mu_ml = log_blend_mu_eff_si(step.mu_c_si, phi_ml)

    if physics_blend_enabled():
        alpha = physics_blend_alpha()
        phi_hybrid = (alpha * phi_ml + (1.0 - alpha) * phi_phys).clamp(1e-6, 1.0 - 1e-6)
        mu_hybrid = alpha * mu_ml.reshape(-1) + (1.0 - alpha) * mu_phys.reshape(-1)
    else:
        phi_hybrid = phi_ml
        mu_hybrid = mu_ml.reshape(-1)

    return {
        "phi_phys": phi_phys.reshape(-1),
        "phi_ml": phi_ml.reshape(-1),
        "phi_hybrid": phi_hybrid.reshape(-1),
        "mu_phys": mu_phys.reshape(-1),
        "mu_ml": mu_ml.reshape(-1),
        "mu_hybrid": mu_hybrid.reshape(-1),
    }


def forward_trigger_rollout_step(
    model: nn.Module,
    data,
    time_index: int,
    *,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    phi_prev: torch.Tensor | None,
    phi_pred_by_time: dict[int, torch.Tensor],
    mu_anchor_si: torch.Tensor | None = None,
    edge_index: torch.Tensor | None = None,
    species_log1p: torch.Tensor | None = None,
    use_soft: bool = True,
    growth_seed: str | None = None,
    hard_commit: bool | None = None,
) -> dict[str, Any]:
    """One deploy-faithful trigger step: hybrid forward + nucleation projection."""
    from src.core_physics.clot_trigger_rollout import (
        _apply_ic_phi_zero,
        _project_step_phi,
        clot_trigger_forward_seed_mode,
    )

    t = int(time_index)
    step = build_clot_phi_step(data, t, phys_cfg, bio_cfg, device)
    bundle = forward_clot_trigger_hybrid(
        model,
        step,
        data,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        device=device,
        edge_index=edge_index,
        species_log1p=species_log1p,
        use_soft=use_soft,
        time_index=t,
        mu_anchor_si=mu_anchor_si,
    )
    mu_anchor_out = mu_anchor_si
    if mu_anchor_out is None:
        mu_anchor_out = bundle["mu_phys"].reshape(-1).detach().clone()
    seed_raw = growth_seed if growth_seed is not None else clot_trigger_forward_seed_mode()
    seed: str = "gt" if seed_raw == "gt" else "pred"
    phi_prev_in = (
        phi_prev.detach()
        if phi_prev is not None and clot_phi_trigger_rollout_detach_prev()
        else phi_prev
    )
    phi_raw = bundle["phi_hybrid"]
    phi_proj = _project_step_phi(
        phi_raw,
        phi_prev_in,
        data,
        t,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        phi_pred_by_time=phi_pred_by_time,
        growth_seed=seed,  # type: ignore[arg-type]
        hard_commit=hard_commit,
    )
    n_nodes = int(data.num_nodes)
    phi_proj, phi_raw = _apply_ic_phi_zero(phi_proj, phi_raw, t, n_nodes, device)
    return {
        "step": step,
        "bundle": bundle,
        "phi": phi_proj.reshape(-1),
        "phi_raw": phi_raw.reshape(-1),
        "mu_anchor_si": mu_anchor_out,
    }


def apply_clot_trigger_deploy_env() -> None:
    """Deploy-faithful default: pred forward E(tau), fixed ceiling loss, growth-only GT labels."""
    apply_neighbor_band_trigger_env()
    os.environ.setdefault("CLOT_TRIGGER_IC_PHI_ZERO", "1")
    os.environ["CLOT_PHI_LOSS_SCOPE"] = "ceiling"
    os.environ["CLOT_PHI_GROWTH_SEED"] = "pred"
    os.environ["CLOT_PHI_CLOT_SEED_SOURCE"] = "wall"
    os.environ["CLOT_PHI_DGAMMA_SLICE"] = "0"
    os.environ["CLOT_PHI_ORACLE_MU"] = "0"
    os.environ.setdefault("CLOT_PHI_CEILING_HOPS", "3")
    apply_clot_trigger_nucleation_env()


def apply_clot_trigger_honest_env() -> None:
    """Alias for deploy-faithful env (legacy name kept for launchers/tests)."""
    apply_clot_trigger_deploy_env()


def apply_clot_trigger_oracle_debug_env() -> None:
    """Explicit oracle: GT growth expansion for loss + GT forward envelope (upper bound only)."""
    apply_neighbor_band_trigger_env()
    os.environ.setdefault("CLOT_TRIGGER_IC_PHI_ZERO", "1")
    os.environ["CLOT_PHI_LOSS_SCOPE"] = "support"
    os.environ["CLOT_PHI_SUPPORT_BAND"] = "ceiling_growth"
    os.environ["CLOT_PHI_GROWTH_SEED"] = "gt"
    os.environ["CLOT_PHI_CLOT_SEED_SOURCE"] = "gt_mu"
    os.environ["CLOT_PHI_DGAMMA_SLICE"] = "1"
    apply_clot_trigger_oracle_forward_env()


def clot_phi_trigger_rollout_enabled() -> bool:
    return _env_bool("CLOT_PHI_TRIGGER_ROLLOUT", False)


def clot_phi_trigger_rollout_detach_prev() -> bool:
    raw = (os.environ.get("CLOT_PHI_TRIGGER_TBPTT_DETACH") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def deploy_env_is_faithful() -> bool:
    """True when loss/forward avoid GT commit leakage (deploy contract)."""
    from src.core_physics.clot_growth_masks import growth_seed_mode
    from src.core_physics.clot_phi_simple import clot_phi_loss_scope
    from src.core_physics.clot_trigger_rollout import forward_path_uses_gt_commits

    return (
        growth_seed_mode() == "pred"
        and not forward_path_uses_gt_commits()
        and clot_phi_loss_scope() in ("ceiling", "nucleation")
    )


def apply_clot_trigger_nucleation_env() -> None:
    """Deploy-faithful forward: wall + 1-hop from **predicted** commits each step."""
    os.environ["CLOT_TRIGGER_NUCLEATION"] = "1"
    os.environ["CLOT_TRIGGER_FORWARD_SEED"] = "pred"
    os.environ.setdefault("CLOT_V2_NUCLEATION_HOPS", "1")
    os.environ.setdefault("CLOT_TRIGGER_COMMIT_THRESH", "0.5")
    os.environ.setdefault("CLOT_TRIGGER_DGAMMA_WALL_SEED", "0")


def apply_clot_trigger_oracle_forward_env() -> None:
    """Debug: forward envelope seeded from GT commits (upper-bound diagnostic only)."""
    apply_clot_trigger_nucleation_env()
    os.environ["CLOT_TRIGGER_FORWARD_SEED"] = "gt"


def apply_deploy_nucleation_mask_env() -> None:
    """Loss on deploy nucleation band only (geometry wall+hops; no GT mu seeds)."""
    apply_neighbor_band_trigger_env()
    os.environ["CLOT_PHI_LOSS_SCOPE"] = "nucleation"
    os.environ["CLOT_PHI_CLOT_SEED_SOURCE"] = "wall"
    os.environ["CLOT_PHI_DGAMMA_SLICE"] = "0"


def apply_oracle_neighbor_mask_env() -> None:
    """Legacy debug: GT mu seeds @ t + GT-flow dgamma slice for loss."""
    apply_neighbor_band_trigger_env()
    os.environ["CLOT_PHI_LOSS_SCOPE"] = "oracle"
    os.environ["CLOT_PHI_CLOT_SEED_SOURCE"] = "gt_mu"
    os.environ["CLOT_PHI_DGAMMA_SLICE"] = "1"


def apply_neighbor_band_trigger_env() -> None:
    """Shared neighbor-band geometry defaults (seed/loss scope set by honest/oracle helpers)."""
    defaults = {
        "CLOT_PHI_MASK_MODE": "neighbor",
        "CLOT_PHI_WALL_HOPS": "1",
        "CLOT_PHI_CLOT_HOPS": "2",
        "CLOT_PHI_CLOT_TOUCH_HOPS": "1",
        "CLOT_PHI_CENTER_EXCLUDE_FRAC": "0.10",
        "CLOT_PHI_DGAMMA_REF_TIME": "0",
        "CLOT_PHI_DGAMMA_WALL_MIN_SI": "100",
        "CLOT_PHI_DGAMMA_OFFWALL_PCT": "80",
        "CLOT_PHI_MU_CAP_SI": "0.10",
        "CLOT_PHI_THRESH_SI": "0.055",
        "CLOT_PHI_SHEAR_MIN_FRAC": "0",
    }
    for key, val in defaults.items():
        os.environ.setdefault(key, val)


def apply_star1_train_env(*, fast: bool = False) -> None:
    """T1: GT flow + GT species hybrid trigger training (honest full-mesh loss)."""
    apply_clot_trigger_honest_env()
    paths = trigger_paths_for_star(ClotTriggerStar.T1_GT_INPUTS)
    os.environ["CLOT_TRIGGER_STAR"] = ClotTriggerStar.T1_GT_INPUTS.value
    os.environ["CLOT_PHI_SWEEP_DIR"] = str(paths.out_dir.parent.relative_to(get_project_root()))
    os.environ["CLOT_PHI_SWEEP_LEG"] = paths.out_dir.name
    os.environ["CLOT_PHI_CKPT_NAME"] = paths.ckpt_name
    os.environ["CLOT_PHI_LOG_NAME"] = paths.log_name

    # Inputs: GT flow from y (default build_clot_phi_step); GT FI/Mat in features.
    os.environ["CLOT_PHI_ORACLE_MU"] = "0"
    os.environ["CLOT_PHI_SPECIES_FEATURES"] = "1"
    os.environ["CLOT_PHI_JOINT_BIO"] = "0"
    os.environ["CLOT_PHI_JOINT_USE_PRED_SPECIES"] = "0"
    os.environ["CLOT_PHI_USE_PRIOR_FEATURES"] = "0"
    os.environ["CLOT_PHI_ROLLOUT"] = "0"
    os.environ["CLOT_PHI_TRIGGER_ROLLOUT"] = "1"
    os.environ.setdefault("CLOT_PHI_TRIGGER_TBPTT_DETACH", "1")
    os.environ.setdefault("CLOT_TRIGGER_TRAIN_SOFT_COMMIT", "1")
    os.environ["CLOT_PHI_MINIMAL_FEATURES"] = "1"
    os.environ["CLOT_PHI_DGAMMA_FEATURE_TIME"] = "current"

    # Hybrid trigger: physics gelation floor + learned residual.
    os.environ["CLOT_PHI_HYBRID"] = "1"
    os.environ["CLOT_PHI_PHYSICS_BLEND"] = "1"
    os.environ["CLOT_PHI_PHYSICS_BLEND_ALPHA"] = "0.55"
    os.environ["CLOT_PHI_PHYSICS_MU_RATIO_MAX"] = "4"
    os.environ["CLOT_PHI_PHYSICS_GELATION_GATE"] = "0"
    os.environ.setdefault("CLOT_PHI_MESH_BULK_LAMBDA", "0.15")

    # Train recipe.
    os.environ["CLOT_PHI_MODEL"] = "mlp"
    os.environ["CLOT_PHI_HIDDEN"] = "32"
    os.environ["CLOT_PHI_MLP_DEPTH"] = "2"
    os.environ["CLOT_PHI_DROPOUT"] = "0.12"
    os.environ["CLOT_PHI_SOFT_LABELS"] = "1"
    os.environ["CLOT_PHI_BALANCED"] = "1"
    os.environ["CLOT_PHI_POS_WEIGHT_CAP"] = "8"
    os.environ["CLOT_PHI_ANCHOR_BALANCED"] = "1"
    os.environ["CLOT_PHI_MU_LOG_LAMBDA"] = "1.25"
    os.environ["CLOT_PHI_DICE_LAMBDA"] = "0.25"
    os.environ["CLOT_PHI_LR"] = "1e-3"
    os.environ["CLOT_PHI_WEIGHT_DECAY"] = "1e-4"
    os.environ["CLOT_PHI_TIME_STRIDE"] = "2"
    os.environ["CLOT_PHI_TIME_STRIDE_AUTO"] = "1"
    os.environ["CLOT_PHI_VAL_ANCHOR"] = "patient007"
    os.environ["CLOT_PHI_EPOCHS"] = "16" if fast else "48"


def apply_star2_eval_env(
    *,
    kine_ckpt: str = "outputs/kinematics/kinematics_best.pth",
) -> None:
    """T2: frozen T1 trigger ckpt, pred GINO-DEQ flow + GT species."""
    apply_clot_trigger_honest_env()
    os.environ["CLOT_TRIGGER_STAR"] = ClotTriggerStar.T2_PRED_FLOW.value
    os.environ["CLOT_PHI_VEL_SOURCE"] = "kinematics"
    os.environ["CLOT_TEMPORAL_VEL_SOURCE"] = "kinematics"
    os.environ["CLOT_PHI_KINE_CKPT"] = kine_ckpt
    os.environ["CLOT_PHI_KINE_TF"] = "0"
    os.environ["CLOT_PHI_ROLLOUT"] = "0"
    os.environ["CLOT_PHI_ORACLE_MU"] = "0"
    os.environ["CLOT_PHI_SPECIES_FEATURES"] = "1"
    os.environ["CLOT_PHI_JOINT_BIO"] = "0"
    os.environ["CLOT_PHI_MINIMAL_FEATURES"] = "1"
    os.environ["CLOT_PHI_HYBRID"] = "1"
    os.environ["CLOT_PHI_PHYSICS_BLEND"] = "1"
    os.environ["CLOT_PHI_PHYSICS_BLEND_ALPHA"] = "0.55"
    os.environ["CLOT_PHI_PHYSICS_MU_RATIO_MAX"] = "4"
    os.environ["CLOT_PHI_SOFT_LABELS"] = "1"


def reset_star2_kinematics_cache() -> None:
    """Clear cached steady GINO-DEQ uv between anchors."""
    from src.core_physics.clot_temporal_growth_rules import reset_temporal_kinematics_cache

    reset_temporal_kinematics_cache()


def default_t1_checkpoint_path() -> Path:
    return trigger_paths_for_star(ClotTriggerStar.T1_GT_INPUTS).ckpt_path


def default_teacher_checkpoint_path() -> Path:
    root = get_project_root()
    for rel in (
        "outputs/biochem/biochem_teacher_best_high_mu.pth",
        "outputs/biochem/biochem_teacher_last.pth",
        "outputs/biochem/sweep_mu_complexity_6h/FULL_step2/biochem_teacher_best_high_mu.pth",
    ):
        path = root / rel
        if path.is_file():
            return path
    return root / "outputs/biochem/biochem_teacher_best_high_mu.pth"


def default_dumped_species_anchor_dir() -> Path:
    return get_project_root() / "outputs" / "biochem" / "anchors_teacher_species"


@dataclass(frozen=True)
class ClotTriggerT5DeployPaths:
    """Artifacts for Star 5 deploy teacher retrain + pred-flow species cache."""

    out_root: Path

    @property
    def teacher_deploy(self) -> Path:
        return self.out_root / "biochem_teacher_deploy.pth"

    @property
    def manifest(self) -> Path:
        return self.out_root / "manifest.json"

    @property
    def eval_live_json(self) -> Path:
        return self.out_root / "t5_deploy_live.json"

    @property
    def eval_dumped_json(self) -> Path:
        return self.out_root / "t5_deploy_dumped.json"


def default_t5_deploy_paths() -> ClotTriggerT5DeployPaths:
    return ClotTriggerT5DeployPaths(
        out_root=get_project_root() / "outputs" / "biochem" / "clot_trigger" / "t5_deploy_teacher"
    )


def default_t5_predkine_species_dump_dir() -> Path:
    return get_project_root() / "outputs" / "biochem" / "anchors_teacher_species_predkine"


def default_t5_deploy_teacher_checkpoint_path() -> Path:
    """Promoted T5 deploy teacher, else global biochem best."""
    path = default_t5_deploy_paths().teacher_deploy
    if path.is_file():
        return path
    return default_teacher_checkpoint_path()


def default_gt_anchor_dir() -> Path:
    from src.config import VesselConfig

    return get_project_root() / VesselConfig(phase="biochem_anchors").graph_output_dir


def apply_star3_dumped_env(
    *,
    kine_ckpt: str = "outputs/kinematics/kinematics_best.pth",
    dump_dir: str | None = None,
) -> None:
    """T3: pred kine + cached teacher species (``dump_teacher_species_to_anchors``)."""
    apply_star2_eval_env(kine_ckpt=kine_ckpt)
    os.environ["CLOT_TRIGGER_STAR"] = ClotTriggerStar.T3_DUMPED_SPECIES.value
    os.environ["CLOT_TRIGGER_SPECIES_SOURCE"] = "dumped"
    if dump_dir:
        os.environ["CLOT_TRIGGER_DUMPED_ANCHOR_DIR"] = dump_dir


def apply_star4_live_teacher_env(
    *,
    kine_ckpt: str = "outputs/kinematics/kinematics_best.pth",
    teacher_ckpt: str | None = None,
) -> None:
    """T4: pred kine + live GraphSAGE species rollout (slow; overnight path)."""
    apply_star2_eval_env(kine_ckpt=kine_ckpt)
    os.environ["CLOT_TRIGGER_STAR"] = ClotTriggerStar.T4_LIVE_TEACHER.value
    os.environ["CLOT_TRIGGER_SPECIES_SOURCE"] = "live"
    os.environ["BIOCHEM_GT_KINE_VEL"] = "0"
    os.environ["BIOCHEM_GT_KINE_SKIP_DEQ"] = "0"
    os.environ["BIOCHEM_VAL_TIME_STRIDE"] = "1"
    os.environ.setdefault("BIOCHEM_DATA_BIO_SPECIES_SCOPE", "fi_mat")
    os.environ.setdefault("BIOCHEM_DATALOADER_WORKERS", "0")
    if teacher_ckpt:
        os.environ["CLOT_TRIGGER_TEACHER_CKPT"] = teacher_ckpt


def apply_star5_deploy_teacher_eval_env(
    *,
    kine_ckpt: str = "outputs/kinematics/kinematics_best.pth",
    teacher_ckpt: str | None = None,
) -> None:
    """T5 eval: pred kine + live/deploy-retrained GraphSAGE species (``BIOCHEM_GT_KINE_VEL=0``)."""
    ckpt = teacher_ckpt or str(default_t5_deploy_teacher_checkpoint_path())
    apply_star4_live_teacher_env(kine_ckpt=kine_ckpt, teacher_ckpt=ckpt)
    os.environ["CLOT_TRIGGER_STAR"] = ClotTriggerStar.T5_DEPLOY_TEACHER.value


def apply_star5_deploy_dumped_eval_env(
    *,
    kine_ckpt: str = "outputs/kinematics/kinematics_best.pth",
    dump_dir: str | None = None,
) -> None:
    """T5 fast eval: pred kine + pred-flow species dump from deploy teacher."""
    apply_star3_dumped_env(
        kine_ckpt=kine_ckpt,
        dump_dir=dump_dir or str(default_t5_predkine_species_dump_dir()),
    )
    os.environ["CLOT_TRIGGER_STAR"] = ClotTriggerStar.T5_DEPLOY_TEACHER.value


def snapshot_t5_deploy_train_config() -> dict[str, Any]:
    """Record env knobs that define the T5 deploy teacher recipe."""
    return {
        "clot_trigger_star": ClotTriggerStar.T5_DEPLOY_TEACHER.value,
        "gt_kine_vel": os.environ.get("BIOCHEM_GT_KINE_VEL", ""),
        "species_scope": os.environ.get("BIOCHEM_DATA_BIO_SPECIES_SCOPE", ""),
        "bio_mask_mode": os.environ.get("BIOCHEM_DATA_BIO_MASK_MODE", ""),
        "teacher_epochs": os.environ.get("BIOCHEM_TEACHER_EPOCHS", ""),
        "run_note": os.environ.get("BIOCHEM_RUN_NOTE", ""),
        "kine_ckpt": os.environ.get("CLOT_PHI_KINE_CKPT", "outputs/kinematics/kinematics_best.pth"),
    }


def apply_star6_coupled_env(
    *,
    kine_ckpt: str = "outputs/kinematics/kinematics_best.pth",
    teacher_ckpt: str | None = None,
    species_live: bool = True,
    dump_dir: str | None = None,
) -> None:
    """T6: T4/T5 species + serial phi/mu -> GINO-DEQ MU_PRIOR feedback.

    Uses frozen T1 trigger (in_dim=5): no carry features in MLP; mu feedback via
    ``CLOT_PHI_FIXED_MU_FROM_PHI`` + ``KinematicsUvProvider`` (Step 5b pattern).
    Species remain offline/live teacher output (not re-rolled per macro step).
    """
    t5_teacher = str(default_t5_deploy_teacher_checkpoint_path())
    resolved_teacher = teacher_ckpt or t5_teacher
    if species_live:
        apply_star5_deploy_teacher_eval_env(kine_ckpt=kine_ckpt, teacher_ckpt=resolved_teacher)
    else:
        predkine = Path(dump_dir) if dump_dir else default_t5_predkine_species_dump_dir()
        if predkine.is_dir() and any(predkine.glob("*.pt")):
            apply_star5_deploy_dumped_eval_env(kine_ckpt=kine_ckpt, dump_dir=str(predkine))
        else:
            apply_star3_dumped_env(
                kine_ckpt=kine_ckpt,
                dump_dir=dump_dir or str(default_dumped_species_anchor_dir()),
            )
    os.environ["CLOT_TRIGGER_STAR"] = ClotTriggerStar.T6_COUPLED.value
    os.environ["CLOT_PHI_ROLLOUT"] = "1"
    os.environ["CLOT_PHI_ROLLOUT_DETACH"] = "1"
    os.environ["CLOT_PHI_CARRY_PHI"] = "0"
    os.environ["CLOT_PHI_CARRY_LOG_MU"] = "0"
    os.environ["CLOT_PHI_FIXED_MU_FROM_PHI"] = "1"
    os.environ["CLOT_TEMPORAL_VEL_SOURCE"] = "coupled"
    os.environ["CLOT_PHI_VEL_SOURCE"] = "kinematics"


def reset_star6_caches() -> None:
    """Clear frozen-kine, coupled-uv, and rollout GINO-DEQ caches."""
    from src.core_physics.clot_coupled_rollout import reset_coupled_uv_cache
    from src.core_physics.clot_phi_rollout import reset_rollout_kine_provider

    reset_star3_caches()
    reset_coupled_uv_cache()
    reset_rollout_kine_provider()


# Backward-compatible aliases (pre-T6 rename).
apply_star5_coupled_env = apply_star6_coupled_env
reset_star5_caches = reset_star6_caches


def init_coupled_trigger_rollout(
    data,
    *,
    device: torch.device,
) -> tuple["ClotPhiRolloutState", "KinematicsUvProvider"]:
    """Seed coupled loop: fluid phi=0 -> mu_c steady kine -> coupled uv cache."""
    from src.core_physics.clot_coupled_rollout import (
        _init_coupled_uv_from_frozen_kine,
        reset_coupled_uv_cache,
    )
    from src.core_physics.clot_phi_rollout import ClotPhiRolloutState, KinematicsUvProvider

    reset_coupled_uv_cache()
    _init_coupled_uv_from_frozen_kine(data, device)
    n_nodes = int(data.num_nodes)
    rollout_state = ClotPhiRolloutState(
        phi_prev=torch.zeros(n_nodes, device=device, dtype=torch.float32),
        log_mu_prev=None,
    )
    provider = KinematicsUvProvider(device)
    return rollout_state, provider


@torch.no_grad()
def advance_coupled_trigger_state(
    data,
    phi_pred: torch.Tensor,
    mu_pred_si: torch.Tensor,
    *,
    rollout_state: "ClotPhiRolloutState",
    kine_provider: "KinematicsUvProvider",
    detach: bool = True,
) -> None:
    """Commit phi/mu and refresh GINO-DEQ [u,v] with ``MU_PRIOR`` from predicted mu."""
    from src.core_physics.clot_coupled_rollout import set_coupled_uv_cache

    rollout_state.update_from_pred(phi_pred, mu_pred_si, detach=detach)
    u, v = kine_provider.uv_nd_from_mu_si(data, mu_pred_si)
    set_coupled_uv_cache(data, u, v)


def build_clot_trigger_coupled_step(
    data,
    time_index: int,
    *,
    pred_species_series: torch.Tensor | None,
    rollout_state: "ClotPhiRolloutState",
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
) -> ClotPhiStepBatch:
    """Coupled clot step: pred species + flow from prior-step phi/mu feedback."""
    species_override = None
    if pred_species_series is not None:
        ti = max(0, min(int(time_index), int(pred_species_series.shape[0]) - 1))
        species_override = pred_species_series[ti, :, sc.SPECIES_BLOCK]
    return build_clot_phi_step(
        data,
        time_index,
        phys_cfg,
        bio_cfg,
        device,
        species_log_override=species_override,
        rollout_state=rollout_state,
    )


def apply_star3_eval_env(
    *,
    kine_ckpt: str = "outputs/kinematics/kinematics_best.pth",
    teacher_ckpt: str | None = None,
) -> None:
    """Backward-compatible alias for live teacher (T4) eval."""
    apply_star4_live_teacher_env(kine_ckpt=kine_ckpt, teacher_ckpt=teacher_ckpt)


def load_trigger_model(ckpt_path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    from src.core_physics.clot_phi_simple import build_clot_phi_model, clot_phi_feature_dim
    from src.evaluation.clot_phi_checkpoint_env import apply_clot_phi_config_from_checkpoint

    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = dict(raw.get("config") or {})
    apply_clot_phi_config_from_checkpoint(cfg)
    hidden = int(cfg.get("hidden", 32))
    in_dim = int(cfg.get("in_dim", clot_phi_feature_dim()))
    model = build_clot_phi_model(in_dim=in_dim, hidden=hidden).to(device)
    model.load_state_dict(raw["model_state_dict"], strict=True)
    model.eval()
    return model, cfg


def load_teacher_for_trigger(device: torch.device, teacher_ckpt: Path | None = None):
    """Load the GraphSAGE species pushforward bundle used to seed the clot trigger.

    Pre-2026-06 this built a GNODE teacher; the trigger now rolls species via the
    ``biochem_gnn`` GraphSAGE stack. The first return value is the species
    bundle (named ``teacher`` for back-compat with the trigger eval/viz scripts).
    """
    from src.config import BiochemConfig, PhysicsConfig
    from src.core_physics.species_gnn_clot_rollout import (
        load_species_gnn_rollout_bundle,
        species_gnn_rollout_ckpt,
    )

    phys_cfg = PhysicsConfig(phase="biochem")
    bio_cfg = BiochemConfig(phase="biochem")
    path = Path(teacher_ckpt) if teacher_ckpt else None
    if path is None or not path.is_file():
        from src.biochem_gnn.config import global_ckpt_path

        path = global_ckpt_path()
        if not path.is_file():
            path = species_gnn_rollout_ckpt()
    if not path.is_file():
        raise FileNotFoundError(f"species GNN checkpoint not found: {path}")
    bundle = load_species_gnn_rollout_bundle(path, device=device)
    if bundle is None:
        raise FileNotFoundError(f"failed to load species GNN bundle: {path}")
    return bundle, bio_cfg, phys_cfg, path


@torch.no_grad()
def rollout_teacher_species_series(data, teacher, bio_cfg: BiochemConfig, device: torch.device) -> torch.Tensor:
    """GraphSAGE species pushforward; returns ``(T, N, 16)`` on the ``data.y`` macro grid.

    ``teacher`` is a ``SpeciesGnnRolloutBundle`` (see ``load_teacher_for_trigger``).
    Only the species block (channels 4:16) is populated.
    """
    from src.config import PhysicsConfig
    from src.core_physics.species_gnn_clot_rollout import (
        prepare_species_gnn_rollout_static,
        rollout_species_gnn_species_series,
    )

    phys_cfg = PhysicsConfig(phase="biochem")
    static = prepare_species_gnn_rollout_static(data, device=device)
    return rollout_species_gnn_species_series(
        data, teacher, static, phys_cfg=phys_cfg, bio_cfg=bio_cfg, device=device,
    )


def build_clot_trigger_step_at_time(
    data,
    time_index: int,
    *,
    pred_species_series: torch.Tensor | None,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
) -> ClotPhiStepBatch:
    """Deploy-faithful clot step: GT labels, optional pred species for forward."""
    species_override = None
    if pred_species_series is not None:
        ti = max(0, min(int(time_index), int(pred_species_series.shape[0]) - 1))
        species_override = pred_species_series[ti, :, sc.SPECIES_BLOCK]
    return build_clot_phi_step(
        data,
        time_index,
        phys_cfg,
        bio_cfg,
        device,
        species_log_override=species_override,
    )


def reset_star3_caches() -> None:
    reset_star2_kinematics_cache()


def rollout_trigger_phi_gt_inputs(
    model: nn.Module,
    data,
    *,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    time_stride: int = 1,
    mode: str = "hybrid",
) -> dict[int, torch.Tensor]:
    """Macro trajectory with GT flow + GT species and nucleation projection."""
    from src.core_physics.clot_trigger_rollout import rollout_clot_trigger_hybrid, rollout_clot_trigger_physics

    if mode == "physics":
        traj = rollout_clot_trigger_physics(
            data,
            phys_cfg=phys_cfg,
            bio_cfg=bio_cfg,
            device=device,
            time_stride=time_stride,
        )
        return {t: v["phi"] for t, v in traj.items()}

    model.eval()
    traj = rollout_clot_trigger_hybrid(
        model,
        data,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        device=device,
        time_stride=time_stride,
        mode=mode,
    )
    key = "phi"
    return {t: v[key] for t, v in traj.items()}


def snapshot_trigger_train_config() -> dict[str, Any]:
    from src.core_physics.clot_phi_simple import clot_phi_loss_scope
    from src.core_physics.clot_trigger_rollout import (
        clot_trigger_forward_seed_mode,
        clot_trigger_nucleation_enabled,
        snapshot_clot_trigger_rollout_config,
    )

    return {
        "clot_trigger_star": os.environ.get("CLOT_TRIGGER_STAR", ""),
        "physics_blend": physics_blend_enabled(),
        "physics_blend_alpha": physics_blend_alpha(),
        "species_features": _env_bool("CLOT_PHI_SPECIES_FEATURES"),
        "joint_bio": _env_bool("CLOT_PHI_JOINT_BIO"),
        "loss_scope": clot_phi_loss_scope(),
        "trigger_rollout": clot_phi_trigger_rollout_enabled(),
        "forward_seed": clot_trigger_forward_seed_mode(),
        "nucleation": clot_trigger_nucleation_enabled(),
        **snapshot_clot_trigger_rollout_config(),
    }
