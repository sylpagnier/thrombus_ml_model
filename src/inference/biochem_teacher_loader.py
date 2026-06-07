"""Load GNODE_Phase3 biochem teacher checkpoints for rollout / dump / inference."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from src.architecture.gnode_biochem import (
    GNODE_Phase3,
    apply_biochem_forward_policy_from_checkpoint_meta,
    resolve_gnode_phase3_ctor_kwargs,
)
from src.config import BiochemConfig, PhysicsConfig


def resolve_rollout_mu_ratio_max(
    bio_cfg: BiochemConfig,
    *,
    cli_value: float | None = None,
) -> float:
    """COMSOL mu1/mu2 step ceiling for rollout (not mu_eff ratio)."""
    if cli_value is not None:
        return max(float(cli_value), 1.0)
    raw = (os.environ.get("BIOCHEM_TEACHER_MU_RATIO_MAX") or "").strip()
    if raw:
        return max(float(raw), 1.0)
    return max(float(getattr(bio_cfg, "mu_ratio_max", 80.0)), 1.0)


def build_biochem_teacher(
    ckpt: dict[str, Any],
    *,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    mu_ratio_max: float,
    quiet: bool = False,
) -> GNODE_Phase3:
    state_dict = ckpt.get("model_state_dict") or ckpt
    bio_prior_default = max(0, int(os.environ.get("BIOCHEM_BIO_ENCODER_PRIOR_DIM", "2") or "2"))
    ctor = resolve_gnode_phase3_ctor_kwargs(
        ckpt,
        state_dict,
        bio_encoder_prior_dim_default=bio_prior_default,
        latent_dim_default=256,
        fourier_bands_default=16,
        use_siren_default=True,
        gnode_layers_default=1,
        max_inner_iters_default=10,
    )
    teacher = GNODE_Phase3(
        phys_cfg=phys_cfg,
        in_channels=int(ctor["in_channels"]),
        spatial_channels=int(ctor["spatial_channels"]),
        latent_dim=int(ctor["latent_dim"]),
        max_inner_iters=max(3, int(ctor.get("max_inner_iters", 10))),
        bio_encoder_prior_dim=int(ctor["bio_encoder_prior_dim"]),
        mu_ratio_max=mu_ratio_max,
        mat_crit=float(bio_cfg.viscosity_mat_crit),
        fi_crit=float(bio_cfg.viscosity_fi_crit),
        temp_mat=float(bio_cfg.viscosity_gnode_temp_mat),
        temp_fi=float(bio_cfg.viscosity_gnode_temp_fi),
        num_fourier_freqs=int(ctor["num_fourier_freqs"]),
        use_siren_decoder=bool(ctor["use_siren_decoder"]),
        gnode_layers=int(ctor["gnode_layers"]),
        use_hard_bcs=bool(ctor["use_hard_bcs"]),
    ).to(device)
    teacher.load_state_dict(state_dict, strict=False)
    apply_biochem_forward_policy_from_checkpoint_meta(ckpt, quiet=quiet)
    teacher.eval()
    if not quiet:
        print(
            f"[i]  GNODE teacher: in={int(ctor['in_channels'])} spatial={int(ctor['spatial_channels'])} "
            f"prior={int(ctor['bio_encoder_prior_dim'])} latent={int(ctor['latent_dim'])} "
            f"mu_ratio_max={mu_ratio_max:g}",
            flush=True,
        )
    return teacher


def load_biochem_teacher_checkpoint(
    teacher_path: str | Path,
    device: torch.device,
    *,
    mu_ratio_max: float | None = None,
    pred_kine: bool | None = None,
    quiet: bool = False,
) -> tuple[GNODE_Phase3, dict[str, Any], float]:
    """Load teacher from ``.pth`` and apply rollout env (pred kine, mu_ratio cap)."""
    path = Path(teacher_path)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    phys_cfg = PhysicsConfig(phase="biochem")
    bio_cfg = BiochemConfig(phase="biochem")
    ratio = resolve_rollout_mu_ratio_max(bio_cfg, cli_value=mu_ratio_max)
    os.environ["BIOCHEM_TEACHER_MU_RATIO_MAX"] = f"{ratio:g}"
    if pred_kine is None:
        pred_kine = (os.environ.get("BIOCHEM_GT_KINE_VEL") or "0").strip() != "1"
    if pred_kine:
        os.environ["BIOCHEM_GT_KINE_VEL"] = "0"
    else:
        os.environ["BIOCHEM_GT_KINE_VEL"] = "1"
    teacher = build_biochem_teacher(
        ckpt,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        device=device,
        mu_ratio_max=ratio,
        quiet=quiet,
    )
    return teacher, ckpt, ratio
