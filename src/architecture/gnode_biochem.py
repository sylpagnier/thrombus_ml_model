from __future__ import annotations

import math
import os
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch import Tensor
from torch_geometric.utils import degree
# ``odeint`` OOMs on large graphs; ``odeint_adjoint`` fixes that. For short TBPTT teacher
# windows, ``BIOCHEM_ODEINT_USE_ADJOINT=0`` uses dense ``odeint`` backward (more VRAM,
# often stabler than adjoint on stiff segments / low RK substep counts).
from torchdiffeq import odeint, odeint_adjoint
from src.architecture.ginodeq import GINOBlock, SpectralLinear
from src.architecture.siren_decoder import SIRENDecoder
from src.config import BiochemConfig, NodeFeat, PredChannels
from src.core_physics.anderson import anderson_acceleration
from src.core_physics.kinematics_clot_prior import clot_prior_features, clot_prior_score_flat
from src.utils.batching import get_batch_tensor

# Matches BiochemPhysicsKernels: species channels are log1p(species_nd).
_SPECIES_LOG1P_MIN = -10.0
_SPECIES_LOG1P_MAX = 8.0


def _biochem_ode_grad_checkpoint_enabled() -> bool:
    """Recompute GINO derivative block during backward to lower peak VRAM (more compute).

    Set ``BIOCHEM_ODE_GRADIENT_CHECKPOINT=1`` during biochem / teacher training on tight GPUs.
    """
    v = (os.environ.get("BIOCHEM_ODE_GRADIENT_CHECKPOINT", "0") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _biochem_kin_grad_checkpoint_enabled() -> bool:
    """Recompute kinematic GINO stack in backward to reduce VRAM on low-memory GPUs."""
    v = (os.environ.get("BIOCHEM_KIN_GRADIENT_CHECKPOINT", "0") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _biochem_gelation_prior_gate_enabled() -> bool:
    """When true (default), scale FI/Mat + learned gelation by wall-local kinematic clot-risk prior.

    Set ``BIOCHEM_GELATION_PRIOR_GATE=0`` to restore legacy behaviour (gelation can lift μ everywhere).
    """
    raw = (os.environ.get("BIOCHEM_GELATION_PRIOR_GATE", "1") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _biochem_delta_mu_head_enabled() -> bool:
    """Enable residual viscosity correction head on top of analytic rheology.

    Set ``BIOCHEM_USE_DELTA_MU_HEAD=1`` to apply a bounded multiplicative correction.
    """
    raw = (os.environ.get("BIOCHEM_USE_DELTA_MU_HEAD", "0") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _biochem_split_mu_regime_head_enabled() -> bool:
    """Enable split residual log-μ heads with trigger-gated bulk/tail mixing."""
    raw = (os.environ.get("BIOCHEM_USE_SPLIT_MU_HEAD", "0") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _biochem_mu_gemini_fix_enabled() -> bool:
    """Additive bulk+tail log-μ residuals with symmetric bulk clip and nonnegative wall clip."""
    return _biochem_env_truthy("BIOCHEM_MU_GEMINI_FIX")


def _biochem_mu_additive_delta_enabled() -> bool:
    """Stack bulk and gated tail additively instead of (1-gate)*bulk + gate*tail."""
    if _biochem_mu_gemini_fix_enabled():
        return True
    return _biochem_env_truthy("BIOCHEM_MU_ADDITIVE_DELTA")


def _biochem_mu_simple_log_residual_enabled() -> bool:
    """Carreau baseline × exp(Δlogμ) only — no explicit μ₁/μ₂ gelation multiplier."""
    return _biochem_env_truthy("BIOCHEM_MU_SIMPLE_LOG_RESIDUAL")


def _biochem_mu_disable_explicit_gelation() -> bool:
    """Skip FI/Mat sigmoid gelation and learned clot penalty in the μ multiplier."""
    return _biochem_mu_simple_log_residual_enabled() or _biochem_env_truthy(
        "BIOCHEM_MU_DISABLE_EXPLICIT_GELATION"
    )


def _biochem_gt_kine_vel_enabled() -> bool:
    """Use COMSOL ``y`` channels ``[u,v,p]`` for macro kinematics instead of the frozen GINO-DEQ solve."""
    return _biochem_env_truthy("BIOCHEM_GT_KINE_VEL")


def _biochem_gt_kine_skip_deq() -> bool:
    """When GT kinematics are on, skip Anderson DEQ (default on — saves time and avoids bad frozen flow)."""
    return _biochem_env_truthy("BIOCHEM_GT_KINE_SKIP_DEQ", default=True)


def resolve_gt_kine_uvp_at_step(
    batch,
    y_true_trajectory: torch.Tensor | None,
    time_index: int,
    truth_mask: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    *,
    fallback_uvp: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Return COMSOL ND ``[u,v,p]`` at ``time_index``, blended with ``fallback_uvp`` off truth nodes."""
    yt = None
    if y_true_trajectory is not None and y_true_trajectory.ndim == 3 and int(y_true_trajectory.shape[-1]) >= 3:
        yt = y_true_trajectory
    elif hasattr(batch, "y") and batch.y is not None and batch.y.ndim == 3 and int(batch.y.shape[-1]) >= 3:
        yt = batch.y
    if yt is None:
        return None
    ti = min(max(int(time_index), 0), int(yt.shape[0]) - 1)
    gt = yt[ti, :, 0:3].to(device=device, dtype=dtype)
    if fallback_uvp is not None and truth_mask.any():
        fb = fallback_uvp.to(device=device, dtype=dtype)
        return torch.where(truth_mask.unsqueeze(-1), gt, fb)
    return gt


def _biochem_mu_ic_steady_kin_enabled() -> bool:
    """Bootstrap rollout ``μ_eff`` at macro step 0 from one-shot frozen-kin DEQ (viz steady panel).

    Also skips ``exp(Δlogμ)`` on step 0 so the stored channel matches steady kinematics μ,
    not graph ``MU_PRIOR`` + learned residual. Set ``BIOCHEM_MU_IC_STEADY_KIN=1``.
    """
    return _biochem_env_truthy("BIOCHEM_MU_IC_STEADY_KIN")


def _biochem_mu_k10d_simple_enabled() -> bool:
    """K10d proof: ``μ_eff = μ_ss + softplus(learned_Δμ_SI)``; overrides Carreau/exp path each step."""
    return (
        _biochem_env_truthy("BIOCHEM_MU_K10D_SIMPLE")
        and not _biochem_mu_k10e_simple_enabled()
    )


def _biochem_mu_k10e_simple_enabled() -> bool:
    """K10e: ``μ_eff = μ_ss + adj_mask * softplus(Δμ_nd)*scale``; clots only in wall-adjacent band."""
    return _biochem_env_truthy("BIOCHEM_MU_K10E_SIMPLE")


def _biochem_k10g_oracle_clots_enabled() -> bool:
    """Sanity: in the wall-adjacent band, set ``μ_eff`` from GT excess over ``μ_ss`` (needs ``y_true_trajectory``)."""
    return _biochem_env_truthy("BIOCHEM_K10G_ORACLE_CLOTS")


def _k10e_env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def k10e_wall_adjacent_mask(
    sdf_nd: torch.Tensor,
    wall_mask: torch.Tensor,
    *,
    d_peak_nd: float | None = None,
    sigma_nd: float | None = None,
    sdf_max_nd: float | None = None,
) -> torch.Tensor:
    """Smooth mask in ``[0,1]`` peaked one layer off the wall, zero on ``mask_wall`` nodes.

    ``sdf_nd`` is distance to nearest wall (0 at wall, positive in lumen). Clots nucleate in
    the boundary layer, not on the solid wall nodes themselves.
    """
    d_peak = _k10e_env_float("BIOCHEM_K10E_D_PEAK_ND", 0.004) if d_peak_nd is None else d_peak_nd
    sigma = max(_k10e_env_float("BIOCHEM_K10E_SIGMA_ND", 0.0035) if sigma_nd is None else sigma_nd, 1e-6)
    sdf_cap = _k10e_env_float("BIOCHEM_K10E_SDF_MAX_ND", 0.02) if sdf_max_nd is None else sdf_max_nd
    d = sdf_nd.reshape(-1, 1).to(dtype=torch.float32).clamp(min=0.0)
    wm = wall_mask.reshape(-1, 1).to(dtype=d.dtype)
    off_wall = (1.0 - wm).clamp(0.0, 1.0)
    band = torch.exp(-0.5 * ((d - d_peak) / sigma) ** 2)
    if sdf_cap > 0.0:
        band = band * (d <= sdf_cap).to(dtype=d.dtype)
    return (band * off_wall).clamp(0.0, 1.0)



def _biochem_mu_disable_mu1() -> bool:
    return _biochem_env_truthy("BIOCHEM_MU_DISABLE_MU1")


def _biochem_mu_disable_mu2() -> bool:
    return _biochem_env_truthy("BIOCHEM_MU_DISABLE_MU2")


def biochem_explicit_gelation_terms(
    model: "GNODE_Phase3",
    fi_si: torch.Tensor,
    mat_si: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """μ₁/μ₂ terms that enter forward ``μ_kin * (1 + μ₁ + μ₂)`` (zeros when ablated).

    Matches ``GNODE_Phase3.forward`` explicit-gelation branch (caps, disable flags).
    Use for viz health metrics and plots — not raw species sigmoids when gelation is off.
    """
    if _biochem_mu_disable_explicit_gelation():
        zeros = torch.zeros_like(fi_si)
        return zeros, zeros
    mu1_term = (
        torch.zeros_like(mat_si)
        if _biochem_mu_disable_mu1()
        else model.mu1_sigmoid(mat_si)
    )
    mu2_term = (
        torch.zeros_like(fi_si)
        if _biochem_mu_disable_mu2()
        else model.mu2_sigmoid(fi_si)
    )
    mu2_cap_raw = (os.environ.get("BIOCHEM_MU2_SIGMOID_CAP") or "").strip()
    if mu2_cap_raw:
        mu2_cap = max(float(mu2_cap_raw), 0.0)
        mu2_term = torch.clamp(mu2_term, max=mu2_cap)
    return mu1_term, mu2_term


def _biochem_delta_mu_symmetric_bulk_clip() -> bool:
    if _biochem_mu_gemini_fix_enabled() or _biochem_mu_simple_log_residual_enabled():
        return True
    return _biochem_env_truthy("BIOCHEM_DELTA_MU_SYMMETRIC_BULK_CLIP")


def _biochem_env_truthy(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return (raw or "").strip().lower() in ("1", "true", "yes", "on")


# Canonical μ/rollout policy persisted in ``model_config["forward_policy"]`` (schema 1).
_BIOCHEM_FORWARD_POLICY_BOOL_FIELDS: tuple[tuple[str, str], ...] = (
    ("use_delta_mu_head", "BIOCHEM_USE_DELTA_MU_HEAD"),
    ("use_split_mu_head", "BIOCHEM_USE_SPLIT_MU_HEAD"),
    ("use_wall_delta_head", "BIOCHEM_USE_WALL_DELTA_HEAD"),
    ("mu_disable_explicit_gelation", "BIOCHEM_MU_DISABLE_EXPLICIT_GELATION"),
    ("mu_simple_log_residual", "BIOCHEM_MU_SIMPLE_LOG_RESIDUAL"),
    ("mu_ic_steady_kin", "BIOCHEM_MU_IC_STEADY_KIN"),
    ("mu_additive_delta", "BIOCHEM_MU_ADDITIVE_DELTA"),
    ("mu_gemini_fix", "BIOCHEM_MU_GEMINI_FIX"),
    ("gelation_prior_gate", "BIOCHEM_GELATION_PRIOR_GATE"),
    ("mu_disable_mu1", "BIOCHEM_MU_DISABLE_MU1"),
    ("mu_disable_mu2", "BIOCHEM_MU_DISABLE_MU2"),
    ("delta_mu_symmetric_bulk_clip", "BIOCHEM_DELTA_MU_SYMMETRIC_BULK_CLIP"),
    ("use_clot_nucleation_growth", "BIOCHEM_USE_CLOT_NUCLEATION_GROWTH"),
    ("use_bio_gate_suppressor", "BIOCHEM_USE_BIO_GATE_SUPPRESSOR"),
    ("mu_k10d_simple", "BIOCHEM_MU_K10D_SIMPLE"),
    ("mu_k10e_simple", "BIOCHEM_MU_K10E_SIMPLE"),
    ("mu_k10g_oracle_clots", "BIOCHEM_K10G_ORACLE_CLOTS"),
)

_BIOCHEM_FORWARD_POLICY_CLIP_FIELDS: tuple[tuple[str, str], ...] = (
    ("delta_mu_log_clip", "BIOCHEM_DELTA_MU_LOG_CLIP"),
    ("delta_mu_log_clip_bulk", "BIOCHEM_DELTA_MU_LOG_CLIP_BULK"),
    ("delta_mu_log_clip_wall", "BIOCHEM_DELTA_MU_LOG_CLIP_WALL"),
)


def snapshot_biochem_forward_policy() -> dict[str, Any]:
    """Effective μ/rollout branch flags at save time (not only raw env strings)."""
    policy: dict[str, Any] = {
        "schema": 1,
        "use_delta_mu_head": _biochem_delta_mu_head_enabled(),
        "use_split_mu_head": _biochem_split_mu_regime_head_enabled(),
        "use_wall_delta_head": _biochem_wall_delta_head_enabled(),
        "mu_disable_explicit_gelation": _biochem_mu_disable_explicit_gelation(),
        "mu_simple_log_residual": _biochem_mu_simple_log_residual_enabled(),
        "mu_ic_steady_kin": _biochem_mu_ic_steady_kin_enabled(),
        "mu_additive_delta": _biochem_mu_additive_delta_enabled(),
        "mu_gemini_fix": _biochem_mu_gemini_fix_enabled(),
        "gelation_prior_gate": _biochem_gelation_prior_gate_enabled(),
        "mu_disable_mu1": _biochem_mu_disable_mu1(),
        "mu_disable_mu2": _biochem_mu_disable_mu2(),
        "delta_mu_symmetric_bulk_clip": _biochem_delta_mu_symmetric_bulk_clip(),
        "use_clot_nucleation_growth": _biochem_clot_nucleation_growth_enabled(),
        "use_bio_gate_suppressor": _biochem_env_truthy("BIOCHEM_USE_BIO_GATE_SUPPRESSOR"),
        "mu_k10d_simple": _biochem_mu_k10d_simple_enabled(),
        "mu_k10e_simple": _biochem_mu_k10e_simple_enabled(),
        "mu_k10g_oracle_clots": _biochem_k10g_oracle_clots_enabled(),
    }
    if policy["mu_k10d_simple"]:
        raw_max = (os.environ.get("BIOCHEM_K10D_MU_DELTA_SI_MAX") or "").strip()
        if raw_max:
            policy["k10d_mu_delta_si_max"] = raw_max
    if policy["mu_k10e_simple"]:
        for field, env_key in (
            ("k10e_d_peak_nd", "BIOCHEM_K10E_D_PEAK_ND"),
            ("k10e_sigma_nd", "BIOCHEM_K10E_SIGMA_ND"),
            ("k10e_sdf_max_nd", "BIOCHEM_K10E_SDF_MAX_ND"),
            ("k10e_mu_delta_nd_max", "BIOCHEM_K10E_MU_DELTA_ND_MAX"),
        ):
            raw = (os.environ.get(env_key) or "").strip()
            if raw:
                policy[field] = raw
    for field, env_key in _BIOCHEM_FORWARD_POLICY_CLIP_FIELDS:
        raw = (os.environ.get(env_key) or "").strip()
        if raw:
            policy[field] = raw
    return policy


def biochem_forward_policy_from_checkpoint_meta(
    meta: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Read ``forward_policy`` from nested ``model_config`` or top-level checkpoint metadata."""
    if not meta:
        return None
    for container in (
        (meta or {}).get("model_config"),
        meta,
    ):
        if not isinstance(container, Mapping):
            continue
        fp = container.get("forward_policy")
        if isinstance(fp, Mapping) and int(fp.get("schema", 0)) == 1:
            return dict(fp)
    return None


def format_biochem_forward_policy_summary(policy: Mapping[str, Any] | None) -> str:
    if not policy or int(policy.get("schema", 0)) != 1:
        return ""
    tags: list[str] = []
    if policy.get("mu_ic_steady_kin"):
        tags.append("IC_steady_kin")
    if policy.get("mu_disable_explicit_gelation"):
        tags.append("no_explicit_gelation")
    if policy.get("use_delta_mu_head"):
        tags.append("delta_mu")
    if policy.get("use_split_mu_head"):
        tags.append("split_mu")
    if policy.get("use_wall_delta_head"):
        tags.append("wall_delta")
    if policy.get("mu_additive_delta"):
        tags.append("additive_delta")
    if policy.get("mu_k10d_simple"):
        tags.append("k10d_mu_ss_plus_learned")
    if policy.get("mu_k10e_simple"):
        tags.append("k10e_wall_adjacent_mu")
    if policy.get("mu_k10g_oracle_clots"):
        tags.append("k10g_oracle_clots")
    if policy.get("mu_gemini_fix"):
        tags.append("gemini_fix")
    if policy.get("gelation_prior_gate") is False:
        tags.append("gel_prior_off")
    return ", ".join(tags) if tags else "default branches"


def apply_biochem_forward_policy(
    policy: Mapping[str, Any] | None,
    *,
    quiet: bool = False,
) -> list[str]:
    """Set ``os.environ`` from checkpoint policy so ``forward()`` matches training."""
    if not policy or int(policy.get("schema", 0)) != 1:
        return []
    applied: list[str] = []
    for field, env_key in _BIOCHEM_FORWARD_POLICY_BOOL_FIELDS:
        if field not in policy:
            continue
        os.environ[env_key] = "1" if bool(policy[field]) else "0"
        applied.append(env_key)
    for field, env_key in _BIOCHEM_FORWARD_POLICY_CLIP_FIELDS:
        if field not in policy:
            continue
        raw = policy.get(field)
        if raw is None or str(raw).strip() == "":
            continue
        os.environ[env_key] = str(raw).strip()
        applied.append(env_key)
    for field, env_key in (
        ("k10e_d_peak_nd", "BIOCHEM_K10E_D_PEAK_ND"),
        ("k10e_sigma_nd", "BIOCHEM_K10E_SIGMA_ND"),
        ("k10e_sdf_max_nd", "BIOCHEM_K10E_SDF_MAX_ND"),
        ("k10e_mu_delta_nd_max", "BIOCHEM_K10E_MU_DELTA_ND_MAX"),
    ):
        if field not in policy:
            continue
        raw = policy.get(field)
        if raw is None or str(raw).strip() == "":
            continue
        os.environ[env_key] = str(raw).strip()
        applied.append(env_key)
    if not quiet:
        summary = format_biochem_forward_policy_summary(policy)
        if summary:
            print(f"   ↳ forward_policy restored ({summary})", flush=True)
    return applied


def apply_biochem_forward_policy_from_checkpoint_meta(
    meta: Mapping[str, Any] | None,
    *,
    quiet: bool = False,
) -> list[str]:
    policy = biochem_forward_policy_from_checkpoint_meta(meta)
    if policy is None:
        return []
    return apply_biochem_forward_policy(policy, quiet=quiet)


# Shell knobs for passive mu-unlock that must survive checkpoint forward_policy restore.
PASSIVE_MU_UNLOCK_SHELL_ENV_KEYS: tuple[str, ...] = (
    "BIOCHEM_PASSIVE_MU_UNLOCK",
    "BIOCHEM_LOSS_ISOLATE",
    "BIOCHEM_LOSS_DATA_ONLY",
    "BIOCHEM_COMPLEXITY_STEP",
    "BIOCHEM_TEACHER_MU_RATIO_MAX",
    "BIOCHEM_TRAIN_MU_ENCODER",
    "BIOCHEM_TRAIN_BIO_ENCODER",
    "BIOCHEM_TRAIN_BIO_DECODER",
    "BIOCHEM_TRAIN_ODE",
    "BIOCHEM_TRAIN_KIN_LORA",
    "BIOCHEM_MU_LOG_ANCHOR_WEIGHT",
    "BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT",
    "BIOCHEM_MU_LOG_WALL_WEIGHT",
    "BIOCHEM_MU_LOG_HIGH_WEIGHT",
    "BIOCHEM_TEACHER_TARGET_MU_LOG_MAE",
    "BIOCHEM_PASSIVE_ADR_BACKPROP",
    "BIOCHEM_GT_KINE_VEL",
    "BIOCHEM_GT_KINE_SKIP_DEQ",
    "BIOCHEM_PASSIVE_SPECIES_VAL",
    "BIOCHEM_PASSIVE_SPECIES_TRAIN_EVAL",
    "BIOCHEM_PASSIVE_MU_UNLOCK_FREEZE_BIO",
    "BIOCHEM_PASSIVE_MU_UNLOCK_FINETUNE",
    "BIOCHEM_USE_DELTA_MU_HEAD",
    "BIOCHEM_USE_SPLIT_MU_HEAD",
    "BIOCHEM_USE_WALL_DELTA_HEAD",
)


def snapshot_passive_mu_unlock_shell_env() -> dict[str, str]:
    """Capture shell env before checkpoint ``forward_policy`` overwrites training knobs."""
    out: dict[str, str] = {}
    for key in PASSIVE_MU_UNLOCK_SHELL_ENV_KEYS:
        raw = os.environ.get(key)
        if raw is not None and str(raw).strip() != "":
            out[key] = str(raw).strip()
    return out


def restore_passive_mu_unlock_shell_env(snapshot: Mapping[str, str]) -> list[str]:
    """Re-apply shell snapshot after ``apply_biochem_forward_policy``."""
    restored: list[str] = []
    for key, val in snapshot.items():
        os.environ[key] = str(val)
        restored.append(key)
    return restored


def _mu_trigger_gate_hard_threshold() -> float:
    """Cutoff center for bulk/tail and wall μ gates; 0 disables (continuous gate)."""
    raw = (os.environ.get("BIOCHEM_MU_TRIGGER_GATE_HARD_THRESH") or "").strip()
    if not raw:
        return 0.0
    return max(float(raw), 0.0)


def _mu_trigger_gate_hard_steepness() -> float:
    """Sigmoid steepness for soft hard-gate (higher = sharper cutoff, still differentiable)."""
    raw = (os.environ.get("BIOCHEM_MU_TRIGGER_GATE_HARD_STEEPNESS") or "10.0").strip() or "10.0"
    return max(float(raw), 1.0)


def _mu_soft_gate_scope() -> str:
    """Where to apply the soft cutoff when ``BIOCHEM_MU_TRIGGER_GATE_HARD_THRESH`` is set.

    ``wall_only`` (default): bulk/tail clot ``gate`` uses bio suppressor + optional floor only;
    wall ``wall_gate`` / ``wall_signal`` get the differentiable cutoff (stops lumen bleed).
    ``all``: legacy behavior — soft cutoff on clot gate too (can saturate bulk when gate > thresh).
    """
    raw = (os.environ.get("BIOCHEM_MU_SOFT_GATE_SCOPE") or "wall_only").strip().lower()
    if raw in ("all", "bulk", "bulk_tail", "clot"):
        return "all"
    return "wall_only"


def _apply_mu_gate_soft_threshold(
    g: Tensor,
    *,
    steepness: float | Tensor | None = None,
) -> Tensor:
    """Differentiable sharp cutoff: ``g * sigmoid(steepness * (g - thresh))``.

    Replaces a hard ``torch.where`` (which blocks gradients below the threshold) on paths
    where lumen must stay quiet. Above ``thresh``, output ≈ ``g`` — do not use on bulk clot
    ``gate`` unless species/bio suppressor already keep ``g`` small in the lumen.
    """
    thresh = _mu_trigger_gate_hard_threshold()
    if thresh <= 0.0:
        return g
    if steepness is None:
        steepness = _mu_trigger_gate_hard_steepness()
    activation = torch.sigmoid(steepness * (g - thresh))
    return g * activation


def _biochem_mu_wall_mix_mode() -> str:
    """How the wall residual branch is mixed into log-μ.

    - ``gate`` (default): ``gain * wall_gate * delta_wall`` (legacy).
    - ``relu_add`` / ``additive``: ``gain * ReLU(delta_wall)`` on wall-mask nodes only (Fix D).
    """
    v = (os.environ.get("BIOCHEM_MU_WALL_MIX_MODE", "gate") or "gate").strip().lower()
    if v in ("relu_add", "additive", "add", "residual", "siren_add"):
        return "relu_add"
    return "gate"


def _biochem_mu_wall_head_activation() -> str:
    return (os.environ.get("BIOCHEM_MU_WALL_HEAD_ACTIVATION", "silu") or "silu").strip().lower()


def _make_mu_delta_wall_head(in_dim: int) -> nn.Sequential:
    act = _biochem_mu_wall_head_activation()
    if act == "relu":
        hidden_act: nn.Module = nn.ReLU()
    elif act == "siren":
        from src.architecture.siren_decoder import Sine

        hidden_act = Sine(w0=30.0)
    else:
        hidden_act = nn.SiLU()
    return nn.Sequential(
        nn.Linear(in_dim, 64),
        hidden_act,
        nn.Linear(64, 1),
    )


def _wall_gate_from_signal(
    wall_signal_val: torch.Tensor,
    *,
    center: float,
    temp: float,
    t_scale: float,
    logit_bias: float = 0.0,
    gate_min: float = 0.0,
) -> torch.Tensor:
    wall_logits = (wall_signal_val - center) / max(temp * max(t_scale, 0.25), 1e-5)
    if logit_bias != 0.0:
        wall_logits = wall_logits + logit_bias
    wall_gate = torch.sigmoid(torch.clamp(wall_logits, min=-50.0, max=50.0))
    if gate_min > 0.0:
        wall_gate = torch.clamp(wall_gate, min=gate_min, max=1.0)
    return wall_gate


def _apply_wall_gate_curriculum(
    wall_gate: torch.Tensor,
    wall_mask: torch.Tensor,
    *,
    teacher_epoch: int | None,
    curriculum_epochs: int,
) -> torch.Tensor:
    """Fix A: force wall gate open on wall-mask nodes for the first N teacher epochs."""
    if curriculum_epochs <= 0 or teacher_epoch is None or teacher_epoch >= curriculum_epochs:
        return wall_gate
    on_wall = wall_mask.view(-1, 1).to(dtype=wall_gate.dtype) > 0.5
    return torch.where(on_wall, torch.ones_like(wall_gate), wall_gate)


def _mu_wall_branch_delta(delta_wall: torch.Tensor) -> torch.Tensor:
    """Activation applied to raw wall-head output before mixing into log-μ."""
    if _biochem_mu_wall_mix_mode() == "relu_add":
        return F.relu(delta_wall)
    act = _biochem_mu_wall_head_activation()
    if act == "relu":
        return F.relu(delta_wall)
    return delta_wall


def _biochem_wall_delta_head_enabled() -> bool:
    """Enable an extra near-wall residual log-μ correction branch."""
    raw = (os.environ.get("BIOCHEM_USE_WALL_DELTA_HEAD", "0") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _biochem_clot_nucleation_growth_enabled() -> bool:
    """Enable sparse clot nucleation + temporal growth trigger dynamics."""
    raw = (os.environ.get("BIOCHEM_USE_CLOT_NUCLEATION_GROWTH", "0") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def biochem_truth_node_mask(batch, num_nodes: int, device: torch.device) -> torch.Tensor:
    """Nodes whose entries in ``y`` are trusted COMSOL labels (per-node mask or graph-level ``is_anchor``)."""
    if not hasattr(batch, "is_anchor"):
        return torch.zeros(num_nodes, dtype=torch.bool, device=device)
    m = batch.is_anchor
    if not torch.is_tensor(m):
        m = torch.tensor(m, dtype=torch.bool)
    m = m.reshape(-1)
    if m.numel() == 1:
        if bool(m.item()):
            return torch.ones(num_nodes, dtype=torch.bool, device=device)
        return torch.zeros(num_nodes, dtype=torch.bool, device=device)
    batch_idx = getattr(batch, "batch", None)
    if batch_idx is not None:
        return m[batch_idx].to(device)
    if m.shape[0] != num_nodes:
        return torch.zeros(num_nodes, dtype=torch.bool, device=device)
    return m.to(device)


def _default_resting_species(num_nodes: int, device: torch.device, batch) -> torch.Tensor:
    current_species = torch.zeros(num_nodes, 12, dtype=torch.float32, device=device)
    # COMSOL-consistent resting blood chemistry in ND/log1p space.
    current_species[:, 0] = math.log1p(1.0)   # RP
    current_species[:, 1] = math.log1p(0.05)  # AP
    current_species[:, 4] = math.log1p(1.0)   # PT
    current_species[:, 6] = math.log1p(1.0)   # AT
    current_species[:, 7] = math.log1p(1.0)   # FG
    if hasattr(batch, "bio_inlet_bc"):
        mask_inlet = batch.mask_inlet.view(-1).bool()
        current_species[mask_inlet, 0:9] = batch.bio_inlet_bc[mask_inlet]
    return current_species


class BioODEFunc(nn.Module):
    """
    Calculates the temporal derivative dz/dt of the biochemical latent state.
    """
    def __init__(self, latent_dim, gnode_layers: int = 1):
        super().__init__()
        self.latent_dim = latent_dim
        self.gnode_layers = max(1, int(gnode_layers))
        # Processor to compute spatial interactions for the derivative
        # Plain Linear (no spectral norm): ODE inner loop runs many times per dopri5 step;
        # spectral_norm power iterations + odeint backprop peak GPU memory.
        self.derivative_processor = GINOBlock(latent_dim, use_spectral_norm=False)
        self.derivative_processor_extra = nn.ModuleList(
            [GINOBlock(latent_dim, use_spectral_norm=False) for _ in range(self.gnode_layers - 1)]
        )
        # Start from a near-steady system (dz/dt ~ 0) and let training grow dynamics.
        self.derivative_scale = nn.Parameter(torch.tensor(1e-5, dtype=torch.float32))
        # Memory-safe accumulator for latent derivative magnitude.
        # We only store detached scalar stats (not full tensors per ODE eval).
        self.derivative_energy_sum = 0.0
        self.derivative_eval_count = 0

    def _derivative_gino_block(
        self,
        z: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch_idx: Tensor,
        mb: Tensor,
    ) -> Tensor:
        n_e = int(edge_index.shape[1])
        mod_dummy = torch.zeros(n_e, 1, dtype=torch.float32, device=z.device)
        out = self.derivative_processor(z, edge_index, edge_attr, batch_idx, mb, mod_dummy, mod_dummy)
        for layer in self.derivative_processor_extra:
            out = layer(out, edge_index, edge_attr, batch_idx, mb, mod_dummy, mod_dummy)
        return out

    def forward(self, t, z, edge_index, edge_attr, batch_idx, mod_biochem=None):
        # The derivative dz/dt depends strictly on the current biochemical state 'z' and frozen physics.
        z = torch.clamp(z, min=-20.0, max=20.0)
        n_e = int(edge_index.shape[1])
        mod_dummy = torch.zeros(n_e, 1, dtype=torch.float32, device=z.device)
        mod_in = mod_biochem if mod_biochem is not None else mod_dummy
        if self.training and _biochem_ode_grad_checkpoint_enabled():
            dz_raw = checkpoint(
                self._derivative_gino_block,
                z,
                edge_index,
                edge_attr,
                batch_idx,
                mod_in,
                use_reentrant=False,
            )
        else:
            dz_raw = self._derivative_gino_block(z, edge_index, edge_attr, batch_idx, mod_in)

        # Explicitly smooth latent states to suppress high-frequency spatial jitter during integration.
        row, col = edge_index
        deg = degree(row, z.size(0), dtype=z.dtype).clamp_(min=1.0)
        laplacian_smooth = torch.zeros_like(z)
        z_j = z[col]
        laplacian_smooth.scatter_add_(
            0,
            row.unsqueeze(-1).expand(-1, z.size(-1)),
            z_j / deg[col].unsqueeze(-1),
        )
        laplacian_smooth = laplacian_smooth - z
        dz_raw = dz_raw + 0.05 * laplacian_smooth

        dz_dt = self.derivative_scale * dz_raw
        dz_dt = torch.clamp(dz_dt, min=-10.0, max=10.0)
        if self.training:
            self.derivative_energy_sum += float(dz_dt.detach().pow(2).mean().item())
            self.derivative_eval_count += 1

        return dz_dt

def infer_use_siren_decoder_from_state_dict(state_dict: Mapping[str, Any]) -> bool | None:
    """Infer decoder type from checkpoint keys (legacy checkpoints without ``model_config``)."""
    has_siren = any(str(k).startswith("siren_decoder.") for k in state_dict)
    has_linear = any(str(k).startswith("kinematics_decoder.") for k in state_dict)
    if has_siren:
        return True
    if has_linear:
        return False
    return None


def infer_fourier_bands_from_state_dict(state_dict: Mapping[str, Any]) -> int | None:
    """``kin_encoder`` input width = 15 + 10 * num_fourier_freqs (+3 width priors when enabled)."""
    w = state_dict.get("kin_encoder.0.weight")
    if w is None or not hasattr(w, "shape") or len(w.shape) != 2:
        return None
    width_extra = 3 if bool(int(os.environ.get("KINEMATICS_USE_WIDTH_PRIORS", "1"))) else 0
    bands = (int(w.shape[1]) - 15 - width_extra) // 10
    return bands if bands >= 1 else None


def infer_latent_dim_from_state_dict(state_dict: Mapping[str, Any]) -> int | None:
    w = state_dict.get("kin_encoder.0.weight")
    if w is not None and hasattr(w, "shape") and len(w.shape) == 2:
        return int(w.shape[0])
    w = state_dict.get("bio_encoder.linear.parametrizations.weight.original")
    if w is not None and hasattr(w, "shape") and len(w.shape) == 2:
        return int(w.shape[0])
    return None


def snapshot_biochem_model_config(model: "GNODE_Phase3") -> dict[str, Any]:
    """Persisted in ``.pth`` so viz/resume do not rely on training env flags."""
    return {
        "schema": 1,
        "latent_dim": int(model.latent_dim),
        "max_inner_iters": int(model.max_inner_iters),
        "bio_encoder_prior_dim": int(model.bio_encoder_prior_dim),
        "num_fourier_freqs": int(model.num_fourier_freqs),
        "use_siren_decoder": bool(model.use_siren_decoder),
        "gnode_layers": int(model.gnode_layers),
        "use_hard_bcs": bool(model.use_hard_bcs),
        "in_channels": 12,
        "spatial_channels": 15,
        "forward_policy": snapshot_biochem_forward_policy(),
    }


def resolve_gnode_phase3_ctor_kwargs(
    meta: Mapping[str, Any] | None,
    state_dict: Mapping[str, Any],
    *,
    bio_encoder_prior_dim_default: int = 2,
    latent_dim_default: int = 256,
    fourier_bands_default: int = 8,
    use_siren_default: bool = False,
    gnode_layers_default: int = 1,
    max_inner_iters_default: int = 10,
) -> dict[str, Any]:
    """Build ``GNODE_Phase3`` kwargs from checkpoint ``model_config``, else tensor shapes."""
    saved = dict((meta or {}).get("model_config") or {})
    if int(saved.get("schema", 0)) == 1:
        return {
            "latent_dim": max(8, int(saved.get("latent_dim", latent_dim_default))),
            "max_inner_iters": max(3, int(saved.get("max_inner_iters", max_inner_iters_default))),
            "bio_encoder_prior_dim": max(0, int(saved.get("bio_encoder_prior_dim", bio_encoder_prior_dim_default))),
            "num_fourier_freqs": max(1, int(saved.get("num_fourier_freqs", fourier_bands_default))),
            "use_siren_decoder": bool(saved.get("use_siren_decoder", use_siren_default)),
            "gnode_layers": max(1, int(saved.get("gnode_layers", gnode_layers_default))),
            "use_hard_bcs": bool(saved.get("use_hard_bcs", False)),
            "in_channels": int(saved.get("in_channels", 12)),
            "spatial_channels": int(saved.get("spatial_channels", 15)),
        }

    inferred_siren = infer_use_siren_decoder_from_state_dict(state_dict)
    inferred_fourier = infer_fourier_bands_from_state_dict(state_dict)
    inferred_latent = infer_latent_dim_from_state_dict(state_dict)
    return {
        "latent_dim": inferred_latent if inferred_latent is not None else latent_dim_default,
        "max_inner_iters": max_inner_iters_default,
        "bio_encoder_prior_dim": bio_encoder_prior_dim_default,
        "num_fourier_freqs": inferred_fourier if inferred_fourier is not None else fourier_bands_default,
        "use_siren_decoder": inferred_siren if inferred_siren is not None else use_siren_default,
        "gnode_layers": gnode_layers_default,
        "use_hard_bcs": bool(int(os.environ.get("KINEMATICS_USE_HARD_BCS", "0"))),
        "in_channels": 12,
        "spatial_channels": 15,
    }


class GNODE_Phase3(nn.Module):
    """
    Biochem Physics-Informed Graph Neural ODE for dynamic Thrombosis Simulation.
    Replaces the steady-state DEQ with a continuous-time latent ODE solver.
    """

    def __init__(
        self,
        phys_cfg,
        in_channels=12,
        spatial_channels=15,
        latent_dim=64,
        max_inner_iters=25,
        bio_encoder_prior_dim: int = 0,
        mu_ratio_max=80.0,
        mat_crit=2e7,
        fi_crit=0.6,
        temp_mat=1e6,
        temp_fi=0.05,
        rtol=1e-3,
        atol=1e-4,
        num_fourier_freqs: int = 8,
        use_siren_decoder: bool = False,
        gnode_layers: int = 1,
        use_hard_bcs: bool | None = None,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.max_inner_iters = max_inner_iters
        self.rtol = rtol
        self.atol = atol
        self.phys_cfg = phys_cfg  # Store physics config internally
        # Micro-step ODE restart behavior:
        # - "rk4" is restart-friendly for short segments.
        # - set BIOCHEM_ODE_METHOD=implicit_adams to preserve legacy behavior.
        self.micro_ode_method = str(os.environ.get("BIOCHEM_ODE_METHOD", "rk4")).strip().lower()
        # Optional TBPTT-style truncation at macro boundaries.
        self.detach_macro_state_default = bool(int(os.environ.get("BIOCHEM_DETACH_MACRO_STATE", "0")))
        if use_hard_bcs is None:
            self.use_hard_bcs = bool(int(os.environ.get("KINEMATICS_USE_HARD_BCS", "0")))
        else:
            self.use_hard_bcs = bool(use_hard_bcs)
        # Match kinematics_best.pth encoder width (178 = 175 + 3 width priors when bands=16).
        self.use_width_priors = bool(int(os.environ.get("KINEMATICS_USE_WIDTH_PRIORS", "1")))

        # COMSOL step ceilings for mu1_sigmoid / mu2_sigmoid (not clot mu_eff/bulk ratio).
        self.mu_ratio_max = mu_ratio_max
        self.mat_crit = mat_crit
        self.fi_crit = fi_crit

        # Temperature scaling for soft sigmoids
        self.temp_mat = temp_mat
        self.temp_fi = temp_fi
        self.T_scale = 1.0

        # ==========================================
        # 1. KINEMATICS BACKBONE (FROZEN)
        # ==========================================
        self.edge_decay_k = float(self.phys_cfg.gino_edge_decay_k)
        self.curve_log_clamp_min = float(self.phys_cfg.gino_curve_log_clamp_min)
        self.rheo_log_clamp_min = float(self.phys_cfg.gino_rheo_log_clamp_min)
        self.adv_log_clamp_min = float(self.phys_cfg.gino_adv_log_clamp_min)

        self.num_fourier_freqs = max(1, int(num_fourier_freqs))
        freqs = (2.0 ** torch.arange(self.num_fourier_freqs)) * torch.pi
        if bool(int(os.environ.get("KINEMATICS_FOURIER_LEARNABLE", "0"))):
            self.fourier_freqs = nn.Parameter(freqs)
        else:
            self.register_buffer("fourier_freqs", freqs)
        self.use_siren_decoder = bool(use_siren_decoder)
        self.gnode_layers = max(1, int(gnode_layers))

        fourier_channels = 5 * self.num_fourier_freqs * 2
        width_extra = 3 if self.use_width_priors else 0
        encoded_channels = (15 - 5) + 5 + fourier_channels + width_extra

        self.kin_encoder = nn.Sequential(
            nn.Linear(encoded_channels, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.kin_processor = GINOBlock(latent_dim, edge_dim=3, num_global_tokens=16)
        self.kin_processor_extra = nn.ModuleList(
            [GINOBlock(latent_dim, edge_dim=3, num_global_tokens=16) for _ in range(self.gnode_layers - 1)]
        )
        if self.use_siren_decoder:
            self.siren_decoder = SIRENDecoder(latent_dim=latent_dim)
            self.kinematics_decoder = None
        else:
            self.siren_decoder = None
            self.kinematics_decoder = nn.Linear(latent_dim, 3)
        self.mu_encoder = nn.Linear(1, latent_dim)
        self.z_prior_proj = SpectralLinear(4, latent_dim)
        mu_scale = float(self.phys_cfg.mu_viscosity_nd_scale)
        self.mu_inf_nd = float(self.phys_cfg.mu_inf / mu_scale)
        self.mu_0_nd = float(self.phys_cfg.mu_0 / mu_scale)
        # Frozen stage-A μ decoder (t=0 DEQ parity with kinematics_best.pth).
        self.mu_decoder = nn.Sequential(
            SpectralLinear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, 1),
        )

        # ==========================================
        # 2. BIOCHEMISTRY NEURAL ODE
        # ==========================================
        # Initial Condition Encoder (Maps spatial config to z0)
        self.bio_encoder_prior_dim = max(0, int(bio_encoder_prior_dim))
        self._bio_cfg = BiochemConfig(phase="biochem")
        self.bio_encoder = SpectralLinear(
            in_features=in_channels + 3 + spatial_channels + self.bio_encoder_prior_dim,
            out_features=latent_dim,
        )
        _bio = self._bio_cfg
        self.sgt = _bio.sgt
        self.T_grad = _bio.soft_step_T_grad
        # 5x attention multiplier for edges in decelerating zones (log-space bias)
        self.biochem_attention_boost = math.log(5.0)

        # The ODE Function
        self.ode_func = BioODEFunc(latent_dim, gnode_layers=self.gnode_layers)

        # Physical Decoder
        self.biochem_decoder = SpectralLinear(in_features=latent_dim, out_features=12)

        # Kinematic baseline + biochem-only corrector: μ_eff = μ_kin * (1 + explicit_gel + learned_gel).
        # ``learned_clot_penalty`` sees only log1p species (12); Softplus ⇒ nonnegative learned gelation.
        self.learned_clot_penalty = nn.Sequential(
            nn.Linear(12, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )
        self._init_learned_clot_penalty_near_zero()
        # Optional residual log-space correction for μ; kept near-zero by default.
        self.mu_delta_head = nn.Sequential(
            nn.Linear(latent_dim + 12, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        # Optional split residual correction:
        # log(μ) = log(μ_base) + (1-g)*Δ_bulk + g*Δ_tail
        # where g∈[0,1] is a trigger gate from species + mechanics cues.
        self.mu_delta_bulk_head = nn.Sequential(
            nn.Linear(latent_dim + 12, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.mu_trigger_feat_dim = 19
        self.mu_delta_tail_head = nn.Sequential(
            nn.Linear(latent_dim + self.mu_trigger_feat_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.mu_trigger_gate_head = nn.Sequential(
            nn.Linear(self.mu_trigger_feat_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        # Sparse trigger decomposition:
        # - nucleation head detects local initiation cues
        # - growth head continues expansion from existing activated regions
        self.mu_nucleation_gate_head = nn.Sequential(
            nn.Linear(self.mu_trigger_feat_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        self.mu_growth_gate_head = nn.Sequential(
            nn.Linear(self.mu_trigger_feat_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        # Optional near-wall residual branch.
        # Uses kinematics + species + wall cues to correct underfit wall viscosity.
        self.mu_delta_wall_head = _make_mu_delta_wall_head(latent_dim + self.mu_trigger_feat_dim)
        self.mu_trigger_gate_temp = max(float(os.environ.get("BIOCHEM_MU_TRIGGER_GATE_TEMP", "0.20")), 1e-5)
        # Learnable steepness for wall-only soft cutoff (``BIOCHEM_MU_GATE_LEARNED_TEMP=1``).
        self.mu_soft_gate_log_temp = nn.Parameter(torch.tensor(10.0))
        self.mu_delta_log_clip = max(float(os.environ.get("BIOCHEM_DELTA_MU_LOG_CLIP", "1.5")), 1e-6)
        self.mu_wall_gate_temp = max(float(os.environ.get("BIOCHEM_MU_WALL_GATE_TEMP", "0.18")), 1e-5)
        self.mu_wall_gate_center = float(os.environ.get("BIOCHEM_MU_WALL_GATE_CENTER", "0.55"))
        gate_pos_init = os.environ.get("BIOCHEM_MU_WALL_GATE_POS_INIT", "").strip()
        if gate_pos_init:
            self._mu_wall_gate_logit_bias = float(gate_pos_init)
        else:
            self._mu_wall_gate_logit_bias = float(os.environ.get("BIOCHEM_WALL_GATE_BIAS", "0.0"))
        self._mu_wall_gate_curriculum_epochs = max(
            0, int(os.environ.get("BIOCHEM_WALL_GATE_CURRICULUM_EPOCHS", "0"))
        )
        self._biochem_teacher_epoch: int | None = None
        self.mu_wall_delta_gain = float(os.environ.get("BIOCHEM_MU_WALL_DELTA_GAIN", "0.65"))
        self.mu_wall_mask_mix = min(
            max(float(os.environ.get("BIOCHEM_MU_WALL_MASK_MIX", "0.80")), 0.0),
            1.0,
        )
        self.mu_clot_growth_memory = min(
            max(float(os.environ.get("BIOCHEM_CLOT_GROWTH_MEMORY", "0.85")), 0.0),
            0.995,
        )
        self.mu_clot_wall_prior_mix = min(
            max(float(os.environ.get("BIOCHEM_CLOT_GATE_WALL_PRIOR_MIX", "0.65")), 0.0),
            1.0,
        )
        self.mu_clot_wall_prior_floor = min(
            max(float(os.environ.get("BIOCHEM_CLOT_GATE_WALL_PRIOR_FLOOR", "0.10")), 0.0),
            1.0,
        )
        self.mu_clot_nucleation_bias = float(os.environ.get("BIOCHEM_CLOT_GATE_NUCLEATION_BIAS", "0.0"))
        self.mu_clot_growth_bias = float(os.environ.get("BIOCHEM_CLOT_GATE_GROWTH_BIAS", "0.0"))
        self._init_mu_delta_head_near_zero()

        self.register_buffer("species_si_scales", _bio.get_species_scales(device="cpu"))

    def _mu_soft_gate_steepness(self) -> float | Tensor:
        if _biochem_env_truthy("BIOCHEM_MU_GATE_LEARNED_TEMP"):
            return F.softplus(self.mu_soft_gate_log_temp) + 1e-3
        return _mu_trigger_gate_hard_steepness()

    def _kinematics_prior_tail(self, batch, u_nd: torch.Tensor, v_nd: torch.Tensor) -> torch.Tensor | None:
        """Extra ``bio_encoder`` channels from kinematics clot prior (COMSOL-aligned cues)."""
        if self.bio_encoder_prior_dim <= 0:
            return None
        props = {
            "u_ref": batch.u_ref.view(-1, 1).to(dtype=torch.float32),
            "d_bar": batch.d_bar.view(-1, 1).to(dtype=torch.float32),
        }
        return clot_prior_features(
            batch,
            u_nd.reshape(-1),
            v_nd.reshape(-1),
            self._bio_cfg,
            props,
            n_features=self.bio_encoder_prior_dim,
        )

    def train(self, mode=True):
        """
        Override the default train method to ensure the frozen kinematic
        backbone strictly remains in evaluation mode.
        """
        super().train(mode)
        self.kin_encoder.eval()
        self.kin_processor.eval()
        for layer in self.kin_processor_extra:
            layer.eval()
        if self.kinematics_decoder is not None:
            self.kinematics_decoder.eval()
        if self.siren_decoder is not None:
            self.siren_decoder.eval()
        self.mu_encoder.eval()
        self.mu_decoder.eval()

    def _apply_kin_processor_stack(self, z_in: torch.Tensor, batch, mod_adv, mod_rheo, mod_curve) -> torch.Tensor:
        num_nodes = int(z_in.shape[0])
        device = z_in.device
        batch_idx = get_batch_tensor(batch, num_nodes, device)
        use_ckpt = self.training and _biochem_kin_grad_checkpoint_enabled()
        if use_ckpt:
            z_out = checkpoint(
                self.kin_processor,
                z_in,
                batch.edge_index,
                batch.edge_attr,
                batch_idx,
                mod_adv,
                mod_rheo,
                mod_curve,
                use_reentrant=False,
            )
        else:
            z_out = self.kin_processor(z_in, batch.edge_index, batch.edge_attr, batch_idx, mod_adv, mod_rheo, mod_curve)
        for layer in self.kin_processor_extra:
            if use_ckpt:
                z_out = checkpoint(
                    layer,
                    z_out,
                    batch.edge_index,
                    batch.edge_attr,
                    batch_idx,
                    mod_adv,
                    mod_rheo,
                    mod_curve,
                    use_reentrant=False,
                )
            else:
                z_out = layer(z_out, batch.edge_index, batch.edge_attr, batch_idx, mod_adv, mod_rheo, mod_curve)
        return z_out

    def _apply_fourier_encoding(self, x, pos_nd=None):
        # Canonical kinematics layout is 15 channels; width priors append three more (see NodeFeat).
        xb = x[:, :15] if x.size(1) >= 15 else x
        nodes_nd = pos_nd if pos_nd is not None else xb[:, NodeFeat.XY]
        sdf_nd = xb[:, NodeFeat.SDF]
        shear_pot = xb[:, NodeFeat.SHEAR_POT]
        wall_normal = xb[:, NodeFeat.WALL_NORMAL]
        rest = xb[:, NodeFeat.REST]
        uv_prior = xb[:, NodeFeat.UV_PRIOR]
        mu_prior = xb[:, NodeFeat.MU_PRIOR]
        wss_prior = xb[:, NodeFeat.WSS_PRIOR]

        features_to_encode = torch.cat([nodes_nd, sdf_nd, wall_normal], dim=1)
        N, C = features_to_encode.shape

        x_proj = (features_to_encode.unsqueeze(-1) * self.fourier_freqs).contiguous()
        x_proj = x_proj.view(N, -1)
        fourier_feats = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

        encoded_x = torch.cat([
            shear_pot, features_to_encode, fourier_feats,
            rest, uv_prior, mu_prior, wss_prior
        ], dim=1)
        if getattr(self, "use_width_priors", False):
            if x.size(1) >= NodeFeat.WIDTH_D2.stop:
                width_features = x[:, NodeFeat.WIDTH_ND.start : NodeFeat.WIDTH_D2.stop]
            else:
                width_features = torch.zeros(x.size(0), 3, device=x.device, dtype=x.dtype)
            encoded_x = torch.cat([encoded_x, width_features], dim=1)
        return encoded_x

    def _kinematics_input_cols(self, batch) -> int:
        """Kinematics node features: 15 base + optional 3 width-prior channels."""
        if getattr(self, "use_width_priors", False) and batch.x.size(1) >= NodeFeat.WIDTH_D2.stop:
            return NodeFeat.WIDTH_D2.stop
        return 15

    def _slice_kinematics_input(self, batch, *, clone: bool = False) -> torch.Tensor:
        feat = batch.x[:, : self._kinematics_input_cols(batch)]
        return feat.clone() if clone else feat

    def _kinematics_deq_warm_start(self, kin_in: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
        """Stage-A warm start: geometric encoder output + prior projection."""
        kin_encoded = self._apply_fourier_encoding(kin_in)
        x_enc = self.kin_encoder(kin_encoded)
        return x_enc + self.z_prior_proj(priors)

    def _compute_kinematics_modulators(self, batch):
        """Computes physical edge modulators (advection, rheology, curvature) for GINO."""
        row, col = batch.edge_index
        edge_vec = batch.edge_attr[:, :2]
        wall_normals = batch.x[:, NodeFeat.WALL_NORMAL]

        e_dir = F.normalize(edge_vec, p=2, dim=-1, eps=1e-8)
        n_dir_row = F.normalize(wall_normals[row], p=2, dim=-1, eps=1e-8)
        n_dir_col = F.normalize(wall_normals[col], p=2, dim=-1, eps=1e-8)

        dot_prod = torch.abs((e_dir * n_dir_row).sum(dim=-1, keepdim=True))
        dot_prod = torch.clamp(dot_prod, max=1.0)

        sdf_nd = batch.x[:, NodeFeat.SDF]
        sdf_edge = sdf_nd[row]
        decay_factor = torch.exp(-self.edge_decay_k * sdf_edge)

        curve_dot = (n_dir_row * n_dir_col).sum(dim=-1, keepdim=True)
        mod_curve = torch.log(torch.clamp(1.0 - curve_dot, min=self.curve_log_clamp_min, max=1.0)) * decay_factor
        mod_rheo = torch.log(torch.clamp(dot_prod, min=self.rheo_log_clamp_min, max=1.0)) * decay_factor
        mod_adv = torch.log(torch.clamp((1.0 - dot_prod), min=self.adv_log_clamp_min, max=1.0)) * decay_factor

        return mod_adv, mod_rheo, mod_curve

    def _decode_constrained_uvp(self, z_kin: torch.Tensor, kin_in: torch.Tensor, batch) -> torch.Tensor:
        """Decode latent kinematics to (u, v, p); optional SDF hard BCs when stage-A expects them."""
        if self.use_siren_decoder and self.siren_decoder is not None:
            pos_nd = kin_in[:, NodeFeat.XY]
            u_v_p_raw, _ = self.siren_decoder(z_kin, pos_nd)
            u_v_p = u_v_p_raw[:, PredChannels.KINEMATICS]
        else:
            u_v_p = self.kinematics_decoder(z_kin)
        if self.use_hard_bcs:
            sdf = kin_in[:, NodeFeat.SDF]
            uv_prior = kin_in[:, NodeFeat.UV_PRIOR]
            if bool(int(os.environ.get("KINEMATICS_BC_ENVELOPE", "0"))):
                bc_lambda = float(os.environ.get("KINEMATICS_BC_LAMBDA", "10.0"))
                envelope = 1.0 - torch.exp(-bc_lambda * sdf)
                u_v_constrained = uv_prior + envelope * u_v_p[:, :2]
            else:
                u_v_constrained = uv_prior + sdf * u_v_p[:, :2]
            return torch.cat([u_v_constrained, u_v_p[:, 2:3]], dim=1)
        return u_v_p

    def _initial_mu_eff_si(self, batch) -> torch.Tensor:
        """Bootstrap macro μ from spatial Carreau prior in graph features (not uniform μ_inf)."""
        mu_nd_scale = self.phys_cfg.mu_viscosity_nd_scale
        mu_prior_nd = batch.x[:, NodeFeat.MU_PRIOR]
        return (mu_prior_nd * mu_nd_scale).clone()

    def _steady_kinematics_mu_uv(
        self,
        batch,
        z_kin_ws: torch.Tensor,
        mod_adv: torch.Tensor,
        mod_rheo: torch.Tensor,
        mod_curve: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One-shot frozen-kin DEQ (``use_mu_decoder``) — same μ path as steady GINO-DEQ viz."""
        mu_nd_scale = self.phys_cfg.mu_viscosity_nd_scale
        mu_seed_si = self._initial_mu_eff_si(batch)
        kin_in = self._slice_kinematics_input(batch, clone=True)
        kin_in[:, NodeFeat.MU_PRIOR] = mu_seed_si / mu_nd_scale
        kin_encoded = self._apply_fourier_encoding(kin_in)
        z_kin, u_v_p, z_kin_ws_out = self._solve_kinematics_macro(
            kin_encoded,
            kin_in,
            batch,
            mod_adv,
            mod_rheo,
            mod_curve,
            z_kin_ws,
            use_mu_decoder=True,
            current_mu_eff=mu_seed_si,
        )
        mu_nd = self._decode_mu_nd_from_latent(z_kin.detach())
        mu_eff_si = (mu_nd * mu_nd_scale).clamp(min=1e-8)
        return mu_eff_si, u_v_p, z_kin_ws_out

    def _decode_mu_nd_from_latent(self, z_flat: torch.Tensor) -> torch.Tensor:
        """ND μ from frozen stage-A ``mu_decoder`` (matches GINO_DEQ ``decode_mu``)."""
        mu_raw = self.mu_decoder(z_flat)
        return self.mu_inf_nd + (self.mu_0_nd - self.mu_inf_nd) * torch.sigmoid(mu_raw)

    def _solve_kinematics_macro(
        self,
        kin_encoded: torch.Tensor,
        kin_in: torch.Tensor,
        batch,
        mod_adv: torch.Tensor,
        mod_rheo: torch.Tensor,
        mod_curve: torch.Tensor,
        z_kin_ws: torch.Tensor,
        *,
        use_mu_decoder: bool,
        current_mu_eff: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Anderson DEQ for frozen kinematics (stage-A parity); returns (z_kin, u_v_p, z_kin_ws_carry)."""
        mu_nd_scale = self.phys_cfg.mu_viscosity_nd_scale
        num_nodes = kin_encoded.size(0)
        batch_idx = get_batch_tensor(batch, num_nodes, kin_encoded.device)

        def f_kin(curr_z: torch.Tensor) -> torch.Tensor:
            # Anderson uses (batch, nodes, dim); GINOBlock expects (nodes, dim).
            curr_z_flat = curr_z.squeeze(0) if curr_z.ndim == 3 else curr_z
            if use_mu_decoder:
                mu_nd = self._decode_mu_nd_from_latent(curr_z_flat)
            else:
                mu_nd = current_mu_eff / mu_nd_scale
            mu_enc = self.mu_encoder(mu_nd)
            injection = self.kin_encoder(kin_encoded) + mu_enc
            out = self._apply_kin_processor_stack(
                injection + curr_z_flat, batch, mod_adv, mod_rheo, mod_curve
            )
            return out.unsqueeze(0) if curr_z.ndim == 3 else out

        z_init = z_kin_ws.unsqueeze(0) if z_kin_ws.ndim == 2 else z_kin_ws
        with torch.no_grad():
            z_kin_ws = anderson_acceleration(
                f_kin,
                z_init,
                batch_idx=batch_idx,
                max_iter=self.max_inner_iters,
                beta=0.8,
                warmup_iters=5,
            )

        if use_mu_decoder:
            mu_nd = self._decode_mu_nd_from_latent(z_kin_ws.detach())
        else:
            mu_nd = current_mu_eff / mu_nd_scale
        mu_enc = self.mu_encoder(mu_nd)
        injection = self.kin_encoder(kin_encoded) + mu_enc
        z_kin = self._apply_kin_processor_stack(
            injection + z_kin_ws.detach(), batch, mod_adv, mod_rheo, mod_curve
        )
        u_v_p = self._decode_constrained_uvp(z_kin, kin_in, batch)
        return z_kin, u_v_p, z_kin_ws

    def _decode_species_log1p(self, raw_species: torch.Tensor) -> torch.Tensor:
        """Decoder predicts log1p(species_nd) directly; clamp to a safe training range."""
        return torch.clamp(raw_species, min=_SPECIES_LOG1P_MIN, max=_SPECIES_LOG1P_MAX)

    def species_log_nd_to_si(self, species_log: Tensor) -> Tensor:
        """Map stored ``log1p(c / c_scale)`` bulk+wall channels (12) to SI concentrations."""
        sc = self.species_si_scales.to(device=species_log.device, dtype=species_log.dtype)
        sp = torch.clamp(species_log, min=_SPECIES_LOG1P_MIN, max=_SPECIES_LOG1P_MAX)
        return torch.expm1(sp) * sc

    def _init_learned_clot_penalty_near_zero(self) -> None:
        """Small random weights, zero biases, then shift last linear bias so Softplus output starts ~0."""
        with torch.no_grad():
            for m in self.learned_clot_penalty.modules():
                if isinstance(m, nn.Linear):
                    m.weight.uniform_(-1e-4, 1e-4)
                    if m.bias is not None:
                        m.bias.zero_()
            last_lin = self.learned_clot_penalty[2]
            if isinstance(last_lin, nn.Linear):
                last_lin.bias.fill_(-6.0)

    def _init_mu_delta_head_near_zero(self) -> None:
        """Keep residual μ correction neutral at startup (factor ~= 1)."""
        bias_nd_raw = (os.environ.get("BIOCHEM_K10E_DELTA_BIAS_ND") or "").strip()
        bias_nd = float(bias_nd_raw) if bias_nd_raw else 0.0
        with torch.no_grad():
            for head in (
                self.mu_delta_head,
                self.mu_delta_bulk_head,
                self.mu_delta_tail_head,
                self.mu_delta_wall_head,
            ):
                for m in head.modules():
                    if isinstance(m, nn.Linear):
                        m.weight.uniform_(-1e-4, 1e-4)
                        if m.bias is not None:
                            m.bias.zero_()
            if bias_nd > 0.0 and isinstance(self.mu_delta_head, nn.Sequential):
                last_lin = self.mu_delta_head[-1]
                if isinstance(last_lin, nn.Linear) and last_lin.bias is not None:
                    target = max(float(bias_nd), 1e-4)
                    last_lin.bias.fill_(float(torch.log(torch.expm1(torch.tensor(target))).item()))
            for gate_head in (self.mu_trigger_gate_head, self.mu_nucleation_gate_head, self.mu_growth_gate_head):
                for m in gate_head.modules():
                    if isinstance(m, nn.Linear):
                        m.weight.uniform_(-1e-4, 1e-4)
                        if m.bias is not None:
                            m.bias.zero_()

    def mu1_sigmoid(self, mat):
        """Platelet step μ₁(Mat) in [0, mu_ratio_max - 1] (COMSOL step ceiling, not mu_eff ratio)."""
        safe_t_scale = max(self.T_scale, 1e-5)
        norm_val = (mat - self.mat_crit) / (self.temp_mat * safe_t_scale)
        safe_val = torch.clamp(norm_val, min=-50.0, max=50.0)
        return (self.mu_ratio_max - 1.0) * torch.sigmoid(safe_val)

    def mu2_sigmoid(self, fi):
        """Fibrin step μ₂(FI) in [0, mu_ratio_max] (COMSOL step ceiling, not mu_eff ratio)."""
        safe_t_scale = max(self.T_scale, 1e-5)
        norm_val = (fi - self.fi_crit) / (self.temp_fi * safe_t_scale)
        safe_val = torch.clamp(norm_val, min=-50.0, max=50.0)
        return self.mu_ratio_max * torch.sigmoid(safe_val)

    def autoencode(self, batch):
        """
        Kinematics: Pure spatial representation learning (no ODE time integration).
        Predict biochemical state at t=0 from priors + kinematics that match the rollout path
        (resting species, DEQ velocities from frozen backbone — not ground-truth ``batch.y``).
        """
        num_nodes = int(batch.x.shape[0])
        device = batch.x.device
        current_species = _default_resting_species(num_nodes, device, batch)

        mu_nd_scale = self.phys_cfg.mu_viscosity_nd_scale
        current_mu_eff = self._initial_mu_eff_si(batch)

        kin_in = self._slice_kinematics_input(batch, clone=True)
        kin_in[:, NodeFeat.MU_PRIOR] = current_mu_eff / mu_nd_scale
        kin_encoded = self._apply_fourier_encoding(kin_in)
        uv_prior = kin_in[:, NodeFeat.UV_PRIOR]
        p_prior = kin_in[:, NodeFeat.SHEAR_POT]
        mu_prior = kin_in[:, NodeFeat.MU_PRIOR]
        priors = torch.cat([uv_prior, p_prior, mu_prior], dim=1)
        z_kin_ws = self._kinematics_deq_warm_start(kin_in, priors)

        mod_adv, mod_rheo, mod_curve = self._compute_kinematics_modulators(batch)

        with torch.enable_grad():
            z_kin, u_v_p, _ = self._solve_kinematics_macro(
                kin_encoded,
                kin_in,
                batch,
                mod_adv,
                mod_rheo,
                mod_curve,
                z_kin_ws,
                use_mu_decoder=True,
                current_mu_eff=current_mu_eff,
            )

        prior_tail = self._kinematics_prior_tail(batch, u_v_p[:, 0], u_v_p[:, 1])
        if prior_tail is None:
            bio_in = torch.cat([current_species, u_v_p, batch.x[:, :15]], dim=-1)
        else:
            bio_in = torch.cat([current_species, u_v_p, batch.x[:, :15], prior_tail], dim=-1)

        z = self.bio_encoder(bio_in)
        raw_species = self.biochem_decoder(z)
        next_species_flat = self._decode_species_log1p(raw_species)

        wall_mask_view = batch.mask_wall.view(-1, 1).float()
        surface_species = next_species_flat[:, 9:12] * wall_mask_view
        return torch.cat([next_species_flat[:, 0:9], surface_species], dim=1)

    def forward(
        self,
        batch,
        evaluation_times,
        y_true_trajectory=None,
        teacher_forcing_ratio=0.0,
        start_idx=0,
        initial_species=None,
        detach_macro_state=None,
    ):
        """
        Forward pass for the Neural ODE with Two-Way Macro-Micro Coupling.

        Designed for long clinical time horizons (e.g., 10,000 - 30,000 seconds with ~150s
        macro-steps, yielding 66-200 steps per trajectory).

        Args:
            batch: PyG graph batch containing spatial features.
            evaluation_times: A 1D tensor of times [ 0.0, t1, t2, ..., t_n ] to evaluate the clot state.
            y_true_trajectory: Ground truth log-space species trajectory for scheduled sampling.
            teacher_forcing_ratio: Probability (0.0 to 1.0) of injecting the ground truth state at each step.
            start_idx: Starting index for Truncated BPTT (TBPTT) offset.
            initial_species: Optional detached state from a previous chunk for warm-starting TBPTT.
            detach_macro_state: If True, detaches the computational graph at each macro-step
                to prevent OOM on long 200-step rollouts.
        """
        num_nodes = int(batch.x.shape[0])
        device = batch.x.device
        if self.training:
            self.ode_func.derivative_energy_sum = 0.0
            self.ode_func.derivative_eval_count = 0
        self._last_mu_trigger_gate = None
        self._last_mu_delta_bulk = None
        self._last_mu_delta_tail = None
        self._last_mu_wall_gate = None
        self._last_mu_delta_wall = None
        self._last_log_mu_before_wall = None
        self._last_mu_nucleation_prob = None
        self._last_mu_growth_prob = None
        self._last_mu_nucleation_cue = None

        truth_mask = biochem_truth_node_mask(batch, num_nodes, device)
        species_prior = _default_resting_species(num_nodes, device, batch)

        u_ref = batch.u_ref.view(-1, 1)
        d_bar = batch.d_bar.view(-1, 1)
        num_times = len(evaluation_times)

        # ==========================================
        # 1. INITIALIZE BIOCHEMICAL SPECIES
        # Default: physical resting prior everywhere. With TBPTT (start_idx > 0) or full-sequence TF,
        # anchor nodes take species from ``y`` / ``y_true_trajectory``; non-anchors keep the prior
        # (nodes without COMSOL-matched truth keep the resting prior). Synthetic graphs: all-prior at t=0.
        # Optional ``initial_species`` lets the caller warm-start from a pre-rolled state.
        # ==========================================
        if initial_species is not None:
            current_species = initial_species.to(device=device, dtype=species_prior.dtype)
        elif hasattr(batch, 'y') and batch.y is not None and start_idx > 0:
            safe_start_idx = min(start_idx, int(batch.y.shape[0]) - 1)
            gt_species = batch.y[safe_start_idx, :, 4:16].to(device)
            current_species = torch.where(truth_mask.unsqueeze(-1), gt_species, species_prior)
        elif y_true_trajectory is not None and y_true_trajectory.shape[0] > 0:
            gt_species = y_true_trajectory[0, :, 4:16].to(device)
            current_species = torch.where(truth_mask.unsqueeze(-1), gt_species, species_prior)
        else:
            current_species = species_prior
        current_species = torch.clamp(current_species, min=_SPECIES_LOG1P_MIN, max=_SPECIES_LOG1P_MAX)

        # USE CENTRALIZED RHEOLOGY PARAMS
        mu_inf = self.phys_cfg.mu_inf
        mu_0 = self.phys_cfg.mu_0
        lam = self.phys_cfg.lam
        n_idx = self.phys_cfg.n
        mu_nd_scale = self.phys_cfg.mu_viscosity_nd_scale

        mu_ic_steady_kin = _biochem_mu_ic_steady_kin_enabled()
        mu_k10d_simple = _biochem_mu_k10d_simple_enabled()
        mu_k10e_simple = _biochem_mu_k10e_simple_enabled()
        current_mu_eff = self._initial_mu_eff_si(batch)
        mu_eff_ic_steady_si: torch.Tensor | None = None
        mu_ss_const: torch.Tensor | None = None

        # Warm-start for the kinematic DEQ: carried detached between macro time steps so TBPTT
        # does not retain a cross-time graph through the fixed-point iteration.
        kin_in_ws = self._slice_kinematics_input(batch)
        uv_prior_ws = kin_in_ws[:, NodeFeat.UV_PRIOR]
        p_prior_ws = kin_in_ws[:, NodeFeat.SHEAR_POT]
        mu_prior_ws = kin_in_ws[:, NodeFeat.MU_PRIOR]
        priors_ws = torch.cat([uv_prior_ws, p_prior_ws, mu_prior_ws], dim=1)
        z_kin_ws = self._kinematics_deq_warm_start(kin_in_ws, priors_ws)

        mod_adv, mod_rheo, mod_curve = self._compute_kinematics_modulators(batch)

        if mu_ic_steady_kin:
            mu_eff_ic_steady_si, _u_ic, z_kin_ws = self._steady_kinematics_mu_uv(
                batch, z_kin_ws, mod_adv, mod_rheo, mod_curve
            )
            current_mu_eff = mu_eff_ic_steady_si.clone()

        pred_trajectory = []
        detach_macro_state = self.detach_macro_state_default if detach_macro_state is None else bool(detach_macro_state)
        prev_clot_gate = None

        # ==========================================
        # 2. MACRO-MICRO STEPPING (Two-Way Coupling)
        # ==========================================
        use_gt_kine = _biochem_gt_kine_vel_enabled()
        skip_deq_for_gt = use_gt_kine and _biochem_gt_kine_skip_deq()

        for i in range(num_times):
            # --- A. MACRO STEP: SOLVE KINEMATICS (Frozen Biochemistry) ---
            kin_in = self._slice_kinematics_input(batch, clone=True)

            # ND viscosity for kinematics matches label channel (mu_viscosity_nd_scale)
            kin_in[:, NodeFeat.MU_PRIOR] = current_mu_eff / mu_nd_scale
            kin_encoded = self._apply_fourier_encoding(kin_in)

            u_v_p = None
            if skip_deq_for_gt:
                u_v_p = resolve_gt_kine_uvp_at_step(
                    batch,
                    y_true_trajectory,
                    i,
                    truth_mask,
                    device,
                    kin_encoded.dtype,
                )
            if u_v_p is None:
                z_kin, u_v_p, z_kin_ws = self._solve_kinematics_macro(
                    kin_encoded,
                    kin_in,
                    batch,
                    mod_adv,
                    mod_rheo,
                    mod_curve,
                    z_kin_ws,
                    use_mu_decoder=(i == 0),
                    current_mu_eff=current_mu_eff,
                )
            else:
                z_kin = z_kin_ws
            if use_gt_kine and not skip_deq_for_gt:
                gt_uvp = resolve_gt_kine_uvp_at_step(
                    batch,
                    y_true_trajectory,
                    i,
                    truth_mask,
                    device,
                    u_v_p.dtype,
                    fallback_uvp=u_v_p,
                )
                if gt_uvp is not None:
                    u_v_p = gt_uvp

            # --- B. UPDATE DYNAMIC RHEOLOGY FOR CURRENT TIME ---
            u_nd = u_v_p[:, 0:1]
            v_nd = u_v_p[:, 1:2]

            du_dx_nd = torch.sparse.mm(batch.G_x, u_nd)
            du_dy_nd = torch.sparse.mm(batch.G_y, u_nd)
            dv_dx_nd = torch.sparse.mm(batch.G_x, v_nd)
            dv_dy_nd = torch.sparse.mm(batch.G_y, v_nd)

            scale_grad = u_ref / d_bar
            gamma_dot = torch.sqrt(2 * ((du_dx_nd * scale_grad) ** 2 + (dv_dy_nd * scale_grad) ** 2) +
                                   ((du_dy_nd * scale_grad) + (dv_dx_nd * scale_grad)) ** 2 + 1e-8)
            dshear_dx = torch.sparse.mm(batch.G_x, gamma_dot)
            dshear_dy = torch.sparse.mm(batch.G_y, gamma_dot)
            vel_mag = torch.sqrt(u_nd ** 2 + v_nd ** 2) + 1e-8
            u_dir = u_nd / vel_mag
            v_dir = v_nd / vel_mag
            dshear_ds = (u_dir * dshear_dx) + (v_dir * dshear_dy)
            dshear_ds_phys = dshear_ds / torch.clamp(d_bar, min=1e-8)

            # 1) Pure shear-thinning kinematic baseline (no species coupling).
            mu_kin_baseline = mu_inf + (mu_0 - mu_inf) * torch.pow(
                1.0 + (lam * gamma_dot) ** 2, (n_idx - 1.0) / 2.0
            )

            # 2) Biochem gelation: explicit FI/Mat gates + species-only learned nonnegative penalty.
            sp_safe = torch.clamp(current_species, _SPECIES_LOG1P_MIN, _SPECIES_LOG1P_MAX)
            species_si = self.species_log_nd_to_si(sp_safe)
            FI_si = species_si[:, 8:9]
            # STRICT COMSOL PARITY: viscosity depends on Mat (channel 11) only.
            Mat_si = species_si[:, 11:12]

            if _biochem_mu_disable_explicit_gelation():
                explicit_gelation = torch.zeros_like(Mat_si)
                learned_gelation = torch.zeros_like(Mat_si)
            else:
                mu1_term = (
                    torch.zeros_like(Mat_si)
                    if _biochem_mu_disable_mu1()
                    else self.mu1_sigmoid(Mat_si)
                )
                mu2_term = (
                    torch.zeros_like(FI_si)
                    if _biochem_mu_disable_mu2()
                    else self.mu2_sigmoid(FI_si)
                )
                mu2_cap_raw = (os.environ.get("BIOCHEM_MU2_SIGMOID_CAP") or "").strip()
                if mu2_cap_raw:
                    mu2_cap = max(float(mu2_cap_raw), 0.0)
                    mu2_term = torch.clamp(mu2_term, max=mu2_cap)
                explicit_gelation = mu1_term + mu2_term
                learned_gelation = self.learned_clot_penalty(sp_safe)
            gel_extra = explicit_gelation + learned_gelation
            if _biochem_gelation_prior_gate_enabled():
                # Wall-localised [0,1] map: keeps bulk lumen near pure Carreau (μ_kin_baseline) unless
                # kinematics indicate clot-risk (separation / low-shear × wall proximity).
                props_g = {"u_ref": u_ref.to(dtype=torch.float32), "d_bar": d_bar.to(dtype=torch.float32)}
                p_gate = clot_prior_score_flat(
                    batch,
                    u_nd.reshape(-1),
                    v_nd.reshape(-1),
                    self._bio_cfg,
                    props_g,
                )
                p_gate = p_gate.detach().clamp(0.0, 1.0).reshape(-1, 1).to(dtype=gel_extra.dtype)
                gel_extra = gel_extra * p_gate
            total_multiplier = 1.0 + gel_extra
            current_mu_eff = mu_kin_baseline * total_multiplier
            if _biochem_delta_mu_head_enabled():
                if _biochem_split_mu_regime_head_enabled():
                    # Trigger features (SI/physics aligned):
                    # [log1p species(12), FI_si, Mat_si, gamma_dot, wall_prox, wall_mask,
                    #  adverse_shear_cue, low_shear_cue]
                    sdf_nd = kin_in[:, NodeFeat.SDF]
                    wall_prox = torch.exp(-torch.abs(sdf_nd)).to(dtype=sp_safe.dtype)
                    wall_mask = batch.mask_wall.view(-1, 1).to(dtype=sp_safe.dtype)
                    adverse_shear_cue = torch.relu(-(dshear_ds_phys - self.sgt)).to(dtype=sp_safe.dtype)
                    adverse_shear_cue = adverse_shear_cue / (adverse_shear_cue + abs(float(self.sgt)) + 1e-6)
                    low_shear_cue = 1.0 / (1.0 + gamma_dot.to(dtype=sp_safe.dtype))
                    trigger_feats = torch.cat(
                        [
                            sp_safe,
                            FI_si,
                            Mat_si,
                            gamma_dot.to(dtype=sp_safe.dtype),
                            wall_prox,
                            wall_mask,
                            adverse_shear_cue,
                            low_shear_cue,
                        ],
                        dim=1,
                    )
                    gate_temp = max(self.mu_trigger_gate_temp * max(self.T_scale, 0.25), 1e-5)
                    use_nucleation_growth = _biochem_clot_nucleation_growth_enabled()
                    if use_nucleation_growth:
                        nuc_logits = self.mu_nucleation_gate_head(trigger_feats) + self.mu_clot_nucleation_bias
                        growth_logits = self.mu_growth_gate_head(trigger_feats) + self.mu_clot_growth_bias
                        nucleation_prob = torch.sigmoid(torch.clamp(nuc_logits / gate_temp, min=-50.0, max=50.0))
                        growth_prob = torch.sigmoid(torch.clamp(growth_logits / gate_temp, min=-50.0, max=50.0))
                        if prev_clot_gate is None:
                            gate = nucleation_prob
                        else:
                            growth_mix = (self.mu_clot_growth_memory * prev_clot_gate) + (
                                (1.0 - self.mu_clot_growth_memory) * growth_prob
                            )
                            gate = torch.maximum(nucleation_prob, growth_mix)
                        wall_prior = torch.maximum(wall_prox, self.mu_clot_wall_prior_mix * wall_mask)
                        gate = gate * torch.clamp(wall_prior, min=self.mu_clot_wall_prior_floor, max=1.0)
                    else:
                        gate_logits = self.mu_trigger_gate_head(trigger_feats)
                        gate = torch.sigmoid(torch.clamp(gate_logits / gate_temp, min=-50.0, max=50.0))
                        nucleation_prob = gate
                        growth_prob = gate
                    # Physics prior: suppress neural tail/wall corrections when no local clotting mass exists.
                    bio_suppressor_enabled = (
                        (os.environ.get("BIOCHEM_USE_BIO_GATE_SUPPRESSOR", "0") or "").strip().lower()
                        in ("1", "true", "yes", "on")
                    )
                    if bio_suppressor_enabled:
                        suppressor_thresh = max(
                            float(os.environ.get("BIOCHEM_BIO_SUPPRESSOR_THRESHOLD_SI", "1e-4")),
                            1e-8,
                        )
                        suppressor_power = max(
                            float(os.environ.get("BIOCHEM_BIO_SUPPRESSOR_POWER", "1.0")),
                            0.1,
                        )
                        bio_gate_floor = min(
                            max(float(os.environ.get("BIOCHEM_BIO_SUPPRESSOR_GATE_FLOOR", "0.0")), 0.0),
                            0.95,
                        )
                        bio_signal = torch.clamp((FI_si + Mat_si) / suppressor_thresh, min=0.0, max=1.0)
                        bio_signal = torch.pow(bio_signal, suppressor_power)
                        if bio_gate_floor > 0.0:
                            bio_signal = torch.clamp(bio_signal, min=bio_gate_floor, max=1.0)
                        # Detached to avoid rewarding artificial FI/Mat spikes to open the gate.
                        gate = gate * bio_signal.detach()
                    gate_hard_thresh = _mu_trigger_gate_hard_threshold()
                    soft_gate_steepness = self._mu_soft_gate_steepness()
                    if gate_hard_thresh > 0.0 and _mu_soft_gate_scope() == "all":
                        gate = _apply_mu_gate_soft_threshold(gate, steepness=soft_gate_steepness)
                    else:
                        trigger_gate_min = min(
                            max(float(os.environ.get("BIOCHEM_TRIGGER_GATE_MIN", "0.0")), 0.0),
                            0.95,
                        )
                        if trigger_gate_min > 0.0:
                            gate = torch.clamp(gate, min=trigger_gate_min, max=1.0)
                    delta_bulk_raw = self.mu_delta_bulk_head(torch.cat([z_kin, sp_safe], dim=1))
                    delta_tail_raw = self.mu_delta_tail_head(torch.cat([z_kin, trigger_feats], dim=1))
                    bulk_clip = float(os.environ.get("BIOCHEM_DELTA_MU_LOG_CLIP_BULK", "1.5"))
                    wall_clip = float(os.environ.get("BIOCHEM_DELTA_MU_LOG_CLIP_WALL", "5.0"))
                    if _biochem_delta_mu_symmetric_bulk_clip():
                        delta_bulk = torch.clamp(delta_bulk_raw, min=-bulk_clip, max=bulk_clip)
                    else:
                        delta_bulk = delta_bulk_raw
                    if _biochem_mu_gemini_fix_enabled():
                        delta_tail = torch.clamp(delta_tail_raw, min=0.0, max=wall_clip)
                    else:
                        delta_tail = delta_tail_raw
                    if _biochem_mu_additive_delta_enabled():
                        delta_log_mu_bulk_tail = delta_bulk + (gate * delta_tail)
                    else:
                        delta_log_mu_bulk_tail = ((1.0 - gate) * delta_bulk) + (gate * delta_tail)
                    delta_log_mu = delta_log_mu_bulk_tail

                    if _biochem_wall_delta_head_enabled():
                        wall_mask = batch.mask_wall.view(-1, 1).to(dtype=sp_safe.dtype)
                        wall_signal_val = torch.maximum(
                            wall_prox,
                            self.mu_wall_mask_mix * wall_mask,
                        )
                        wall_gate_min = min(
                            max(float(os.environ.get("BIOCHEM_WALL_GATE_MIN", "0.0")), 0.0),
                            0.99,
                        )
                        wall_gate = _wall_gate_from_signal(
                            wall_signal_val,
                            center=self.mu_wall_gate_center,
                            temp=self.mu_wall_gate_temp,
                            t_scale=float(self.T_scale),
                            logit_bias=float(self._mu_wall_gate_logit_bias),
                            gate_min=wall_gate_min,
                        )

                        if bio_suppressor_enabled:
                            # --- FIXED: Listen to the Alpha environment variable! ---
                            wall_alpha = float(os.environ.get("BIOCHEM_BIO_SUPPRESS_WALL_ALPHA", "1.0"))
                            # If alpha=0, effective_bio is 1.0 (no suppression).
                            # If alpha=1, effective_bio is strictly the biological mass.
                            effective_bio = bio_signal.detach() * wall_alpha + (1.0 - wall_alpha)
                            wall_gate = wall_gate * effective_bio

                        if gate_hard_thresh > 0.0:
                            wall_gate = _apply_mu_gate_soft_threshold(
                                wall_gate, steepness=soft_gate_steepness
                            )
                            wall_gate = wall_gate * _apply_mu_gate_soft_threshold(
                                wall_signal_val, steepness=soft_gate_steepness
                            )

                        wall_isolate_geom = (
                            (os.environ.get("BIOCHEM_WALL_HEAD_ISOLATE_GEOM", "0") or "").strip().lower()
                            in ("1", "true", "yes", "on")
                        )
                        # Keep wall-head tensor width unchanged while allowing geometric isolation/blending.
                        geom_only_feats = torch.cat(
                            [
                                torch.zeros_like(sp_safe),
                                torch.zeros_like(FI_si),
                                torch.zeros_like(Mat_si),
                                sdf_nd.view(-1, 1).to(dtype=sp_safe.dtype),
                                wall_prox,
                                torch.zeros_like(wall_mask),
                                torch.zeros_like(adverse_shear_cue),
                                torch.zeros_like(low_shear_cue),
                            ],
                            dim=1,
                        )
                        geom_blend = min(
                            max(float(os.environ.get("BIOCHEM_WALL_HEAD_GEOM_BLEND", "0.0")), 0.0),
                            1.0,
                        )
                        if wall_isolate_geom:
                            geom_blend = 1.0
                        wall_input_feats = ((1.0 - geom_blend) * trigger_feats) + (geom_blend * geom_only_feats)

                        delta_wall_raw = self.mu_delta_wall_head(
                            torch.cat([z_kin, wall_input_feats], dim=1)
                        )
                        if _biochem_mu_gemini_fix_enabled():
                            delta_wall = torch.clamp(
                                delta_wall_raw, min=0.0, max=wall_clip
                            )
                        else:
                            delta_wall = delta_wall_raw
                        if (
                            (os.environ.get("BIOCHEM_WALL_SPATIAL_DECAY", "0") or "").strip().lower()
                            in ("1", "true", "yes", "on")
                        ):
                            wall_decay_factor = max(
                                float(os.environ.get("BIOCHEM_WALL_SPATIAL_DECAY_FACTOR", "6.0")),
                                0.0,
                            )
                            wall_decay_floor = min(
                                max(float(os.environ.get("BIOCHEM_WALL_SPATIAL_DECAY_FLOOR", "0.0")), 0.0),
                                1.0,
                            )
                            wall_dist = torch.abs(sdf_nd).view(-1, 1).to(dtype=sp_safe.dtype)
                            wall_decay = torch.exp(-wall_dist * wall_decay_factor)
                            if wall_decay_floor > 0.0:
                                wall_decay = wall_decay_floor + ((1.0 - wall_decay_floor) * wall_decay)
                            delta_wall = delta_wall * wall_decay
                        self._last_log_mu_before_wall = (
                            torch.log(current_mu_eff.clamp(min=1e-8)) + delta_log_mu
                        )
                        wall_delta_act = _mu_wall_branch_delta(delta_wall)
                        wall_mix = _biochem_mu_wall_mix_mode()
                        if wall_mix == "relu_add":
                            wall_term = self.mu_wall_delta_gain * wall_delta_act
                            delta_log_mu = delta_log_mu + (wall_mask * wall_term)
                            wall_gate = torch.ones_like(wall_gate)
                        else:
                            wall_gate = _apply_wall_gate_curriculum(
                                wall_gate,
                                wall_mask,
                                teacher_epoch=self._biochem_teacher_epoch,
                                curriculum_epochs=self._mu_wall_gate_curriculum_epochs,
                            )
                            delta_log_mu = delta_log_mu + (
                                self.mu_wall_delta_gain * wall_gate * wall_delta_act
                            )
                        self._last_mu_wall_gate = wall_gate
                        self._last_mu_delta_wall = delta_wall

                    delta_log_mu = torch.clamp(
                        delta_log_mu,
                        min=-wall_clip,
                        max=wall_clip,
                    )

                    # Expose lightweight diagnostics for training loss/metrics.
                    self._last_mu_trigger_gate = gate
                    self._last_mu_delta_bulk = delta_bulk
                    self._last_mu_delta_tail = delta_tail
                    self._last_mu_nucleation_prob = nucleation_prob
                    self._last_mu_growth_prob = growth_prob
                    self._last_mu_nucleation_cue = (
                        0.50 * adverse_shear_cue + 0.20 * wall_prox + 0.30 * torch.clamp((FI_si + Mat_si), min=0.0, max=1.0)
                    ).detach()
                    prev_clot_gate = gate.detach() if detach_macro_state else gate
                else:
                    delta_in = torch.cat([z_kin, sp_safe], dim=1)
                    delta_log_mu = self.mu_delta_head(delta_in)
                effective_delta_clip = self.mu_delta_log_clip
                if (
                    (os.environ.get("BIOCHEM_USE_SPLIT_MU_HEAD", "0") or "").strip().lower()
                    in ("1", "true", "yes", "on")
                ):
                    effective_delta_clip = max(
                        effective_delta_clip,
                        float(os.environ.get("BIOCHEM_DELTA_MU_LOG_CLIP_WALL", "5.0")),
                    )
                delta_log_mu = torch.clamp(delta_log_mu, min=-effective_delta_clip, max=effective_delta_clip)
                if mu_ic_steady_kin and i == 0 and mu_eff_ic_steady_si is not None:
                    current_mu_eff = mu_eff_ic_steady_si.clone()
                else:
                    current_mu_eff = current_mu_eff * torch.exp(delta_log_mu)
            elif mu_ic_steady_kin and i == 0 and mu_eff_ic_steady_si is not None:
                current_mu_eff = mu_eff_ic_steady_si.clone()
            elif mu_k10d_simple or mu_k10e_simple:
                sp_safe = torch.clamp(current_species, _SPECIES_LOG1P_MIN, _SPECIES_LOG1P_MAX)
                if mu_ss_const is None:
                    mu_ss_const, _, z_kin_ws = self._steady_kinematics_mu_uv(
                        batch, z_kin_ws, mod_adv, mod_rheo, mod_curve
                    )
                learned_delta_raw = F.softplus(
                    self.mu_delta_head(torch.cat([z_kin, sp_safe], dim=1))
                )
                if mu_k10e_simple:
                    if mu_ic_steady_kin and i == 0 and mu_eff_ic_steady_si is not None:
                        current_mu_eff = mu_eff_ic_steady_si.clone()
                        self._last_k10e_adj_mask = torch.zeros_like(mu_eff_ic_steady_si)
                        self._last_mu_delta_bulk = torch.zeros_like(mu_eff_ic_steady_si)
                    else:
                        learned_delta_nd = torch.clamp(
                            learned_delta_raw,
                            max=max(_k10e_env_float("BIOCHEM_K10E_MU_DELTA_ND_MAX", 18.0), 1e-6),
                        )
                        sdf_nd = kin_in[:, NodeFeat.SDF]
                        wall_mask = batch.mask_wall.view(-1, 1).to(dtype=learned_delta_nd.dtype)
                        adj_mask = k10e_wall_adjacent_mask(sdf_nd, wall_mask)
                        if _biochem_env_truthy("BIOCHEM_K10E_CORONA_GROWTH", default=True):
                            adj_mask = _k10e_dilate_adjacent_mask(
                                adj_mask, batch.edge_index.to(device=adj_mask.device)
                            )
                        learned_delta_si = learned_delta_nd * mu_nd_scale
                        if (
                            _biochem_k10g_oracle_clots_enabled()
                            and y_true_trajectory is not None
                            and int(y_true_trajectory.shape[0]) > i
                            and int(y_true_trajectory.shape[-1]) > PredChannels.MU_EFF_ND
                        ):
                            mu_gt_si = self.phys_cfg.viscosity_nd_to_si(
                                y_true_trajectory[i, :, PredChannels.MU_EFF_ND : PredChannels.MU_EFF_ND + 1]
                            ).to(device=mu_ss_const.device, dtype=mu_ss_const.dtype)
                            clot_boost = (mu_gt_si - mu_ss_const).clamp(min=0.0)
                            current_mu_eff = mu_ss_const + (adj_mask * clot_boost)
                            self._last_mu_delta_bulk = (clot_boost / mu_nd_scale).detach()
                        else:
                            current_mu_eff = mu_ss_const + (adj_mask * learned_delta_si)
                            self._last_mu_delta_bulk = learned_delta_nd.detach()
                        self._last_k10e_adj_mask = adj_mask.detach()
                else:
                    learned_delta = learned_delta_raw
                    max_delta_raw = (os.environ.get("BIOCHEM_K10D_MU_DELTA_SI_MAX") or "").strip()
                    if max_delta_raw:
                        learned_delta = torch.clamp(learned_delta, max=max(float(max_delta_raw), 1e-8))
                    current_mu_eff = mu_ss_const + learned_delta
                    self._last_mu_delta_bulk = learned_delta.detach()
                self._last_mu_delta_tail = torch.zeros_like(self._last_mu_delta_bulk)
                self._last_mu_trigger_gate = torch.zeros_like(self._last_mu_delta_bulk)
            current_mu_eff = torch.clamp(current_mu_eff, min=1e-8)

            # ==========================================
            # BIOCHEM ATTENTION MODULATOR (Streamwise + detached)
            # ==========================================
            row, _ = batch.edge_index
            dshear_edge = dshear_ds_phys[row]

            scaled_temp = max(self.T_grad * self.T_scale, 1e-5)
            separation_logits = -(dshear_edge - self.sgt) / scaled_temp
            is_separation_edge = torch.sigmoid(torch.clamp(separation_logits, min=-50.0, max=50.0))
            mod_separation = (is_separation_edge * self.biochem_attention_boost).detach()

            # --- C. RECORD COUPLED STATE (Record prediction first) ---
            current_mu_eff_nd = current_mu_eff / mu_nd_scale
            # Use the ND version for the recorded trajectory (species at this macro time, pre-TF).
            pred_step = torch.cat([u_v_p, current_mu_eff_nd, sp_safe], dim=-1)
            pred_trajectory.append(pred_step)

            # --- C.5 TEACHER FORCING INJECTION (For next ODE step only) ---
            if self.training and y_true_trajectory is not None and i > 0:
                # Scheduled sampling in log1p-space states for long trajectories (66-200 steps):
                # We DO NOT blend values linearly in log-space, as this physically skews concentrations.
                # Instead, choose 100% GT or 100% model state for anchor nodes via a probabilistic coin flip.
                tf = float(max(0.0, min(1.0, teacher_forcing_ratio)))
                if tf > 0.0:
                    if tf >= 1.0 - 1e-6:
                        use_ground_truth = True
                    else:
                        use_ground_truth = bool((torch.rand((), device=device) < tf).item())
                    if use_ground_truth:
                        gt_species = y_true_trajectory[i, :, 4:16].to(device)
                        current_species = torch.where(truth_mask.unsqueeze(-1), gt_species, current_species)
                        current_species = torch.clamp(current_species, min=_SPECIES_LOG1P_MIN, max=_SPECIES_LOG1P_MAX)

            # --- D. MICRO STEP: INTEGRATE BIOCHEMISTRY (Frozen Kinematics) ---
            if i < num_times - 1:
                # Physical time [s] so dz/dt matches finite-difference d_pred_dt in training losses.
                t_span = evaluation_times[i: i + 2]

                # Encode current physical state into latent representation
                safe_species = torch.clamp(current_species, min=_SPECIES_LOG1P_MIN, max=_SPECIES_LOG1P_MAX)
                prior_tail = self._kinematics_prior_tail(batch, u_v_p[:, 0], u_v_p[:, 1])
                if prior_tail is None:
                    bio_in = torch.cat([safe_species, u_v_p, batch.x[:, :15]], dim=-1)
                else:
                    bio_in = torch.cat([safe_species, u_v_p, batch.x[:, :15], prior_tail], dim=-1)
                z_current = self.bio_encoder(bio_in)

                # Integrate ODE over the Delta t interval (adjoint: memory-safe backward).
                dt_seg = float((t_span[-1] - t_span[0]).abs().item())
                _min_dt = 1e-9
                if dt_seg < _min_dt:
                    # Duplicate/near-duplicate timestamps → no evolution.
                    z_next = z_current
                else:
                    def odefunc_wrapper(t, z):
                        batch_idx = get_batch_tensor(batch, num_nodes, device)
                        dz = self.ode_func(t, z, batch.edge_index, batch.edge_attr, batch_idx, mod_separation)
                        return torch.clamp(dz, min=-10.0, max=10.0)

                    # Use an explicit method (like "rk4") by default for large 150s jumps to avoid
                    # implicit solver history-drop penalties on restarted biochemical segment solves.
                    solver_method = self.micro_ode_method
                    ode_kwargs_base = dict(
                        method=solver_method,
                        adjoint_method=solver_method,
                        adjoint_params=tuple(self.ode_func.parameters()),
                        rtol=self.rtol,
                        atol=self.atol,
                        adjoint_rtol=self.rtol,
                        adjoint_atol=self.atol,
                    )
                    # ``evaluation_times`` are nondimensional (see ``to_t_nd(..., t_final)``); env cap is SI seconds.
                    max_step_env = (os.environ.get("BIOCHEM_ODE_MAX_STEP_S") or "").strip()
                    max_step_phys_s = max(float(max_step_env), 1e-9) if max_step_env else 10.0
                    t_int_ref = float(getattr(self._bio_cfg, "t_final", 30000.0))
                    max_step_nd = max_step_phys_s / max(t_int_ref, 1e-12)
                    rk_sub_env = (os.environ.get("BIOCHEM_ADJOINT_RK4_SUBSTEPS") or "").strip()
                    n_rk_sub = max(1, int(rk_sub_env) if rk_sub_env else 32)
                    use_adjoint = (os.environ.get("BIOCHEM_ODEINT_USE_ADJOINT", "1") or "").strip().lower() not in (
                        "0",
                        "false",
                        "no",
                        "off",
                    )

                    def _one_odeint(z0, span: torch.Tensor, step_hint: float) -> torch.Tensor:
                        ode_kwargs = dict(ode_kwargs_base)
                        if solver_method == "rk4":
                            # ``step_hint`` is the macro subsegment length; torchdiffeq's fixed ``rk4``
                            # would otherwise take a single step over the whole segment (unstable).
                            seg_len = abs(float(span[-1] - span[0]))
                            sh = max(seg_len / float(n_rk_sub), 1e-12)
                            ode_kwargs["options"] = {"step_size": sh}
                            ode_kwargs["adjoint_options"] = {"step_size": sh}
                        if use_adjoint:
                            z_out = odeint_adjoint(
                                odefunc_wrapper,
                                z0,
                                span,
                                **ode_kwargs,
                            )
                        else:
                            plain = dict(
                                method=solver_method,
                                rtol=self.rtol,
                                atol=self.atol,
                            )
                            if solver_method == "rk4" and "options" in ode_kwargs:
                                plain["options"] = ode_kwargs["options"]
                            z_out = odeint(odefunc_wrapper, z0, span, **plain)
                        return z_out[1]

                    if dt_seg > max_step_nd:
                        t0 = float(t_span[0].detach().item())
                        t1 = float(t_span[-1].detach().item())
                        duration = t1 - t0
                        n_sub = max(1, int(math.ceil(abs(duration) / max_step_nd)))
                        sub_dt = duration / float(n_sub)
                        z_run = z_current
                        for j in range(n_sub):
                            ts0 = t0 + j * sub_dt
                            ts1 = t0 + (j + 1) * sub_dt
                            t_sub = torch.tensor([ts0, ts1], device=device, dtype=t_span.dtype)
                            z_run = _one_odeint(z_run, t_sub, step_hint=sub_dt)
                        z_next = z_run
                    else:
                        z_next = _one_odeint(z_current, t_span, step_hint=dt_seg)
                raw_species = self.biochem_decoder(z_next)

                next_species_flat = self._decode_species_log1p(raw_species)

                # Enforce surface species only on walls
                wall_mask_view = batch.mask_wall.view(-1, 1).float()
                surface_species = next_species_flat[:, 9:12] * wall_mask_view

                # Update species state for the next macro-step
                current_species = torch.cat([next_species_flat[:, 0:9], surface_species], dim=1)
                current_species = torch.clamp(current_species, min=_SPECIES_LOG1P_MIN, max=_SPECIES_LOG1P_MAX)
                if detach_macro_state:
                    # Prevent OOM during BPTT across long 66-200 step macro trajectories on
                    # finite-memory GPUs by severing the carried-state graph each macro step.
                    current_species = current_species.detach()
                    current_mu_eff = current_mu_eff.detach()

        # Stack into shape: [ Time, Nodes, 16 ]
        pred_series = torch.stack(pred_trajectory, dim=0)

        return pred_series