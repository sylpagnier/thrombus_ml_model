"""Coupled clot-phi rollout (rung 6): serial MLP steps with optional kine feedback."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Any

import torch

from src.config import NodeFeat, PhysicsConfig
from src.utils.paths import get_project_root

VelSource = Literal["gt", "kinematics"]


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name) or ("1" if default else "0")).strip().lower()
    return raw in ("1", "true", "yes", "on")


def clot_phi_rollout_enabled() -> bool:
    return _env_bool("CLOT_PHI_ROLLOUT", False)


def clot_phi_rollout_detach_carry() -> bool:
    return _env_bool("CLOT_PHI_ROLLOUT_DETACH", True)


def clot_phi_carry_phi_enabled() -> bool:
    return _env_bool("CLOT_PHI_CARRY_PHI", True)


def clot_phi_carry_log_mu_enabled() -> bool:
    return _env_bool("CLOT_PHI_CARRY_LOG_MU", True)


def clot_phi_carry_gt_warmup_epochs() -> int:
    """Train-only: first N epochs feed log(GT mu @ ti) in carry slot (R1D-aligned)."""
    raw = (os.environ.get("CLOT_PHI_CARRY_GT_WARMUP_EPOCHS") or "0").strip()
    try:
        return max(int(raw), 0)
    except ValueError:
        return 0


def clot_phi_carry_gt_warmup_steps() -> int:
    """Train-only: first K macro indices per graph use GT log mu in carry slot (-1 = disabled)."""
    raw = (os.environ.get("CLOT_PHI_CARRY_GT_WARMUP_STEPS") or "0").strip()
    try:
        return int(raw)
    except ValueError:
        return 0


def clot_phi_carry_gt_fade_epochs() -> int:
    """Train-only: linear blend GT -> pred carry over this many epochs after warmup_epochs."""
    raw = (os.environ.get("CLOT_PHI_CARRY_GT_FADE_EPOCHS") or "0").strip()
    try:
        return max(int(raw), 0)
    except ValueError:
        return 0


def carry_gt_warmup_active(time_index: int, train_epoch: int | None) -> bool:
    """GT carry bridge applies only during training (eval always uses pred carry)."""
    if train_epoch is None:
        return False
    warm_ep = clot_phi_carry_gt_warmup_epochs()
    if warm_ep > 0 and train_epoch < warm_ep:
        return True
    warm_steps = clot_phi_carry_gt_warmup_steps()
    if warm_steps < 0:
        return True
    if warm_steps > 0 and int(time_index) < warm_steps:
        return True
    return False


def carry_gt_fade_alpha(train_epoch: int | None) -> float | None:
    """Blend weight on pred carry during fade window; None if not fading."""
    if train_epoch is None:
        return None
    warm_ep = clot_phi_carry_gt_warmup_epochs()
    fade_ep = clot_phi_carry_gt_fade_epochs()
    if fade_ep <= 0 or train_epoch < warm_ep:
        return None
    if train_epoch >= warm_ep + fade_ep:
        return None
    return float(train_epoch - warm_ep) / float(fade_ep)


def resolve_carry_log_mu_feature(
    *,
    time_index: int,
    train_epoch: int | None,
    gt_mu_cap_si: torch.Tensor,
    rollout_state: "ClotPhiRolloutState | None",
    device: torch.device,
) -> torch.Tensor | None:
    """Carry-column log mu: GT warm-up (R1D) -> optional fade -> pred carry."""
    if not clot_phi_carry_log_mu_enabled():
        return None
    log_gt = torch.log(gt_mu_cap_si.reshape(-1).to(device=device, dtype=torch.float32).clamp(min=1e-8))
    fade_a = carry_gt_fade_alpha(train_epoch)
    if fade_a is not None:
        if rollout_state is not None and rollout_state.log_mu_prev is not None:
            log_pred = rollout_state.log_mu_prev.reshape(-1).to(device=device, dtype=log_gt.dtype)
        else:
            log_pred = log_gt
        return (1.0 - fade_a) * log_gt + fade_a * log_pred
    if carry_gt_warmup_active(time_index, train_epoch):
        return log_gt
    if rollout_state is not None and rollout_state.log_mu_prev is not None:
        return rollout_state.log_mu_prev.reshape(-1).to(device=device)
    return None


def clot_phi_vel_source(data: Any | None = None) -> VelSource:
    raw = (
        os.environ.get("CLOT_PHI_VEL_SOURCE")
        or os.environ.get("T0_R4_FLOW_SOURCE")
        or "gt"
    ).strip().lower()
    if raw in ("kin", "kinematics", "deq", "gino"):
        return "kinematics"
    if data is not None:
        if not hasattr(data, "y") or data.y is None or data.y.numel() == 0 or bool((data.y == 0).all().item()):
            return "kinematics"
    return "gt"


def clot_phi_kine_teacher_forcing() -> float:
    """Blend pred vs GT [u,v] when vel_source=kinematics (0=pred only, 1=GT only)."""
    try:
        return max(0.0, min(float(os.environ.get("CLOT_PHI_KINE_TF", "0") or "0"), 1.0))
    except ValueError:
        return 0.0


def clot_phi_rollout_extra_feature_dim() -> int:
    if not clot_phi_rollout_enabled():
        return 0
    n = 0
    if clot_phi_carry_phi_enabled():
        n += 1
    if clot_phi_carry_log_mu_enabled():
        n += 1
    return n


def sync_rollout_env_from_checkpoint(cfg: dict) -> None:
    """Match rollout/carry env to ``cfg['in_dim']`` when legacy ckpts omit rollout keys."""
    if not cfg or "rollout" in cfg:
        return
    ckpt_in = int(cfg.get("in_dim", 0) or 0)
    if ckpt_in <= 0:
        return
    from src.core_physics.clot_phi_simple import clot_phi_feature_dim

    os.environ["CLOT_PHI_ROLLOUT"] = "0"
    os.environ["CLOT_PHI_CARRY_PHI"] = "0"
    os.environ["CLOT_PHI_CARRY_LOG_MU"] = "0"
    base = clot_phi_feature_dim()
    extra = ckpt_in - base
    if extra <= 0:
        return
    os.environ["CLOT_PHI_ROLLOUT"] = "1"
    if extra >= 1:
        os.environ["CLOT_PHI_CARRY_PHI"] = "1"
    if extra >= 2:
        os.environ["CLOT_PHI_CARRY_LOG_MU"] = "1"
    os.environ.setdefault("CLOT_PHI_VEL_SOURCE", str(cfg.get("rollout_vel_source") or "gt"))
    os.environ.setdefault("CLOT_PHI_ROLLOUT_DETACH", "1")


def append_rollout_carry_features(
    feats: torch.Tensor,
    *,
    phi_prev: torch.Tensor | None,
    log_mu_prev: torch.Tensor | None,
    n_nodes: int,
    device: torch.device,
    dtype: torch.dtype,
    log_mu_override: torch.Tensor | None = None,
) -> torch.Tensor:
    """Concatenate carry channels (zeros at t=0 when prev is None)."""
    cols: list[torch.Tensor] = [feats]
    if clot_phi_carry_phi_enabled():
        if phi_prev is None:
            cols.append(torch.zeros(n_nodes, 1, device=device, dtype=dtype))
        else:
            cols.append(phi_prev.reshape(-1, 1).to(device=device, dtype=dtype))
    if clot_phi_carry_log_mu_enabled():
        log_col = log_mu_override if log_mu_override is not None else log_mu_prev
        if log_col is None:
            cols.append(torch.zeros(n_nodes, 1, device=device, dtype=dtype))
        else:
            cols.append(log_col.reshape(-1, 1).to(device=device, dtype=dtype))
    return torch.cat(cols, dim=1)


@dataclass
class ClotPhiRolloutState:
    phi_prev: torch.Tensor | None = None
    log_mu_prev: torch.Tensor | None = None

    def update_from_pred(
        self,
        phi_pred: torch.Tensor,
        mu_pred_si: torch.Tensor,
        *,
        detach: bool,
    ) -> None:
        phi = phi_pred.reshape(-1)
        mu = mu_pred_si.reshape(-1)
        if detach:
            phi = phi.detach()
            mu = mu.detach()
        self.phi_prev = phi
        self.log_mu_prev = torch.log(mu.clamp(min=1e-8))


class KinematicsUvProvider:
    """One steady GINO-DEQ solve per clot step with ``MU_PRIOR`` from predicted mu."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._model = None
        self._phys_cfg: PhysicsConfig | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from src.architecture.ginodeq import GINO_DEQ
        from src.architecture.kinematics_model_config import (
            build_gino_deq_from_ctor,
            kinematics_checkpoint_tensors,
            resolve_gino_deq_ctor_kwargs,
        )
        from src.utils.paths import resolve_checkpoint

        ckpt_path = (os.environ.get("CLOT_PHI_KINE_CKPT") or "").strip()
        if not ckpt_path:
            ckpt_path = str(resolve_checkpoint("a", "kinematics_best.pth"))
        root = get_project_root()
        path = root / ckpt_path if not os.path.isabs(ckpt_path) else ckpt_path
        raw = torch.load(path, map_location=self.device, weights_only=False)
        meta, state = kinematics_checkpoint_tensors(raw)
        ctor = resolve_gino_deq_ctor_kwargs(meta, state)
        self._phys_cfg = PhysicsConfig(phase="kinematics")
        self._model = build_gino_deq_from_ctor(self._phys_cfg, ctor).to(self.device)
        self._model.load_state_dict(state, strict=False)
        self._model.eval()

    @torch.no_grad()
    def uv_nd_from_mu_si(
        self,
        data,
        mu_eff_si: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_loaded()
        assert self._model is not None and self._phys_cfg is not None
        batch = data
        mu_nd = self._phys_cfg.viscosity_si_to_nd(mu_eff_si.reshape(-1, 1))
        kin_in = batch.x.clone()
        kin_in[:, NodeFeat.MU_PRIOR] = mu_nd.to(device=kin_in.device, dtype=kin_in.dtype)
        batch_k = batch.clone()
        batch_k.x = kin_in
        pred = self._model(batch_k)
        out = pred[0] if isinstance(pred, tuple) else pred
        u = out[:, 0]
        v = out[:, 1]
        return u, v


_kine_provider: KinematicsUvProvider | None = None


def reset_rollout_kine_provider() -> None:
    """Clear cached GINO-DEQ provider (coupled rollout / per-anchor reset)."""
    global _kine_provider
    _kine_provider = None


def resolve_uv_for_rollout_step(
    data,
    time_index: int,
    mu_eff_si: torch.Tensor | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (u_nd, v_nd, u_gt, v_gt) for feature building at ``time_index``."""
    if hasattr(data, "y") and data.y is not None and data.y.numel() > 0:
        y = data.y[time_index].to(device=device)
        u_gt = y[:, 0]
        v_gt = y[:, 1]
    else:
        u_gt = torch.zeros(int(data.num_nodes), device=device, dtype=torch.float32)
        v_gt = torch.zeros(int(data.num_nodes), device=device, dtype=torch.float32)
    src = clot_phi_vel_source(data)
    if src == "gt" or mu_eff_si is None:
        return u_gt, v_gt, u_gt, v_gt
    global _kine_provider
    if _kine_provider is None or _kine_provider.device != device:
        _kine_provider = KinematicsUvProvider(device)
    u_p, v_p = _kine_provider.uv_nd_from_mu_si(data, mu_eff_si)
    tf = clot_phi_kine_teacher_forcing()
    if tf <= 0.0:
        return u_p, v_p, u_gt, v_gt
    if tf >= 1.0:
        return u_gt, v_gt, u_gt, v_gt
    u = (1.0 - tf) * u_p + tf * u_gt
    v = (1.0 - tf) * v_p + tf * v_gt
    return u, v, u_gt, v_gt


def snapshot_carry_gt_warmup_config() -> dict[str, int]:
    return {
        "carry_gt_warmup_epochs": clot_phi_carry_gt_warmup_epochs(),
        "carry_gt_warmup_steps": clot_phi_carry_gt_warmup_steps(),
        "carry_gt_fade_epochs": clot_phi_carry_gt_fade_epochs(),
    }
