"""Canonical biochem loss policy (frozen-kin teacher, viscosity + species goal).

See docs/BIOCHEM_TRAINING_PROGRESS.md section "Loss policy (approved vs deprecated)".

Set ``BIOCHEM_LEGACY_LOSSES=1`` to re-enable deprecated isolate keys, sweep presets, and
aux-loss blocks removed from the default forward path.
"""
from __future__ import annotations

import os
from typing import FrozenSet, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

_DOC = "docs/BIOCHEM_TRAINING_PROGRESS.md (Loss policy)"

# --- Approved for current Phase I teacher (GT flow, STOP_AFTER_TEACHER=1) ---

APPROVED_ISOLATE_KEYS: FrozenSet[str] = frozenset({
    "DATA_BIO",
    "DATA_KINE",
    "MU_LOG",
    "MU_SI",
    "PASSIVE",
    "ONE_WAY",
})

# Diagnostics only: short Phase-A probes, not promotion paths.
APPROVED_DIAGNOSTIC_ISOLATE_KEYS: FrozenSet[str] = frozenset({
    "ADR_S",
    "ADR_F",
})

# --- Deprecated: historical sweeps; chronicle retained in training progress doc ---

DEPRECATED_ISOLATE_KEYS: dict[str, str] = {
    "MU_LOG_WALL": "Wall-only isolate improves wall logMAE but wrecks all/high-mu (sweep_wall_overcomp).",
    "MU_LOG_HIGH": "High-tail isolate does not create spatial clots; all-truth stays poor.",
    "K10E": "Superseded by PASSIVE align + MU_LOG unlock + step-2 bridge; viz clots still failed.",
    "K11": "Clot-gate BCE sweeps (clot6h); gate collapse, not in current isolate resolver.",
    "W_BIO": "Wall bio flux alone is unstable / trivial in short probes.",
    "W_PHY": "Wall physics flux alone is schedule-sensitive, not a primary viscosity path.",
    "BIO_IO": "Bio in/out isolate does not move val mu.",
    "NS_MOM": "NS momentum residual dominated failed step-3 runs.",
    "KINE_PRIOR": "Kinematic prior auxiliary; not used on frozen GT flow path.",
    "PHYS_TEMP": "COMSOL temporal term did not beat step-2 baseline on val mu.",
    "LATENT": "ODE latent reg isolate; pretrain-only in practice.",
    "VISC": "Viscosity reg isolate; marginal.",
    "VISC_REG": "Alias for VISC.",
    "FI_GATE": "FI gate start penalty; no validated gain.",
    "FI_GATE_START": "Alias for FI_GATE.",
    "RES_SPARSE": "Residual sparsity prior; no validated spatial clot gain.",
    "RESIDUAL_SPARSE": "Alias for RES_SPARSE.",
    "PSEUDO": "Corrector pseudo labels (Phase II, not started).",
    "MU_MSE": "K10d proof only; use MU_LOG + delta head for iteration.",
    "MU_DATA": "Alias for MU_MSE.",
}

DEPRECATED_PRESET_ALIASES: dict[str, str] = {
    "sweep_wall_sentinel": "Wall-heavy MU_LOG sweep era.",
    "sweep_wall_overcomp": "MU_LOG_WALL overcompensation sweep.",
    "sweep_bio_suppressor": "Bio suppressor + MU_LOG; wall pinned ~2.59.",
    "sweep_clot_nuc_growth": "Clot nucleation growth sweep (pre passive-align).",
    "sweep_free_wall_a": "Free wall A geometry sweep.",
    "sweep_free_wall_b": "Free wall B high clot penalty sweep.",
    "sweep_gemini": "Gemini dlogmu clip experiment.",
    "thrombus_corona": "Full corrector + corona bundle (unvalidated).",
    "comprehensive_mu": "Long corona + mu kitchen sink (unvalidated).",
}

DEPRECATED_AUX_ENV_KEYS: FrozenSet[str] = frozenset({
    "BIOCHEM_TRIGGER_GATE_FLOOR_WEIGHT",
    "BIOCHEM_TRIGGER_LEARNED_FLOOR_WEIGHT",
    "BIOCHEM_CLOT_TRIGGER_SPARSITY_WEIGHT",
    "BIOCHEM_CLOT_TRIGGER_NONWALL_WEIGHT",
    "BIOCHEM_CLOT_NUCLEATION_ALIGN_WEIGHT",
    "BIOCHEM_FI_GATE_START_WEIGHT",
    "BIOCHEM_RESIDUAL_SPARSE_LAMBDA_START",
    "BIOCHEM_RESIDUAL_SPARSE_LAMBDA_END",
})


def _truthy(name: str, *, default: bool = False) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def biochem_legacy_losses_enabled() -> bool:
    return _truthy("BIOCHEM_LEGACY_LOSSES")


def biochem_legacy_aux_losses_enabled() -> bool:
    """Aux blocks (trigger floors, residual sparse, FI gate) computed only when legacy."""
    if biochem_legacy_losses_enabled():
        return True
    for key in DEPRECATED_AUX_ENV_KEYS:
        if float(os.environ.get(key, "0") or "0") > 0.0:
            return True
    return False


def normalize_isolate_key(key: str) -> str:
    k = (key or "").strip().upper()
    aliases = {
        "ADR_FAST": "ADR_F",
        "ADR_SLOW": "ADR_S",
        "WALL_BIO": "W_BIO",
        "WALL_PHYS": "W_PHY",
        "BIO_INOUT": "BIO_IO",
        "MOM": "NS_MOM",
        "DK": "DATA_KINE",
        "DB": "DATA_BIO",
        "KP": "KINE_PRIOR",
        "PT": "PHYS_TEMP",
        "MU_SI_ANCHOR": "MU_SI",
        "MU_LOG_ANCHOR": "MU_LOG",
    }
    return aliases.get(k, k)


def validate_isolate_key(key: str) -> None:
    """Raise if isolate is deprecated and legacy mode is off."""
    k = normalize_isolate_key(key)
    if not k:
        return
    if k in APPROVED_ISOLATE_KEYS or k in APPROVED_DIAGNOSTIC_ISOLATE_KEYS:
        return
    if biochem_legacy_losses_enabled():
        if k in DEPRECATED_ISOLATE_KEYS:
            print(
                f"[WARN]  BIOCHEM_LOSS_ISOLATE={k} is deprecated ({DEPRECATED_ISOLATE_KEYS[k]}) "
                f"LEGACY_LOSSES=1.",
                flush=True,
            )
        return
    if k in DEPRECATED_ISOLATE_KEYS:
        raise ValueError(
            f"BIOCHEM_LOSS_ISOLATE={k!r} is deprecated for the frozen-kin viscosity teacher path: "
            f"{DEPRECATED_ISOLATE_KEYS[k]} See {_DOC}. "
            "Set BIOCHEM_LEGACY_LOSSES=1 to reproduce old sweeps."
        )
    # Unknown keys are ignored by callers that filter against this policy.


def check_deprecated_preset(preset: str) -> bool:
    """Return True if preset applied (legacy) or not deprecated. False if blocked."""
    p = (preset or "").strip().lower()
    if not p or p not in DEPRECATED_PRESET_ALIASES:
        return True
    if biochem_legacy_losses_enabled():
        print(
            f"[WARN]  BIOCHEM_PRESET={p} is deprecated ({DEPRECATED_PRESET_ALIASES[p]}). LEGACY_LOSSES=1.",
            flush=True,
        )
        return True
    print(
        f"[i]  BIOCHEM_PRESET={p} ignored (deprecated: {DEPRECATED_PRESET_ALIASES[p]}). "
        f"See {_DOC}. Set BIOCHEM_LEGACY_LOSSES=1 to enable.",
        flush=True,
    )
    return False


def warn_step3_multitask_if_disabled() -> None:
    step = (os.environ.get("BIOCHEM_COMPLEXITY_STEP") or "").strip().lower()
    if step in ("3", "3.0", "phase3", "full_multitask", "corrector_full"):
        if not biochem_legacy_losses_enabled() and not _truthy("BIOCHEM_LOSS_DATA_ONLY", default=True):
            print(
                "[WARN]  BIOCHEM_COMPLEXITY_STEP=3 without LEGACY_LOSSES: step-3 Kendall multitask "
                "regressed vs MU_LOG teacher (~4.2 vs ~0.46 val). Prefer step-2 data-only.",
                flush=True,
            )


def approved_backward_summary() -> str:
    return (
        "Approved backward (default): PASSIVE/Data_Bio species lane; step-2 LOSS_DATA_ONLY + "
        "W_MuLog/W_MuSI; MU_LOG unlock (delta head, bio frozen); masked ADR via PASSIVE_ADR_BACKPROP "
        "after species stable. GT_KINE_VEL=1."
    )


class SpatialFocalLoss(nn.Module):
    """Focal BCE for sparse wall-band species triggers (crushes medium-confidence halos).

    ``alpha`` skews toward the minority positive (clot) class on imbalanced wall bands.
    ``alpha=0.90`` -> positive errors weighted ~9x vs negative.
    Pass per-channel ``alpha`` / ``gamma`` / ``channel_weight`` as length-C sequences.
    """

    def __init__(
        self,
        alpha: float | Sequence[float] = 0.90,
        gamma: float | Sequence[float] = 2.0,
        channel_weight: Sequence[float] | None = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.channel_weight = channel_weight

    def _broadcast(self, val: float | Sequence[float], *, ref: torch.Tensor) -> torch.Tensor:
        if isinstance(val, (list, tuple)):
            t = torch.tensor(val, device=ref.device, dtype=ref.dtype)
        else:
            t = torch.tensor(float(val), device=ref.device, dtype=ref.dtype)
        if t.ndim == 0:
            return t
        return t.view(1, -1)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        gamma = self._broadcast(self.gamma, ref=logits)
        focal_weight = (1.0 - p_t) ** gamma
        alpha = self._broadcast(self.alpha, ref=logits)
        alpha_weight = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_weight * focal_weight * bce_loss
        if self.channel_weight is not None:
            cw = self._broadcast(self.channel_weight, ref=logits)
            loss = loss * cw
        return loss.mean()


class ActiveGrowthHuberLoss(nn.Module):
    """Huber on GT-active growth nodes only + heavy FP penalty on inert ceiling nodes.

    Prevents zero-delta collapse when most wall-band nodes have ``target_delta ~= 0``.
    ``band_mask`` should be deployable (e.g. ceiling / wall+hops), not GT clot.
    ``value_scale`` lifts ~1e-5 log-deltas into an O(1) Huber domain for usable grads.
    """

    def __init__(
        self,
        *,
        delta_threshold: float = 1e-5,
        delta_threshold_channels: Sequence[float] | None = None,
        beta: float = 1.0,
        fp_weight: float = 5.0,
        fp_threshold: float | None = None,
        value_scale: float = 1e5,
        channel_weight: Sequence[float] | None = None,
        underpred_weight: float = 2.0,
        mature_frac: float = 0.95,
        mature_exempt_fp: bool = False,
        mature_max_log: Sequence[float] | None = None,
    ):
        super().__init__()
        self.delta_threshold = float(delta_threshold)
        if delta_threshold_channels is None:
            self.delta_threshold_channels = (self.delta_threshold, self.delta_threshold)
        else:
            self.delta_threshold_channels = tuple(float(x) for x in delta_threshold_channels)
        self.beta = float(beta)
        self.fp_weight = float(fp_weight)
        self.fp_threshold = float(fp_threshold) if fp_threshold is not None else self.delta_threshold
        self.value_scale = float(value_scale)
        self.channel_weight = channel_weight
        self.underpred_weight = float(underpred_weight)
        self.mature_frac = float(mature_frac)
        self.mature_exempt_fp = bool(mature_exempt_fp)
        self.mature_max_log = tuple(float(x) for x in mature_max_log) if mature_max_log else None

    def _channel_thr(self, ch: int) -> float:
        if ch < len(self.delta_threshold_channels):
            return float(self.delta_threshold_channels[ch])
        return self.delta_threshold

    def forward(
        self,
        pred_delta: torch.Tensor,
        target_delta: torch.Tensor,
        band_mask: torch.Tensor,
        current_log_state: torch.Tensor | None = None,
        fp_weight_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        m = band_mask.reshape(-1).to(device=pred_delta.device).bool()
        if not bool(m.any().item()):
            return pred_delta.sum() * 0.0
        p_raw = pred_delta.reshape(-1, pred_delta.shape[-1])[m]
        t_raw = target_delta.reshape(-1, target_delta.shape[-1])[m]
        scale = self.value_scale
        p = p_raw * scale
        t = t_raw * scale
        n_ch = int(p.shape[-1])
        cw = self.channel_weight
        if cw is None:
            ch_w = [1.0] * n_ch
        else:
            ch_w = [float(cw[i]) if i < len(cw) else 1.0 for i in range(n_ch)]

        losses: list[torch.Tensor] = []
        beta = self.beta
        for ch in range(n_ch):
            thr_raw = self._channel_thr(ch)
            active = t_raw[:, ch] > thr_raw
            if bool(active.any().item()):
                losses.append(
                    ch_w[ch]
                    * F.huber_loss(p[active, ch], t[active, ch], delta=beta, reduction="mean")
                )
                if self.underpred_weight > 0.0:
                    miss = active & (p_raw[:, ch] < 0.5 * t_raw[:, ch])
                    if bool(miss.any().item()):
                        losses.append(
                            ch_w[ch]
                            * self.underpred_weight
                            * F.mse_loss(p[miss, ch], t[miss, ch], reduction="mean")
                        )
            fp_thr = max(self.fp_threshold, self._channel_thr(ch))
            fp = (~active) & (p_raw[:, ch] > fp_thr)
            if (
                self.mature_exempt_fp
                and current_log_state is not None
                and self.mature_max_log is not None
                and ch < len(self.mature_max_log)
            ):
                st = current_log_state.reshape(-1, current_log_state.shape[-1])[m]
                mature = st[:, ch] >= (self.mature_frac * float(self.mature_max_log[ch]))
                fp = fp & (~mature)
            if bool(fp.any().item()):
                element_loss = F.huber_loss(
                    p[fp, ch],
                    torch.zeros_like(p[fp, ch]),
                    delta=beta,
                    reduction="none",
                )
                if fp_weight_scale is not None:
                    scale_m = fp_weight_scale.reshape(-1)[m]
                    scale_fp = scale_m[fp]
                    element_loss = element_loss * scale_fp
                losses.append(
                    ch_w[ch]
                    * self.fp_weight
                    * element_loss.mean()
                )
        if not losses:
            return pred_delta.sum() * 0.0
        return torch.stack(losses).mean()
