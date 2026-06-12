"""Canonical biochem loss policy (frozen-kin teacher, viscosity + species goal).

See docs/BIOCHEM_TRAINING_PROGRESS.md section "Loss policy (approved vs deprecated)".

Set ``BIOCHEM_LEGACY_LOSSES=1`` to re-enable deprecated isolate keys, sweep presets, and
aux-loss blocks removed from the default forward path.
"""
from __future__ import annotations

import os
from typing import FrozenSet

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
    # Unknown keys still handled by train_biochem_corrector valid list.


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
