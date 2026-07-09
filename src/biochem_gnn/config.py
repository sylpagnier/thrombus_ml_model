"""Canonical biochem deploy stack (hybrid SciML pipeline).

Architecture (locked 2026-06): see docs/MODEL_NOMENCLATURE.md
  pmgp_deq_kine (PMGP-DEQ) -> species_graphsage -> gelation_beta -> clot_trigger_physics
  -> [future] flow_coupling

Import path ``src.biochem_gnn`` is legacy; artifact dirs unchanged.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.model_nomenclature import (
    BIOCHEM_DEPLOY_STACK,
    CLOT_TRIGGER_PHYSICS,
    FLOW_COUPLING,
    GELATION_BETA,
    PMGP_DEQ_KINE,
    SPECIES_GRAPHSAGE,
)
from src.utils.paths import get_project_root

# --- identity (canonical SciML ids; legacy aliases still resolve) ---
STACK_NAME = BIOCHEM_DEPLOY_STACK.id
PHASE_TRAIN = BIOCHEM_DEPLOY_STACK.id
PHASE_CKPT = "biochem_gnn"
PHASE_MANIFEST = "biochem_gnn_baseline"
PHASE_LOAO_INDEX = "biochem_gnn_loao"

VAL_ANCHOR_DEFAULT = "patient007"
# 0 = per-graph last macro-step (full COMSOL timeline). See LEGACY_CAPPED_DEPLOY_HORIZON.
DEPLOY_HORIZON_DEFAULT = 0
LEGACY_CAPPED_DEPLOY_HORIZON = 53
DEPLOY_MAX_UNROLL_CAP = 200
S0_F1_TARGET_P007 = 0.408
GATE_F1_MIN_P007 = 0.65

# --- canonical output tree ---
ROOT_DIR = Path("outputs/biochem/biochem_gnn")
GLOBAL_DIR = ROOT_DIR / "species"
BETA_DIR = ROOT_DIR / "viscosity"
LOAO_DIR = ROOT_DIR / "loao"
LOCKED_DIR = ROOT_DIR / "locked"
STAGING_DIR = ROOT_DIR / "staging"
VIZ_DIR = Path("outputs/biochem/viz/biochem_gnn")

GLOBAL_CKPT = GLOBAL_DIR / "best.pth"
BETA_CKPT = BETA_DIR / "beta.pth"
LOCKED_GNN_CKPT = LOCKED_DIR / "species_gnn_best.pth"
LOCKED_BETA_CKPT = LOCKED_DIR / "viscosity_beta.pth"
LOCKED_MANIFEST = LOCKED_DIR / "manifest.json"

REFERENCE_JSON = Path("data/reference/biochem_gnn_baseline.json")
STAGING_REFERENCE = Path("data/reference/biochem_gnn_staging.json")
STAGING_MANIFEST = STAGING_DIR / "manifest.json"
STAGING_CKPT_PICK = STAGING_DIR / "ckpt_pick.json"
STAGING_LOAO_EVAL = STAGING_DIR / "loao_eval_gt.json"

DEFAULT_KINE_CKPT = Path("outputs/kinematics/kinematics_best.pth")
INIT_WARMSTART = Path("outputs/biochem/biochem_gnn/locked/species_gnn_best.pth")

# --- legacy path fallbacks (read / migration) ---
LEGACY_CLOT_DEPLOY_ROOT = Path("outputs/biochem/clot_deploy_gnn")
LEGACY_CLOT_GLOBAL = LEGACY_CLOT_DEPLOY_ROOT / "global" / "best.pth"
LEGACY_CLOT_BETA = LEGACY_CLOT_DEPLOY_ROOT / "beta" / "beta.pth"
LEGACY_CLOT_LOAO = LEGACY_CLOT_DEPLOY_ROOT / "loao"
LEGACY_CLOT_LOCKED = LEGACY_CLOT_DEPLOY_ROOT / "locked"
LEGACY_CLOT_REF = Path("data/reference/clot_deploy_gnn_baseline.json")
LEGACY_CLOT_STAGING = Path("data/reference/clot_deploy_gnn_staging.json")

LEGACY_GLOBAL = Path("outputs/biochem/biochem_gnn/locked/species_gnn_best.pth")
LEGACY_BETA = Path("outputs/biochem/species_snapshot_s35/beta.pth")
LEGACY_LOAO = Path("outputs/biochem/species_gnn_loao")
LEGACY_LOCKED = Path("outputs/biochem/species_gnn_deploy_baseline")
LEGACY_REFERENCE = Path("data/reference/species_gnn_deploy_baseline.json")
LEGACY_STAGING_MANIFEST = Path("data/reference/species_gnn_deploy_r4.json")
LEGACY_STAGING_EVAL = Path("outputs/biochem/species_gnn_deploy/loao_eval_final.json")
LEGACY_STAGING_PICK = Path("outputs/biochem/species_gnn_deploy/ckpt_pick.json")
LEGACY_VIZ = Path("outputs/biochem/viz/species_gnn_deploy")

# Component keys (canonical SciML ids + legacy aliases for manifests)
COMPONENT_PMGP_DEQ_KINE = PMGP_DEQ_KINE.id
COMPONENT_GINO_DEQ_KINE = COMPONENT_PMGP_DEQ_KINE  # legacy export name
COMPONENT_SPECIES_GNN = SPECIES_GRAPHSAGE.id
COMPONENT_SPECIES = SPECIES_GRAPHSAGE.legacy_ids[0]  # species_gnn
COMPONENT_GELATION_BETA = GELATION_BETA.id
COMPONENT_VISCOSITY = GELATION_BETA.legacy_ids[0]  # viscosity_beta
COMPONENT_CLOT_TRIGGER_PHYSICS = CLOT_TRIGGER_PHYSICS.id
COMPONENT_CLOT = CLOT_TRIGGER_PHYSICS.legacy_ids[0]  # clot_phi
COMPONENT_FLOW_COUPLING = FLOW_COUPLING.id
COMPONENT_FLOW = FLOW_COUPLING.id  # not implemented

GLOBAL_TRAIN_RECIPE: dict[str, str] = {
    "SPECIES_CONTINUOUS_GROWTH_ONLY_LOSS": "1",
    "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
    "SPECIES_CONTINUOUS_PHYSICS_READOUT": "0",
    "SPECIES_KIN_PER_VESSEL_NORM": "1",
    "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
    "SPECIES_CONTINUOUS_TIME_CONTEXT": "1",
    "SPECIES_CONTINUOUS_TIME_REF_S": "3000",
    "SPECIES_CONTINUOUS_TIME_FOURIER_FREQS": "8",
    "SPECIES_CONTINUOUS_MATURE_FP_EXEMPT": "1",
    "SPECIES_CONTINUOUS_MATURE_FRAC": "0.95",
    "SPECIES_CONTINUOUS_SATURATION_SCALE": "80",
    # Baseline temporal handling is explicit tau + Fourier context only.
    "SPECIES_CONTINUOUS_VEL_DECAY": "1",
    "SPECIES_CONTINUOUS_TEACHER_NOISE": "0.02",
    "SPECIES_CONTINUOUS_TEACHER_FP_FRAC": "0.08",
    "SPECIES_CONTINUOUS_TEACHER_BLUR": "0.25",
    "SPECIES_CONTINUOUS_TBPTT_TAIL": "5",
    "SPECIES_CONTINUOUS_CURRICULUM_UNROLL": "1",
    "SPECIES_CONTINUOUS_CLOSED_LOOP_INIT": "0.45",
    "SPECIES_CONTINUOUS_FINAL_STATE_WEIGHT": "0.35",
    "SPECIES_CONTINUOUS_FINAL_STATE_ALL_BAND": "1",
    "SPECIES_CONTINUOUS_SPEED_FP_WEIGHT": "4.0",
    "SPECIES_CONTINUOUS_DEPLOY_HORIZON": str(DEPLOY_HORIZON_DEFAULT),
    "SPECIES_PUSHFORWARD_UNROLL": "10",
    "SPECIES_PUSHFORWARD_MAX_UNROLL": str(DEPLOY_MAX_UNROLL_CAP),
    "SPECIES_PUSHFORWARD_TRAIN_T0_PER_VESSEL": "1",
    "SPECIES_CONTINUOUS_DEPLOY_EVAL_FULL": "1",
    "SPECIES_DEPLOY_HORIZON_ALL_PACKS": "1",
    "SPECIES_DEPLOY_HORIZON_AUX_CAP": "72",
    "SPECIES_CONTINUOUS_HUBER_BETA": "0.5",
    "SPECIES_CONTINUOUS_CHANNEL_WEIGHT_MAT": "4.0",
    "SPECIES_CONTINUOUS_DELTA_VALUE_SCALE": "150000",
    "SPECIES_CONTINUOUS_DELTA_THRESH": "5e-6",
    "SPECIES_CONTINUOUS_FP_WEIGHT": "8",
    "SPECIES_CONTINUOUS_FP_THRESH": "2e-5",
    "SPECIES_CONTINUOUS_SPATIAL_LOSS_WEIGHT": "1.0",
    # Train on COMSOL [u,v] for vel-decay; pred kine only at deploy eval (saves VRAM).
    "SPECIES_TRAIN_VEL_SOURCE": "gt",
    "SPECIES_ROLLOUT_DEPLOY_FAITHFUL": "1",
    "SPECIES_ROLLOUT_VEL_SOURCE": "kinematics",
    "SPECIES_ROLLOUT_PIN_OTHER": "rest",
    "SPECIES_ROLLOUT_IC_SOURCE": "resting",
    "SPECIES_SNAPSHOT_WALL_HOPS": "3",
    "CLOT_PHI_CEILING_HOPS": "3",
}

DEPLOY_INFERENCE_ENV: dict[str, str] = {
    "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
    "SPECIES_KIN_PER_VESSEL_NORM": "1",
    "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
    "SPECIES_CONTINUOUS_TIME_CONTEXT": "1",
    "SPECIES_CONTINUOUS_TIME_REF_S": "3000",
    "SPECIES_CONTINUOUS_TIME_FOURIER_FREQS": "8",
    "SPECIES_CONTINUOUS_VEL_DECAY": "1",
    "SPECIES_CONTINUOUS_MATURE_FP_EXEMPT": "1",
    "SPECIES_VISCOSITY_CALIB": "1",
    "SPECIES_VISCOSITY_BETA_MIN": "0.1",
    "SPECIES_VISCOSITY_BETA_MAX": "2.0",
    "T0_RUNG4_STEP": "species_gnn",
    "SPECIES_ROLLOUT_DEPLOY_FAITHFUL": "1",
    "SPECIES_ROLLOUT_VEL_SOURCE": "kinematics",
    "SPECIES_ROLLOUT_PIN_OTHER": "rest",
    "SPECIES_ROLLOUT_IC_SOURCE": "resting",
    "SPECIES_SNAPSHOT_WALL_HOPS": "3",
    "CLOT_PHI_CEILING_HOPS": "3",
}

CKPT_PHASE_ALIASES = frozenset({PHASE_CKPT, "clot_deploy_gnn", "species_gnn_deploy_baseline"})


def repo_root() -> Path:
    return get_project_root()


def _abs(path: Path | str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = repo_root() / p
    return p


def resolve_existing(*candidates: Path | str) -> Path:
    resolved = [_abs(c) for c in candidates]
    for p in resolved:
        if p.is_file() or p.is_dir():
            return p
    return resolved[0]


def global_ckpt_path() -> Path:
    return resolve_existing(GLOBAL_CKPT, LEGACY_CLOT_GLOBAL, LEGACY_GLOBAL)


def beta_ckpt_path() -> Path:
    return resolve_existing(BETA_CKPT, LEGACY_CLOT_BETA, LEGACY_BETA)


def loao_root_path() -> Path:
    return resolve_existing(LOAO_DIR, LEGACY_CLOT_LOAO, LEGACY_LOAO)


def locked_root_path() -> Path:
    return resolve_existing(LOCKED_DIR, LEGACY_CLOT_LOCKED, LEGACY_LOCKED)


def reference_manifest_path() -> Path:
    return resolve_existing(REFERENCE_JSON, LEGACY_CLOT_REF, LEGACY_REFERENCE)


def staging_manifest_path() -> Path:
    return resolve_existing(STAGING_REFERENCE, STAGING_MANIFEST, LEGACY_CLOT_STAGING, LEGACY_STAGING_MANIFEST)


def staging_loao_eval_path() -> Path:
    return resolve_existing(STAGING_LOAO_EVAL, LEGACY_STAGING_EVAL)


def staging_ckpt_pick_path() -> Path:
    return resolve_existing(STAGING_CKPT_PICK, LEGACY_STAGING_PICK)


def normalize_train_phase(phase: str) -> str:
    p = (phase or "").strip().lower()
    if p in (PHASE_TRAIN, "clot_deploy_gnn", "deploy_gnn", "clot_deploy"):
        return PHASE_CKPT
    return p


def checkpoint_phase_tag(train_phase: str) -> str:
    _ = train_phase
    return PHASE_CKPT


def is_biochem_gnn_checkpoint_phase(phase: str) -> bool:
    pl = (phase or "").lower()
    return (
        pl in CKPT_PHASE_ALIASES
        or "biochem_gnn" in pl
        or "clot_deploy_gnn" in pl
    )


# backward-compatible alias
is_deploy_gnn_checkpoint_phase = is_biochem_gnn_checkpoint_phase


def apply_train_recipe_env(*, overrides: dict[str, str] | None = None, force: bool = False) -> dict[str, str]:
    merged = dict(GLOBAL_TRAIN_RECIPE)
    if overrides:
        merged.update({k: str(v) for k, v in overrides.items()})
    for key, val in merged.items():
        if force or not str(os.environ.get(key, "")).strip():
            os.environ[key] = str(val)
    return merged


def rel_path(path: Path | str) -> str:
    p = _abs(path)
    try:
        return str(p.relative_to(repo_root())).replace("\\", "/")
    except ValueError:
        return str(p)


def default_manifest_payload() -> dict[str, Any]:
    locked = locked_root_path()
    return {
        "phase": PHASE_MANIFEST,
        "stack": STACK_NAME,
        "species_gnn_ckpt": rel_path(locked / "species_gnn_best.pth"),
        "viscosity_beta": rel_path(locked / "viscosity_beta.pth"),
        "kinematics_ckpt": rel_path(DEFAULT_KINE_CKPT),
        "loao_dir": rel_path(locked / "loao"),
        "train_val_anchor": VAL_ANCHOR_DEFAULT,
        "flow_modes": "gt,kinematics",
        "gamma_mode": "max",
        "deploy_horizon": "full",
        "components": {
            COMPONENT_PMGP_DEQ_KINE: "frozen_external",
            COMPONENT_SPECIES_GNN: "trained",
            COMPONENT_GELATION_BETA: "trained",
            COMPONENT_CLOT_TRIGGER_PHYSICS: "physics_nucleation",
            COMPONENT_FLOW_COUPLING: "external_kinematics",
        },
    }


def load_manifest(path: Path | str | None = None) -> dict[str, Any]:
    env_override = (
        os.environ.get("BIOCHEM_GNN_MANIFEST")
        or os.environ.get("CLOT_DEPLOY_GNN_MANIFEST")
        or os.environ.get("SPECIES_GNN_DEPLOY_MANIFEST")
        or ""
    ).strip()
    p = _abs(path) if path else (_abs(env_override) if env_override else reference_manifest_path())
    out: dict[str, Any] = dict(default_manifest_payload())
    if not p.is_file():
        return out
    raw = json.loads(p.read_text(encoding="utf-8-sig"))
    payload = dict(raw.get("baseline") or raw)
    for k, v in payload.items():
        if v is None:
            continue
        if k in ("ckpt_overrides", "beta_overrides") and isinstance(v, dict):
            out[k] = {str(a): str(c) for a, c in v.items()}
        elif k == "loao_preferred" and isinstance(v, list):
            out[k] = [str(a) for a in v]
        else:
            out[k] = v if isinstance(v, (dict, list)) else str(v)
    return out


def resolve_loao_ckpt(anchor: str, loao_dir: Path | str | None = None) -> Path:
    base = _abs(loao_dir) if loao_dir else loao_root_path()
    stem = anchor.strip().replace(".pt", "")
    return base / f"holdout_{stem}" / "best.pth"


def species_ckpt_for_anchor(
    anchor: str,
    manifest: dict[str, Any] | None = None,
    *,
    prefer_loao: bool = True,
) -> Path:
    m = manifest or load_manifest()
    stem = anchor.strip().replace(".pt", "")
    overrides = m.get("ckpt_overrides") or {}
    if isinstance(overrides, dict) and stem in overrides:
        return _abs(str(overrides[stem]))
    global_ckpt = _abs(str(m.get("species_gnn_ckpt", rel_path(LOCKED_GNN_CKPT))))
    if prefer_loao:
        loao = resolve_loao_ckpt(stem, m.get("loao_dir"))
        if loao.is_file():
            pref = m.get("loao_preferred") or []
            if isinstance(pref, list) and stem in pref:
                return loao
            auto = str(m.get("loao_auto", os.environ.get("BIOCHEM_GNN_LOAO_AUTO", ""))).strip().lower()
            if auto in ("1", "true", "yes", "on"):
                return loao
    return global_ckpt


def apply_deploy_env(
    manifest: dict[str, Any] | None = None,
    *,
    overrides: dict[str, str] | None = None,
    anchor: str | None = None,
    prefer_loao: bool = True,
) -> dict[str, str]:
    m = manifest or load_manifest()
    merged = dict(DEPLOY_INFERENCE_ENV)
    pre = {k: str(v) for k, v in (overrides or {}).items()}
    if "SPECIES_GNN_CLOUT_CKPT" in pre:
        merged["SPECIES_GNN_CLOUT_CKPT"] = pre["SPECIES_GNN_CLOUT_CKPT"]
    elif anchor:
        merged["SPECIES_GNN_CLOUT_CKPT"] = str(species_ckpt_for_anchor(anchor, m, prefer_loao=prefer_loao))
    else:
        merged["SPECIES_GNN_CLOUT_CKPT"] = str(m.get("species_gnn_ckpt", rel_path(LOCKED_GNN_CKPT)))
    merged["SPECIES_CONTINUOUS_CKPT"] = merged["SPECIES_GNN_CLOUT_CKPT"]
    merged["T0_R4_SPECIES_GNN_CKPT"] = merged["SPECIES_GNN_CLOUT_CKPT"]
    merged["SPECIES_VISCOSITY_CALIB_PATH"] = str(m.get("viscosity_beta", rel_path(LOCKED_BETA_CKPT)))
    beta_ov = m.get("beta_overrides") or {}
    if anchor and isinstance(beta_ov, dict):
        stem = anchor.strip().replace(".pt", "")
        if stem in beta_ov:
            merged["SPECIES_GELATION_BETA_OVERRIDE"] = str(beta_ov[stem])
    merged["KINEMATICS_CHECKPOINT"] = str(m.get("kinematics_ckpt", rel_path(DEFAULT_KINE_CKPT)))
    scope = m.get("species_scope") or m.get("pushforward_species_scope")
    channels = m.get("species_channels") or m.get("pushforward_species_channels")
    if channels:
        if isinstance(channels, (list, tuple)):
            merged["BIOCHEM_PUSHFORWARD_SPECIES_CHANNELS"] = ",".join(str(int(x)) for x in channels)
        else:
            merged["BIOCHEM_PUSHFORWARD_SPECIES_CHANNELS"] = str(channels)
    elif scope:
        merged["BIOCHEM_PUSHFORWARD_SPECIES_SCOPE"] = str(scope)
    if m.get("loao_auto") is not None:
        merged["BIOCHEM_GNN_LOAO_AUTO"] = str(m.get("loao_auto"))
    merged.update(pre)
    for k, v in merged.items():
        if k in os.environ and k not in (overrides or {}) and k not in (
            "SPECIES_GNN_CLOUT_CKPT",
            "SPECIES_CONTINUOUS_CKPT",
            "T0_R4_SPECIES_GNN_CKPT",
            "SPECIES_VISCOSITY_CALIB_PATH",
            "KINEMATICS_CHECKPOINT",
        ):
            continue
        os.environ[k] = str(v)
    return merged

