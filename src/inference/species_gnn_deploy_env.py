"""Deploy env + manifest for Rung 4 species GNN (s34 + s35 viscosity beta)."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from src.utils.paths import get_project_root

DEFAULT_SPECIES_GNN_CKPT = "outputs/biochem/species_snapshot_s34/best.pth"
DEFAULT_VISCOSITY_BETA = "outputs/biochem/species_snapshot_s35/beta.pth"
DEFAULT_KINE_CKPT = "outputs/kinematics/kinematics_best.pth"
DEFAULT_MANIFEST = "data/reference/species_gnn_deploy_r4.json"

# s34 continuous recipe (must match training launcher for checkpoint meta restore).
SPECIES_GNN_DEPLOY_ENV: dict[str, str] = {
    "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
    "SPECIES_KIN_PER_VESSEL_NORM": "1",
    "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
    "SPECIES_CONTINUOUS_TEMPORAL_GATE": "1",
    "SPECIES_CONTINUOUS_VEL_DECAY": "1",
    "SPECIES_CONTINUOUS_MATURE_FP_EXEMPT": "1",
    "SPECIES_VISCOSITY_CALIB": "1",
    "SPECIES_VISCOSITY_BETA_MIN": "0.1",
    "SPECIES_VISCOSITY_BETA_MAX": "2.0",
    "T0_RUNG4_STEP": "species_gnn",
}


def _resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = get_project_root() / p
    return p


def default_deploy_manifest() -> dict[str, str]:
    return {
        "phase": "species_gnn_deploy_r4",
        "species_gnn_ckpt": DEFAULT_SPECIES_GNN_CKPT,
        "viscosity_beta": DEFAULT_VISCOSITY_BETA,
        "kinematics_ckpt": DEFAULT_KINE_CKPT,
        "loao_dir": "outputs/biochem/species_gnn_loao",
        "train_val_anchor": "patient007",
        "flow_modes": "gt,kinematics",
        "gamma_mode": "max",
    }


def resolve_loao_ckpt_for_anchor(anchor: str, loao_dir: str | Path) -> Path:
    """Checkpoint from fold that held out ``anchor`` (never trained on it)."""
    root = get_project_root()
    base = Path(loao_dir)
    if not base.is_absolute():
        base = root / base
    stem = anchor.strip().replace(".pt", "")
    ckpt = base / f"holdout_{stem}" / "best.pth"
    return ckpt


def load_deploy_manifest(path: str | Path | None = None) -> dict[str, str]:
    p = _resolve(path or os.environ.get("SPECIES_GNN_DEPLOY_MANIFEST") or DEFAULT_MANIFEST)
    if p.is_file():
        raw = json.loads(p.read_text(encoding="utf-8"))
        out = default_deploy_manifest()
        out.update({k: str(v) for k, v in raw.items() if v is not None})
        return out
    return default_deploy_manifest()


def species_ckpt_for_anchor(
    anchor: str,
    manifest: dict[str, str] | None = None,
    *,
    prefer_loao: bool = True,
) -> Path:
    """Resolve species GNN ckpt: LOAO holdout fold when available, else global."""
    m = dict(manifest or load_deploy_manifest())
    if prefer_loao:
        loao = resolve_loao_ckpt_for_anchor(anchor, m.get("loao_dir", ""))
        if loao.is_file():
            return loao
    p = _resolve(str(m.get("species_gnn_ckpt", DEFAULT_SPECIES_GNN_CKPT)))
    return p


def apply_species_gnn_deploy_env(
    manifest: dict[str, str] | None = None,
    *,
    overrides: dict[str, str] | None = None,
    anchor: str | None = None,
    prefer_loao: bool = True,
) -> dict[str, str]:
    """Set process env for species GNN Rung 4 deploy; returns merged env dict."""
    m = dict(manifest or load_deploy_manifest())
    merged = dict(SPECIES_GNN_DEPLOY_ENV)
    if anchor and prefer_loao:
        ckpt = species_ckpt_for_anchor(anchor, m, prefer_loao=True)
        merged["SPECIES_GNN_CLOUT_CKPT"] = str(ckpt)
    else:
        merged["SPECIES_GNN_CLOUT_CKPT"] = str(m.get("species_gnn_ckpt", DEFAULT_SPECIES_GNN_CKPT))
    merged["SPECIES_CONTINUOUS_CKPT"] = merged["SPECIES_GNN_CLOUT_CKPT"]
    merged["T0_R4_SPECIES_GNN_CKPT"] = merged["SPECIES_GNN_CLOUT_CKPT"]
    merged["SPECIES_VISCOSITY_CALIB_PATH"] = str(
        m.get("viscosity_beta", DEFAULT_VISCOSITY_BETA)
    )
    merged["KINEMATICS_CHECKPOINT"] = str(m.get("kinematics_ckpt", DEFAULT_KINE_CKPT))
    if overrides:
        merged.update({k: str(v) for k, v in overrides.items()})
    for k, v in merged.items():
        os.environ[k] = str(v)
    return merged


@contextmanager
def species_gnn_deploy_env(
    manifest: dict[str, str] | None = None,
    *,
    overrides: dict[str, str] | None = None,
    anchor: str | None = None,
    prefer_loao: bool = True,
) -> Iterator[dict[str, str]]:
    saved = {k: os.environ.get(k) for k in SPECIES_GNN_DEPLOY_ENV}
    extra_keys = (
        "SPECIES_GNN_CLOUT_CKPT",
        "SPECIES_CONTINUOUS_CKPT",
        "T0_R4_SPECIES_GNN_CKPT",
        "SPECIES_VISCOSITY_CALIB_PATH",
        "KINEMATICS_CHECKPOINT",
    )
    for k in extra_keys:
        saved[k] = os.environ.get(k)
    try:
        yield apply_species_gnn_deploy_env(
            manifest, overrides=overrides, anchor=anchor, prefer_loao=prefer_loao,
        )
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def write_default_manifest(path: str | Path | None = None) -> Path:
    p = _resolve(path or DEFAULT_MANIFEST)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(default_deploy_manifest(), indent=2), encoding="utf-8")
    return p
