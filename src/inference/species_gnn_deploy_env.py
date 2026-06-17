"""Deploy env + manifest for clot deploy GNN (canonical wrapper).

Prefer ``src.biochem_gnn.config`` for new code. This module keeps backward-compatible
names (SPECIES_GNN_DEPLOY_*, species_gnn_deploy_baseline paths).
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from src.biochem_gnn import config as cdn
from src.utils.paths import get_project_root

# Backward-compatible aliases
DEFAULT_SPECIES_GNN_CKPT = cdn.rel_path(cdn.LOCKED_GNN_CKPT)
DEFAULT_VISCOSITY_BETA = cdn.rel_path(cdn.LOCKED_BETA_CKPT)
DEFAULT_KINE_CKPT = cdn.rel_path(cdn.DEFAULT_KINE_CKPT)
DEFAULT_MANIFEST = cdn.rel_path(cdn.REFERENCE_JSON)
BASELINE_DIR_NAME = cdn.LOCKED_DIR.name
SPECIES_GNN_DEPLOY_ENV = cdn.DEPLOY_INFERENCE_ENV


def baseline_dir() -> Path:
    d = cdn.locked_root_path()
    d.mkdir(parents=True, exist_ok=True)
    return d


def baseline_manifest_path() -> Path:
    return baseline_dir() / "manifest.json"


def default_deploy_manifest() -> dict[str, Any]:
    return cdn.default_manifest_payload()


def load_deploy_manifest(path: str | Path | None = None) -> dict[str, Any]:
    return cdn.load_manifest(path)


def resolve_loao_ckpt_for_anchor(anchor: str, loao_dir: str | Path) -> Path:
    return cdn.resolve_loao_ckpt(anchor, loao_dir)


def species_ckpt_for_anchor(
    anchor: str,
    manifest: dict[str, str] | None = None,
    *,
    prefer_loao: bool = True,
) -> Path:
    return cdn.species_ckpt_for_anchor(anchor, manifest, prefer_loao=prefer_loao)


def apply_species_gnn_deploy_env(
    manifest: dict[str, str] | None = None,
    *,
    overrides: dict[str, str] | None = None,
    anchor: str | None = None,
    prefer_loao: bool = True,
) -> dict[str, str]:
    return cdn.apply_deploy_env(manifest, overrides=overrides, anchor=anchor, prefer_loao=prefer_loao)


@contextmanager
def species_gnn_deploy_env(
    manifest: dict[str, str] | None = None,
    *,
    overrides: dict[str, str] | None = None,
    anchor: str | None = None,
    prefer_loao: bool = True,
) -> Iterator[dict[str, str]]:
    keys = set(SPECIES_GNN_DEPLOY_ENV) | {
        "SPECIES_GNN_CLOUT_CKPT",
        "SPECIES_CONTINUOUS_CKPT",
        "T0_R4_SPECIES_GNN_CKPT",
        "SPECIES_VISCOSITY_CALIB_PATH",
        "SPECIES_GELATION_BETA_OVERRIDE",
        "KINEMATICS_CHECKPOINT",
    }
    saved = {k: os.environ.get(k) for k in keys}
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
    p = Path(path or DEFAULT_MANIFEST)
    if not p.is_absolute():
        p = get_project_root() / p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(default_deploy_manifest(), indent=2), encoding="utf-8")
    return p
