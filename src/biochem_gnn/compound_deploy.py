"""Canonical compound deploy (WC_v7 wall + lumen growth specialist)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from src.utils.paths import get_project_root

COMPOUND_LEG = "WC_v8_compound_front_h1"
COMPOUND_LABEL = "WC v8 compound: WC_v7 wall + frontier-h1 lumen specialist"
REFERENCE_JSON = "data/reference/mat_compound_deploy.json"
LOCKED_GROWTH_CKPT = "outputs/biochem/biochem_gnn/locked/compound_growth_best.pth"
LOCKED_WALL_CKPT = "outputs/biochem/biochem_gnn/locked/species_gnn_best.pth"
WALL_LEG = "WC_v7_clot_phi_mse"

DEFAULT_COMPOUND_DEPLOY_ENV: dict[str, str] = {
    "SPECIES_TWO_MODEL_MODE": "1",
    "SPECIES_TWO_MODEL_ROUTE": "frontier",
    "SPECIES_TWO_MODEL_FRONTIER_HOPS": "1",
    "SPECIES_CONTINUOUS_VEL_DECAY": "1",
    "SPECIES_CONTINUOUS_VEL_DECAY_WALL_ONLY": "1",
}

DEFAULT_COMPOUND_RUNTIME_KWARGS: dict = {
    "two_model_mode": True,
    "two_model_route": "frontier",
    "two_model_frontier_hops": 1,
}

DEFAULT_COMPOUND_PUSHFORWARD_KWARGS: dict = {
    "vel_decay": True,
    "vel_decay_wall_only": True,
}


def compound_reference_path() -> Path:
    return get_project_root() / REFERENCE_JSON


def load_compound_manifest() -> dict:
    path = compound_reference_path()
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_growth_ckpt() -> Path:
    root = get_project_root()
    man = load_compound_manifest()
    rel = man.get("growth_ckpt") or LOCKED_GROWTH_CKPT
    return root / str(rel)


def resolve_wall_ckpt() -> Path:
    root = get_project_root()
    man = load_compound_manifest()
    rel = man.get("wall_ckpt") or LOCKED_WALL_CKPT
    return root / str(rel)


def build_compound_typed_configs() -> tuple[object, object]:
    """Typed PushforwardConfig + BiochemRuntimeConfig for compound deploy."""
    from dataclasses import replace

    from src.architecture.pushforward_config import PushforwardConfig, split_legacy_env_overrides
    from src.architecture.runtime_config import BiochemRuntimeConfig, split_legacy_runtime_env

    man = load_compound_manifest()
    deploy = dict(DEFAULT_COMPOUND_DEPLOY_ENV)
    deploy.update(man.get("deploy") or {})
    pf_kw, rem = split_legacy_env_overrides(deploy)
    rt_kw, _ = split_legacy_runtime_env(rem)
    pf_kw = {**DEFAULT_COMPOUND_PUSHFORWARD_KWARGS, **pf_kw}
    rt_kw = {**DEFAULT_COMPOUND_RUNTIME_KWARGS, **rt_kw}
    growth = resolve_growth_ckpt()
    if growth.is_file():
        rt_kw["offwall_model_ckpt"] = str(growth).replace("\\", "/")
        rt_kw["two_model_mode"] = True
    pf = replace(PushforwardConfig(), **pf_kw) if pf_kw else PushforwardConfig()
    rt = BiochemRuntimeConfig.from_kwargs(rt_kw)
    return pf, rt


def apply_compound_deploy_env(*, force: bool = False) -> dict[str, str]:
    """Pin typed compound deploy config (+ IO path for offwall ckpt if needed)."""
    from src.biochem_gnn.config import _IO_ENV_KEYS, _bind_typed_configs

    pf, rt = build_compound_typed_configs()
    _bind_typed_configs(pf, rt)

    applied: dict[str, str] = {
        "two_model_mode": "1" if rt.offwall.two_model_mode else "0",
        "two_model_route": str(rt.offwall.two_model_route),
        "two_model_frontier_hops": str(rt.offwall.two_model_frontier_hops),
        "vel_decay": "1" if pf.vel_decay else "0",
        "vel_decay_wall_only": "1" if pf.vel_decay_wall_only else "0",
    }
    growth = resolve_growth_ckpt()
    if growth.is_file():
        path_s = str(growth).replace("\\", "/")
        # Process/IO: some loaders still resolve the offwall ckpt path from env.
        if force or "SPECIES_OFFWALL_MODEL_CKPT" not in os.environ:
            os.environ["SPECIES_OFFWALL_MODEL_CKPT"] = path_s
        applied["SPECIES_OFFWALL_MODEL_CKPT"] = path_s
    _ = _IO_ENV_KEYS  # documented: control-plane keys are typed-only
    return applied
