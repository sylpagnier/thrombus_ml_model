"""Shared PMGP-DEQ (Stage-A) checkpoint resolution and inference.

Flow model = ``GINO_DEQ`` class (canonical id ``pmgp_deq_kine`` / PMGP-DEQ).
Prefer ``load_pmgp_deq_kine`` / ``resolve_pmgp_deq_kine_ckpt``; legacy ``load_gino_deq_kine`` aliases retained.

Inference helpers share one DEQ solve per graph: UV predictions and ``z_kin`` are cached together
so pack-build / coupling paths never pay for a second Anderson solve on the same vessel.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data

from src.architecture.ginodeq import GINO_DEQ
from src.architecture.kinematics_model_config import (
    build_gino_deq_from_ctor,
    kinematics_checkpoint_tensors,
    resolve_gino_deq_ctor_kwargs,
)
from src.config import PhysicsConfig
from src.utils.paths import resolve_checkpoint

KINEMATICS_CKPT_CANDIDATES = (
    "kinematics_best.pth",
    "kinematics_ckpt_latest.pth",
    "kinematics_ckpt_100.pth",
)


def resolve_kinematics_checkpoint(explicit: Path | str | None = None) -> Path:
    """Return an existing kinematics checkpoint path (explicit or search candidates)."""
    if explicit is not None:
        path = Path(explicit)
        if path.is_file():
            return path.resolve()
        candidate = resolve_checkpoint("a", path.name)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Kinematics checkpoint not found: {explicit}")

    for ckpt_name in KINEMATICS_CKPT_CANDIDATES:
        candidate = resolve_checkpoint("a", ckpt_name)
        if candidate.exists():
            return candidate

    expected_dir = resolve_checkpoint("a", KINEMATICS_CKPT_CANDIDATES[0]).parent
    raise FileNotFoundError(
        "No kinematics checkpoint found. Tried: "
        + ", ".join(str(expected_dir / name) for name in KINEMATICS_CKPT_CANDIDATES)
    )


def _load_torch_checkpoint(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


# Session cache: eval/viz reload the same Stage-A ckpt many times; training should clear after pack build.
_KINE_MODEL_CACHE: dict[tuple[str, str, int], GINO_DEQ] = {}


def clear_kinematics_predictor_cache() -> None:
    """Drop cached GINO-DEQ handles (frees VRAM once callers drop their refs)."""
    _KINE_MODEL_CACHE.clear()


def load_kinematics_predictor(
    checkpoint: Path | str,
    device: torch.device | str,
    *,
    phys_cfg: PhysicsConfig | None = None,
    max_iters: int = 25,
    cache: bool = True,
) -> GINO_DEQ:
    """Load GINO-DEQ from a kinematics checkpoint with training-default architecture.

    When ``cache=True`` (default), the same ``(ckpt, device, max_iters)`` returns the same
    eval-mode module so multi-vessel eval/viz does not reload weights from disk each time.
    Training pack-build should pass ``cache=False`` or call :func:`clear_kinematics_predictor_cache`
    after features are baked, so VRAM can be released.
    """
    ckpt_path = Path(checkpoint).resolve()
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    cache_key = (str(ckpt_path), str(dev), max(3, int(max_iters)))
    if cache and cache_key in _KINE_MODEL_CACHE:
        return _KINE_MODEL_CACHE[cache_key]

    raw = _load_torch_checkpoint(ckpt_path)
    meta, state = kinematics_checkpoint_tensors(raw)
    ctor = resolve_gino_deq_ctor_kwargs(meta, state)
    ctor["max_iters"] = max(3, int(max_iters))
    cfg = phys_cfg or PhysicsConfig(phase="kinematics")
    model = build_gino_deq_from_ctor(cfg, ctor).to(dev)
    model.load_state_dict(state, strict=False)
    model.eval()
    if cache:
        _KINE_MODEL_CACHE[cache_key] = model
    return model


def _kin_solver_kwargs() -> dict[str, object]:
    solver = os.environ.get("VIZ_KIN_SOLVER", "anderson").strip().lower() or "anderson"
    beta = float(os.environ.get("VIZ_KIN_ANDERSON_BETA", "0.8"))
    warmup = int(os.environ.get("VIZ_KIN_ANDERSON_WARMUP", "5"))
    return {
        "solver": solver,
        "anderson_beta": beta,
        "anderson_warmup_iters": max(0, warmup),
    }


def _graph_key(data) -> tuple[int, int, int]:
    n = int(data.num_nodes)
    e = int(data.edge_index.shape[1])
    ptr = 0
    if hasattr(data, "x") and torch.is_tensor(data.x) and data.x.numel() > 0:
        ptr = int(data.x.untyped_storage().data_ptr())
    return (n, e, ptr)


def _cache_hit(model: GINO_DEQ, key: tuple[int, int, int]) -> bool:
    return getattr(model, "_cache_key", None) == key


def _store_joint_cache(model: GINO_DEQ, key: tuple[int, int, int], pred: torch.Tensor, z: torch.Tensor) -> None:
    model._cache_key = key
    model._cache_pred = pred
    model._cache_latent = z


def _run_joint_solve(model: GINO_DEQ, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
    """One Anderson solve on ``data``; returns ``(pred, z_kin)``."""
    orig_device = next(model.parameters()).device
    kwargs = _kin_solver_kwargs()
    if orig_device.type == "cuda":
        try:
            return model.predict_uv_and_latent(data.to(orig_device), **kwargs)
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            try:
                return model.predict_uv_and_latent(data.to(orig_device), **kwargs)
            except torch.cuda.OutOfMemoryError:
                raise RuntimeError(
                    "predict_kinematics_and_latent OOM on CUDA. Silent fallbacks to CPU are "
                    "disabled by Hardware Execution Policy to prevent hangs."
                ) from e
    return model.predict_uv_and_latent(data, **kwargs)


@torch.no_grad()
def predict_kinematics_and_latent(model: GINO_DEQ, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
    """One GINO-DEQ solve; returns ``(pred [N, C], z_kin [N, latent_dim])`` and fills both caches."""
    key = _graph_key(data)
    if (
        _cache_hit(model, key)
        and getattr(model, "_cache_pred", None) is not None
        and getattr(model, "_cache_latent", None) is not None
    ):
        return model._cache_pred, model._cache_latent

    pred, z = _run_joint_solve(model, data)
    _store_joint_cache(model, key, pred, z)
    return pred, z


def predict_kinematics(model: GINO_DEQ, data: Data) -> torch.Tensor:
    """Run one GINO-DEQ forward pass; returns ``(N, C)`` predictions (joint-caches ``z_kin``)."""
    key = _graph_key(data)
    if _cache_hit(model, key) and getattr(model, "_cache_pred", None) is not None:
        return model._cache_pred
    pred, _ = predict_kinematics_and_latent(model, data)
    return pred


@torch.no_grad()
def predict_kinematics_latent(model: GINO_DEQ, data: Data) -> torch.Tensor:
    """Frozen DEQ latent ``z_kin`` per node, shape ``[N, latent_dim]`` (joint-caches UV pred)."""
    key = _graph_key(data)
    if _cache_hit(model, key) and getattr(model, "_cache_latent", None) is not None:
        return model._cache_latent
    _, z = predict_kinematics_and_latent(model, data)
    return z


# Canonical names (PMGP-DEQ Stage-A flow)
resolve_pmgp_deq_kine_ckpt = resolve_kinematics_checkpoint
load_pmgp_deq_kine = load_kinematics_predictor
predict_pmgp_deq_flow = predict_kinematics
predict_pmgp_deq_latent = predict_kinematics_latent
predict_pmgp_deq_flow_and_latent = predict_kinematics_and_latent
# Legacy GINO-DEQ aliases
resolve_gino_deq_kine_ckpt = resolve_pmgp_deq_kine_ckpt
load_gino_deq_kine = load_pmgp_deq_kine
predict_gino_deq_flow = predict_pmgp_deq_flow
predict_gino_deq_latent = predict_pmgp_deq_latent
predict_gino_deq_flow_and_latent = predict_pmgp_deq_flow_and_latent
