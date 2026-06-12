"""Shared T0 isolation-ladder rung settings (patient007-tuned proxy gamma)."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Iterator

import torch

from src.core_physics.clot_temporal_growth_rules import reset_temporal_kinematics_cache
from src.core_physics.t0_clot_predictor import t0_gt_baseline_env

# Rung 2/3: GT or pred flow + GT species + proxy gamma (no COMSOL spf.sr sidecar).
RUNG2_GAMMA_MODE = "max"
RUNG2_GAMMA_SCALE = 1.0
RUNG2_POISEUILLE_SCALE = 0.75

DEFAULT_KINE_CKPT = "outputs/kinematics/kinematics_best.pth"
DEFAULT_SPECIES_DUMP_DIR = "outputs/biochem/anchors_teacher_species"


def _subsample_graph_time(data, stride: int):
    """Return (data_sub, full_idx) where full_idx maps full timeline -> subsampled rows."""
    stride = max(int(stride), 1)
    if stride <= 1 or not hasattr(data, "y") or data.y.dim() != 3:
        return data, None
    total = int(data.y.shape[0])
    idx = list(range(0, total, stride))
    if idx[-1] != total - 1:
        idx.append(total - 1)
    full_idx = torch.tensor(idx, dtype=torch.long)
    sub = data.clone() if hasattr(data, "clone") else data
    sub.y = data.y.index_select(0, full_idx).contiguous()
    if hasattr(data, "t") and getattr(data, "t") is not None and torch.is_tensor(data.t):
        try:
            sub.t = data.t.index_select(0, full_idx).contiguous()
        except Exception:
            pass
    return sub, full_idx


def _expand_series_to_full_time(series: torch.Tensor, full_idx: torch.Tensor, n_full: int) -> torch.Tensor:
    """Nearest-neighbor upsample subsampled teacher output back to full macro grid."""
    out = torch.empty((n_full, *series.shape[1:]), device=series.device, dtype=series.dtype)
    idx_cpu = full_idx.detach().cpu().tolist()
    for t_full in range(n_full):
        j = min(range(len(idx_cpu)), key=lambda k: abs(idx_cpu[k] - t_full))
        out[t_full] = series[j]
    return out

_TEACHER_ENV_KEYS = (
    "BIOCHEM_GT_KINE_VEL",
    "BIOCHEM_GT_KINE_SKIP_DEQ",
    "BIOCHEM_TEACHER_MU_RATIO_MAX",
    "BIOCHEM_VAL_TIME_STRIDE",
    "BIOCHEM_DATALOADER_WORKERS",
)

_FLOW_ENV_KEYS = (
    "CLOT_PHI_VEL_SOURCE",
    "CLOT_TEMPORAL_VEL_SOURCE",
    "CLOT_PHI_KINE_CKPT",
    "CLOT_PHI_KINE_TF",
)


@contextlib.contextmanager
def t0_rung2_env(*, hard_step: bool = True) -> Iterator[dict[str, str]]:
    """Rung 2: GT flow + GT species + proxy gamma."""
    with t0_gt_baseline_env(
        gamma_mode=RUNG2_GAMMA_MODE,
        gamma_scale=RUNG2_GAMMA_SCALE,
        poiseuille_scale=RUNG2_POISEUILLE_SCALE,
        hard_step=hard_step,
    ) as cfg:
        yield {**cfg, "flow_source": "gt", "rung": "2"}


@contextlib.contextmanager
def t0_rung3_env(
    *,
    kine_ckpt: str = DEFAULT_KINE_CKPT,
    hard_step: bool = True,
) -> Iterator[dict[str, str]]:
    """Rung 3: pred GINO-DEQ flow + GT species + proxy gamma."""
    saved = {k: os.environ.get(k) for k in _FLOW_ENV_KEYS}
    reset_temporal_kinematics_cache()
    with t0_gt_baseline_env(
        gamma_mode=RUNG2_GAMMA_MODE,
        gamma_scale=RUNG2_GAMMA_SCALE,
        poiseuille_scale=RUNG2_POISEUILLE_SCALE,
        hard_step=hard_step,
    ) as cfg:
        os.environ["CLOT_PHI_VEL_SOURCE"] = "kinematics"
        os.environ["CLOT_TEMPORAL_VEL_SOURCE"] = "kinematics"
        os.environ["CLOT_PHI_KINE_CKPT"] = str(kine_ckpt)
        os.environ["CLOT_PHI_KINE_TF"] = "0"
        out = {
            **cfg,
            "flow_source": "kinematics",
            "kine_ckpt": str(kine_ckpt),
            "rung": "3",
        }
        try:
            yield out
        finally:
            reset_temporal_kinematics_cache()
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def resolve_default_teacher_ckpt() -> str:
    """First existing biochem teacher checkpoint (best_high_mu, last, sweep)."""
    from src.training.clot_trigger_stack import default_teacher_checkpoint_path

    return str(default_teacher_checkpoint_path())


@contextlib.contextmanager
def t0_teacher_gt_kine_env() -> Iterator[None]:
    """GNODE species rollout on GT COMSOL [u,v,p] (Rung 4 teacher path)."""
    saved = {k: os.environ.get(k) for k in _TEACHER_ENV_KEYS}
    os.environ["BIOCHEM_GT_KINE_VEL"] = "1"
    os.environ["BIOCHEM_GT_KINE_SKIP_DEQ"] = "1"
    os.environ.setdefault("BIOCHEM_VAL_TIME_STRIDE", "1")
    os.environ.setdefault("BIOCHEM_DATALOADER_WORKERS", "0")
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextlib.contextmanager
def t0_rung4_env(
    *,
    teacher_ckpt: str | None = None,
    hard_step: bool = True,
) -> Iterator[dict[str, str]]:
    """Rung 4: GT flow + pred teacher species + proxy gamma."""
    ckpt = str(teacher_ckpt or resolve_default_teacher_ckpt())
    with t0_rung2_env(hard_step=hard_step) as cfg, t0_teacher_gt_kine_env():
        yield {
            **cfg,
            "species_source": "teacher",
            "teacher_ckpt": ckpt,
            "rung": "4",
        }


@torch.no_grad()
def rollout_t0_pred_species_series(
    data,
    teacher_ckpt: str,
    device: torch.device,
    *,
    bio_cfg=None,
    dumped_graph: str | Path | None = None,
    time_stride: int = 1,
) -> torch.Tensor:
    """Pred species for Rung 4: live GNODE rollout on GT flow or cached dump graph."""
    from src.config import BiochemConfig
    from src.inference.biochem_teacher_loader import load_biochem_teacher_checkpoint
    from src.utils.nondim import to_t_nd

    if dumped_graph is not None:
        path = Path(dumped_graph)
        if path.is_file():
            dumped = torch.load(path, map_location=device, weights_only=False)
            n_full = int(data.y.shape[0])
            n_dump = int(dumped.y.shape[0])
            sp_dump = dumped.y[:, :, 4:16].to(device=device, dtype=data.y.dtype)
            out = data.y.clone().to(device=device)
            if n_dump == n_full:
                out[:, :, 4:16] = sp_dump
            else:
                for t_full in range(n_full):
                    t_dump = min(
                        int(round(t_full * (n_dump - 1) / max(n_full - 1, 1))),
                        n_dump - 1,
                    )
                    out[t_full, :, 4:16] = sp_dump[t_dump]
            return out

    bio = bio_cfg or BiochemConfig(phase="biochem")
    n_full = int(data.y.shape[0])
    batch_sub, full_idx = _subsample_graph_time(data, time_stride)
    batch = batch_sub.to(device)
    eval_times = to_t_nd(bio.resolve_biochem_times(batch, device), bio.t_final)
    with t0_teacher_gt_kine_env():
        teacher, _, _ = load_biochem_teacher_checkpoint(
            teacher_ckpt,
            device,
            pred_kine=False,
            quiet=True,
        )
        pred = teacher(
            batch,
            eval_times,
            y_true_trajectory=batch.y,
            teacher_forcing_ratio=0.0,
            detach_macro_state=True,
        )
    if isinstance(pred, tuple):
        pred = pred[0]
    if full_idx is not None:
        pred = _expand_series_to_full_time(pred, full_idx.to(device=pred.device), n_full)
    return pred


@contextlib.contextmanager
def t0_rung4_step_env(
    *,
    step: str = "s0",
    hard_step: bool = True,
) -> Iterator[dict[str, str]]:
    """Rung 4 mini-ladder step: deploy rules species -> gelation physics -> clot."""
    saved = os.environ.get("T0_RUNG4_STEP")
    os.environ["T0_RUNG4_STEP"] = str(step)
    with t0_rung2_env(hard_step=hard_step) as cfg:
        try:
            yield {
                **cfg,
                "species_source": "rules",
                "rung4_step": str(step),
                "rung": "4",
            }
        finally:
            if saved is None:
                os.environ.pop("T0_RUNG4_STEP", None)
            else:
                os.environ["T0_RUNG4_STEP"] = saved


@contextlib.contextmanager
def t0_rung41_env(
    *,
    rules_mode: str = "s0",
    hard_step: bool = True,
) -> Iterator[dict[str, str]]:
    """Backward-compatible alias for ``t0_rung4_step_env`` (legacy name R4.1)."""
    with t0_rung4_step_env(step=rules_mode, hard_step=hard_step) as cfg:
        yield {**cfg, "rules_mode": str(rules_mode), "rung": "4"}
