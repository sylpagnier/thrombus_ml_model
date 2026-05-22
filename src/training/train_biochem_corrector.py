"""Biochem corrector training (GNODE phase 3).

Default training env (when keys are unset) is tuned for **fast iteration** plus **supervised step 2**
(still no Kendall PDE sum in ``backward()``). ``_apply_pycharm_biochem_optimal_defaults`` sets
``BIOCHEM_LOSS_DATA_ONLY=1``, **μ SI anchor** (``BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT``; optional **multi-step**
Huber via ``BIOCHEM_MU_SI_MULTI_STEP``, default on), optional **val-aligned log-MAE** on SI μ
(``BIOCHEM_MU_LOG_ANCHOR_WEIGHT``), **TBPTT cap 12**
with ``BIOCHEM_TBPTT_WINDOW_CURRICULUM=1`` epoch ramp, **shorter AE / ODE-RXN / teacher** epoch budgets,
**ODE-RXN patience ignores early epochs** (``BIOCHEM_ODE_RXN_MIN_EPOCHS_BEFORE_PLATEAU``), **less frequent teacher validation**, and ``BIOCHEM_COMPLEXITY_STEP=2`` for clarity in logs.

Kendall / physics / aux stay in ``metrics`` only unless multitask loss is enabled. It **removes
``BIOCHEM_MAX_LOAD_VESSELS``** if set so one-graph IDE caps do not apply. For a slow, conservative
reproduction run, set ``BIOCHEM_STOCK_DEFAULTS=1`` and configure env explicitly, or override individual
knobs (e.g. raise ``BIOCHEM_TEACHER_EPOCHS``, ``BIOCHEM_AE_EPOCHS``, ``BIOCHEM_DEBUG=1``).

- **Overnight same-tier run** (longer schedules + diagnostics, still data-only step 2):
  ``BIOCHEM_PRESET=overnight_step2`` (aliases ``comprehensive_step2``, ``step2_overnight``). Adds
  ``BIOCHEM_DATA_ONLY_PHYS_TEMP=1`` so ``w_pt * L_PhysTemp`` joins the anchor backprop objective
  (COMSOL temporal Huber; weight ``BIOCHEM_COMSOL_TEMPORAL_WEIGHT``).
  On **Windows PowerShell**, set it with ``$env:BIOCHEM_PRESET='overnight_step2'`` before ``python``;
  ``set VAR=value`` is **cmd.exe** syntax and does not set process env vars for ``python`` from PowerShell.

- **Compact step 2.5** (same budgets as fast defaults, add temporal anchor loss only):
  ``BIOCHEM_PRESET=step2p5`` (aliases ``phys_temp_only``, ``compact_step2p5``). Sets
  ``BIOCHEM_DATA_ONLY_PHYS_TEMP=1`` and default ``BIOCHEM_COMSOL_TEMPORAL_WEIGHT`` when unset —
  best next step when step-2 μ is flat but you are not ready for full Kendall backprop.

- **Thrombus corona (experimental / unvalidated)**:
  ``BIOCHEM_PRESET=thrombus_corona`` (aliases ``corona_thrombus``, ``biochem_thrombus_corona``).
  Bundles gelation prior gate, **3-hop** prior dilation, ``L_PhysTemp``, and corrector
  (``BIOCHEM_STOP_AFTER_TEACHER=0``). **Not recommended** for current μ work — one logged run
  (2026-05-16) had flat teacher μ (~1.48) with confounded settings; not rerun post-A0.
  See ``scripts/run_biochem_thrombus_corona.ps1`` and ``src/docs/BIOCHEM_TRAINING_PROGRESS.md``.

- **Comprehensive μ (experimental / unvalidated)**:
  ``BIOCHEM_PRESET=comprehensive_mu`` (aliases ``mu_comprehensive``, ``step2_comprehensive``).
  Corona bundle + longer schedules + μ best-practice env. Same validation status as corona.
  See ``scripts/run_biochem_comprehensive_mu.ps1``.

- **Teacher viscosity baseline (recommended new base for incremental loss studies)**:
  ``BIOCHEM_PRESET=teacher_visc_baseline`` (aliases ``visc_teacher_baseline``,
  ``teacher_mu_base``). Keeps teacher-only step-2 data-loss optimization while emphasizing
  all-time SI log-μ fit (global + wall + high-μ_gt tail) and preserving flow/species via
  ``L_Data_Kine`` + ``L_Data_Bio``.

- **Teacher max-complexity (recommended for robust viscosity teacher runs)**:
  ``BIOCHEM_PRESET=teacher_max_complexity`` (aliases ``max_complexity_teacher``,
  ``teacher_step3_robust``). Forces **complexity step 3** (full multitask backward),
  clears single-term isolate state, keeps ``BIOCHEM_STOP_AFTER_TEACHER=1``, and
  enables μ-focused defaults (`MU_LOG` + `MU_SI` anchors, μ-path, high μ cap).
  Use this when your goal is robust teacher-only viscosity fitting before re-enabling
  the corrector.

- **Complexity step 3** (full multitask / Kendall + PDE in ``backward``): set
  ``BIOCHEM_COMPLEXITY_STEP=3`` (aliases ``phase3``, ``full_multitask``, ``corrector_full``). Applies
  after the step-2 helper and forces ``BIOCHEM_LOSS_DATA_ONLY=0``; use when VRAM and adjoint are stable.

- Full multi-task loss: ``BIOCHEM_LOSS_DATA_ONLY=0`` (or unset after clearing env), or
  ``BIOCHEM_STOCK_DEFAULTS=1`` to skip all set-if-missing defaults (and the ``MAX_LOAD`` pop).
- Clear ``BIOCHEM_LOSS_ISOLATE`` in the Run Configuration if you still have a single-term diagnostic set.
- Re-enable the older teacher-stage ``setdefault`` block: ``BIOCHEM_NO_TEACHER_DEFAULTS=0``.

``BIOCHEM_TEACHER_TF_WARMUP_EPOCHS`` is clamped to ``BIOCHEM_TEACHER_EPOCHS`` in the teacher loop; values
greater than teacher length only affect readability. Match them for short teacher runs.

TBPTT: ``BIOCHEM_DETACH_MACRO_STATE=0`` keeps gradients through the rollout; set **1** only to cap
adjoint VRAM (expect much slower loss motion).

PyTorch 2.x may warn once per process that sparse COO invariant checks are off; training calls
``torch.sparse.check_sparse_tensor_invariants.disable()`` by default (and in DataLoader workers).
Set ``BIOCHEM_SPARSE_INVARIANT_CHECKS=1`` to enable slow checks when debugging corrupt sparse graphs.
"""
import argparse
import atexit
import os
import sys
import math
import time
import copy

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

if sys.platform != "win32":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
else:
    # ``expandable_segments`` is unsupported on some Windows PyTorch builds (warning + no-op).
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch_geometric.loader import DataLoader


def resolve_training_device() -> torch.device:
    """Pick compute device. Honors ``BIOCHEM_DEVICE=auto|cuda|cpu`` (default ``auto``).

    Plain ``pip install torch`` is often CPU-only; use ``scripts/install_torch_cuda.ps1``
    or install from https://pytorch.org so ``torch.cuda.is_available()`` is True.
    """
    want = (os.environ.get("BIOCHEM_DEVICE") or "auto").strip().lower()
    cuda_ok = torch.cuda.is_available()

    if want in ("cuda", "gpu"):
        if not cuda_ok:
            print(
                "BIOCHEM_DEVICE=cuda but this PyTorch build has no CUDA "
                "(torch.cuda.is_available() is False).\n"
                "Install a GPU wheel, e.g. run: .\\scripts\\install_torch_cuda.ps1\n"
                "or: py -3 -m pip install torch --upgrade "
                "--index-url https://download.pytorch.org/whl/cu124\n"
                "Verify: py -3 -c \"import torch; print(torch.cuda.is_available())\""
            )
            sys.exit(1)
        dev = torch.device("cuda:0")
        torch.cuda.set_device(0)
        return dev

    if want == "cpu":
        return torch.device("cpu")

    if want != "auto":
        print(f"Unknown BIOCHEM_DEVICE={want!r}; use auto, cuda, or cpu.", file=sys.stderr)
        sys.exit(1)

    if cuda_ok:
        dev = torch.device("cuda:0")
        torch.cuda.set_device(0)
        return dev

    if torch.version.cuda is None:
        print(
            "PyTorch is CPU-only (no CUDA runtime in this install). "
            "GPU present but unused. To enable CUDA, run .\\scripts\\install_torch_cuda.ps1 "
            "or see https://pytorch.org — then rerun training."
        )
    return torch.device("cpu")


def _biochem_env_truthy(key: str, default: bool = False) -> bool:
    """Parse ``1/true/on`` env flags; missing key returns ``default``."""
    v = (os.environ.get(key, "") or "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def _biochem_stop_after_teacher() -> bool:
    """True when teacher-only runs should exit before corrector / pseudo-label distillation."""
    return _biochem_env_truthy("BIOCHEM_STOP_AFTER_TEACHER", default=False)


def _biochem_teacher_mu_ratio_max(bio_cfg) -> float:
    """Rheology saturation scale for teacher / preflight (COMSOL high-μ), not Newtonian-only."""
    raw = (os.environ.get("BIOCHEM_TEACHER_MU_RATIO_MAX") or "").strip()
    if raw:
        return max(float(raw), 1.0)
    return max(float(getattr(bio_cfg, "mu_ratio_max", 80.0)), 1.0)


def _resolve_tbptt_anchor_start_idx(
    actual_num_steps: int,
    window_size: int,
    device: torch.device,
) -> int:
    """Anchor TBPTT slice start: end-aligned (late clot), random, or t=0 (legacy)."""
    max_start = max(0, int(actual_num_steps) - int(window_size))
    if max_start <= 0:
        return 0
    if _biochem_env_truthy("BIOCHEM_TBPTT_ANCHOR_END_BIAS", default=False):
        return int(max_start)
    if _biochem_env_truthy("BIOCHEM_TBPTT_ANCHOR_RANDOM_START", default=False):
        return int(torch.randint(0, max_start + 1, (1,), device=device).item())
    return 0


def _tbptt_mu_time_weights(t_cap: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Per-time weights for μ anchor losses inside a TBPTT window (emphasize late steps when enabled)."""
    if t_cap < 1:
        return torch.ones(1, device=device, dtype=dtype)
    if not _biochem_env_truthy("BIOCHEM_MU_ANCHOR_LATE_TIME_WEIGHT", default=True):
        return torch.ones(t_cap, device=device, dtype=dtype) / float(t_cap)
    power = max(float(os.environ.get("BIOCHEM_MU_LATE_TIME_POWER", "2.0")), 0.0)
    w = torch.linspace(1.0, 1.0 + power, steps=t_cap, device=device, dtype=dtype)
    return w / w.sum().clamp(min=1e-12)


def _apply_biochem_mu_best_practice_env(*, only_if_missing: bool = True) -> None:
    """Defaults for COMSOL-aligned μ: late TBPTT, high-μ rheology cap, log+SI losses, stable ODE."""
    def _put(k: str, v: str) -> None:
        if only_if_missing and k in os.environ:
            return
        os.environ[k] = v

    _put("BIOCHEM_TEACHER_MU_RATIO_MAX", "80.0")
    _put("BIOCHEM_MU_SI_HUBER_DELTA", "0.25")
    _put("BIOCHEM_MU_LOG_ANCHOR_WEIGHT", "2.0")
    _put("BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT", "8.0")
    _put("BIOCHEM_MU_SI_MULTI_STEP", "1")
    _put("BIOCHEM_MU_ANCHOR_LATE_TIME_WEIGHT", "1")
    _put("BIOCHEM_MU_LATE_TIME_POWER", "2.0")
    _put("BIOCHEM_TBPTT_ANCHOR_END_BIAS", "1")
    _put("BIOCHEM_TBPTT_ANCHOR_RANDOM_START", "0")
    _put("BIOCHEM_TEACHER_TBPTT_RANDOM_ANCHOR", "0")
    _put("BIOCHEM_ADJOINT_RK4_SUBSTEPS", "32")
    _put("BIOCHEM_FI_GATE_START_WEIGHT", "0.0")


def _should_log_pretrain_epoch(epoch_idx: int, total_epochs: int, interval: int) -> bool:
    """Whether to print per-epoch AE / ODE-RXN lines (see ``BIOCHEM_PRETRAIN_LOG_INTERVAL``)."""
    if interval <= 1:
        return True
    if epoch_idx == 0 or epoch_idx == total_epochs - 1:
        return True
    return (epoch_idx + 1) % interval == 0


def configure_cuda_for_training(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Optional; frees cached blocks but forces an allocator sync — skip unless chasing fragmentation.
    if _biochem_env_truthy("BIOCHEM_CUDA_EMPTY_CACHE_AT_START", default=False):
        torch.cuda.empty_cache()
    props = torch.cuda.get_device_properties(device)
    try:
        free_b, total_b = torch.cuda.mem_get_info()
        mem_str = f"{free_b / (1024 ** 3):.2f} / {total_b / (1024 ** 3):.2f} GiB free / total"
    except Exception:
        mem_str = f"{props.total_memory / (1024 ** 3):.1f} GiB total"
    print(f"CUDA device: {props.name} | {mem_str}")


def _apply_biochem_matmul_precision() -> None:
    """Use TF32-friendly matmul kernels where available (throughput vs strict FP32)."""
    allowed = ("highest", "high", "medium")
    raw = (os.environ.get("BIOCHEM_MATMUL_PRECISION") or "").strip().lower()
    mode = raw if raw in allowed else ("high" if torch.cuda.is_available() else "")
    if not mode:
        return
    try:
        torch.set_float32_matmul_precision(mode)
        print(f"⚡ torch.set_float32_matmul_precision('{mode}') (BIOCHEM_MATMUL_PRECISION=highest|high|medium).")
    except Exception:
        pass


def _apply_biochem_sparse_invariant_mode() -> None:
    """Opt in/out of PyTorch sparse COO invariant checks (stops repeated 2.x UserWarnings).

    Default: **off** (``torch.sparse.check_sparse_tensor_invariants.disable()``) — matches training
    performance and silences "implicitly disabled" warnings on every ``sparse_coo_tensor`` (notably
    with ``num_workers>0``, where each worker process hits the factory once).

    Set ``BIOCHEM_SPARSE_INVARIANT_CHECKS=1`` to **enable** checks when debugging suspected corrupt
    sparse WLS operators (slower).
    """
    if not hasattr(torch.sparse, "check_sparse_tensor_invariants"):
        return
    ctrl = torch.sparse.check_sparse_tensor_invariants
    try:
        if _biochem_env_truthy("BIOCHEM_SPARSE_INVARIANT_CHECKS"):
            ctrl.enable()
            print(
                "🔎 torch.sparse.check_sparse_tensor_invariants enabled "
                "(BIOCHEM_SPARSE_INVARIANT_CHECKS=1; slower)."
            )
        else:
            ctrl.disable()
    except Exception:
        pass


def _biochem_dataloader_worker_init_fn(_worker_id: int) -> None:
    """Spawned DataLoader workers need the same sparse-invariant policy as the main process."""
    _apply_biochem_sparse_invariant_mode()


def _biochem_dataloader_kw(device: torch.device) -> Dict[str, Any]:
    """Shared DataLoader kwargs; tune with BIOCHEM_DATALOADER_WORKERS / BIOCHEM_PIN_MEMORY."""
    try:
        nw = max(0, int(os.environ.get("BIOCHEM_DATALOADER_WORKERS", "0")))
    except ValueError:
        nw = 0
    pin_default = device.type == "cuda"
    pm = _biochem_env_truthy("BIOCHEM_PIN_MEMORY", default=pin_default)
    kw: Dict[str, Any] = {"num_workers": nw, "pin_memory": pm}
    if nw > 0:
        kw["persistent_workers"] = True
        kw["worker_init_fn"] = _biochem_dataloader_worker_init_fn
        try:
            pf = max(2, int(os.environ.get("BIOCHEM_DATALOADER_PREFETCH", "2")))
        except ValueError:
            pf = 2
        kw["prefetch_factor"] = pf
    return kw


def _biochem_non_blocking_transfer(device: torch.device, dl_kw: Dict[str, Any]) -> bool:
    return device.type == "cuda" and bool(dl_kw.get("pin_memory"))


from tqdm import tqdm
import random
from torch_geometric.data import Dataset
from src.utils.paths import (
    data_root,
    get_project_root,
    reports_training_dir,
    stage_b_dir,
    resolve_checkpoint,
)
from src.architecture.gnode_biochem import GNODE_Phase3, biochem_truth_node_mask
from src.architecture.lora_injection import inject_lora_to_spectral_linears
from src.core_physics.biochem_physics_kernels import BiochemPhysicsKernels
from src.core_physics.kinematics_clot_prior import clot_prior_score_flat
from src.core_physics.physics_kernels import PhysicsKernels
from src.config import (
    VesselConfig,
    PhysicsConfig,
    BiochemConfig,
    CurriculumConfig,
    STATE_CHANNEL_MU_EFF_ND,
)
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import WeightedRandomSampler
from src.utils.batching import get_batch_tensor
from src.utils.metrics import DynamicLossWeighter
from src.utils.boundary_flux import flux_debug_from_graph_data, flux_debug_to_training_metrics
from src.utils.training_diary import TrainingDiary, env_snapshot
from src.training.physics_curriculum import ease01 as _ease01
from src.utils.channel_schema import BIO_Y_SCHEMA, assert_graph_schema, infer_missing_schema
from src.utils.nondim import to_t_nd


@dataclass(frozen=True)
class BiochemTrainingConfig:
    """Resolved training knobs for reproducible Biochem runs.

    Values are read after the one-click/default env setup has run, so OS/IDE
    overrides still win while the active configuration can be logged as data.
    """

    kine_prior_weight: float
    kine_prior_ramp_epochs: int
    latent_reg_scale: float
    teacher_physics_ceiling: float

    @classmethod
    def from_env(cls) -> "BiochemTrainingConfig":
        return cls(
            kine_prior_weight=float(os.environ.get("BIOCHEM_KINE_PRIOR_WEIGHT", "0.0")),
            kine_prior_ramp_epochs=max(0, int(os.environ.get("BIOCHEM_KINE_PRIOR_RAMP_EPOCHS", "0"))),
            latent_reg_scale=float(os.environ.get("BIOCHEM_LATENT_REG_SCALE", "5e-2")),
            teacher_physics_ceiling=float(os.environ.get("BIOCHEM_TEACHER_PHYSICS_PRECISION_CEILING", "1e-8")),
        )


def _apply_biochem_complexity_step2_env() -> None:
    """Set-if-missing defaults for **complexity step 2** (μ SI + wider TBPTT ramp).

    Activates when ``BIOCHEM_COMPLEXITY_STEP`` is one of: ``2``, ``2.0``, ``phase2``, ``teacher_v2``.
    Skipped when ``BIOCHEM_STOCK_DEFAULTS=1``. The main ``_apply_pycharm_biochem_optimal_defaults`` preset
    already sets these values when keys are unset; this helper remains for ``BIOCHEM_STOCK_DEFAULTS=1``
    workflows where you opt in with ``BIOCHEM_COMPLEXITY_STEP=2`` alone.

    Effects (each key only set if missing — explicit Run Configuration still wins):

    - **μ SI anchor** in data-only backprop via ``BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT`` (direct Huber on
      effective viscosity in SI on COMSOL anchors; complements variance-normalized ``L_Data_Kine``).
    - **TBPTT:** higher ``BIOCHEM_TBPTT_MAX_WINDOW`` cap and ``BIOCHEM_TBPTT_WINDOW_CURRICULUM=1`` so
      the temporal slice grows with epoch (still VRAM-bounded). **Late clot:** ``BIOCHEM_TBPTT_ANCHOR_END_BIAS=1``
      aligns the window to the **end** of the trajectory; optional ``BIOCHEM_TBPTT_ANCHOR_RANDOM_START=1``.

    On tight GPUs, lower ``BIOCHEM_TBPTT_MAX_WINDOW`` or set ``BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT=0``.
    """
    if (os.environ.get("BIOCHEM_STOCK_DEFAULTS", "") or "").strip().lower() in ("1", "true", "yes", "on"):
        return
    step = (os.environ.get("BIOCHEM_COMPLEXITY_STEP") or "").strip().lower()
    if step not in ("2", "2.0", "phase2", "teacher_v2"):
        return

    def _s(k: str, v: str) -> None:
        if k not in os.environ:
            os.environ[k] = v

    _apply_biochem_mu_best_practice_env(only_if_missing=True)
    _s("BIOCHEM_MU_SI_ANCHOR_AUX_EARLY_EPOCHS", "8")
    _s("BIOCHEM_MU_SI_ANCHOR_AUX_EARLY_MULT", "1.5")
    _s("BIOCHEM_TBPTT_MAX_WINDOW", "12")
    _s("BIOCHEM_TBPTT_WINDOW_CURRICULUM", "1")


def _apply_biochem_complexity_step3_env() -> None:
    """Defaults for **complexity step 3**: Kendall multitask + PDE terms in ``backward`` (not data-only).

    Activates when ``BIOCHEM_COMPLEXITY_STEP`` is one of:
    ``3``, ``3.0``, ``phase3``, ``full_multitask``, ``corrector_full``.

    When active, **always** sets ``BIOCHEM_LOSS_DATA_ONLY=0`` (overrides the fast-iterate default).
    Other keys are set-if-missing: slightly tighter TBPTT cap, milder μ SI duplicate penalty vs full PDE,
    and ``BIOCHEM_DATA_ONLY_PHYS_TEMP=0`` so logs do not imply a data-only-only temporal path.

    Skipped when ``BIOCHEM_STOCK_DEFAULTS=1``.
    """
    if (os.environ.get("BIOCHEM_STOCK_DEFAULTS", "") or "").strip().lower() in ("1", "true", "yes", "on"):
        return
    step = (os.environ.get("BIOCHEM_COMPLEXITY_STEP") or "").strip().lower()
    if step not in ("3", "3.0", "phase3", "full_multitask", "corrector_full"):
        return

    def _s(k: str, v: str) -> None:
        if k not in os.environ:
            os.environ[k] = v

    os.environ["BIOCHEM_LOSS_DATA_ONLY"] = "0"
    os.environ["BIOCHEM_DATA_ONLY_PHYS_TEMP"] = "0"
    _s("BIOCHEM_TBPTT_MAX_WINDOW", "8")
    _s("BIOCHEM_TBPTT_WINDOW_CURRICULUM", "1")
    _s("BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT", "0.35")


# Opt-in overnight / diagnostic run at **same supervised tier** as default step-2 (data-only + μ SI +
# TBPTT ramp; still no Kendall PDE sum). Set ``BIOCHEM_PRESET=overnight_step2`` (aliases below); these
# keys **override** the fast-iterate defaults from ``_apply_pycharm_biochem_optimal_defaults``.
_OVERNIGHT_STEP2_PRESET_ALIASES = frozenset({"overnight_step2", "comprehensive_step2", "step2_overnight"})


def _apply_biochem_preset_overnight_step2_if_requested() -> None:
    """Apply ``BIOCHEM_PRESET=overnight_step2`` (or alias): longer pretrain/teacher, finer val, full debug.

    Still ``BIOCHEM_LOSS_DATA_ONLY=1``. Adds **COMSOL temporal** Huber on anchors when
    ``BIOCHEM_DATA_ONLY_PHYS_TEMP=1`` (set in this preset). Does **not** enable multitask Kendall loss.

    Overrides env keys even if the fast preset already set them — opt-in only via ``BIOCHEM_PRESET``.
    Skipped when ``BIOCHEM_STOCK_DEFAULTS=1``. On 4 GiB GPUs, lower ``BIOCHEM_TBPTT_MAX_WINDOW`` if OOM.
    """
    preset = (os.environ.get("BIOCHEM_PRESET") or "").strip().lower()
    if preset not in _OVERNIGHT_STEP2_PRESET_ALIASES:
        return
    if (os.environ.get("BIOCHEM_STOCK_DEFAULTS", "") or "").strip().lower() in ("1", "true", "yes", "on"):
        return

    overnight: Dict[str, str] = {
        "BIOCHEM_DEBUG": "1",
        "BIOCHEM_TEACHER_EPOCHS": "60",
        "BIOCHEM_TEACHER_TF_WARMUP_EPOCHS": "60",
        "BIOCHEM_EPOCHS": "72",
        "BIOCHEM_AE_EPOCHS": "40",
        "BIOCHEM_AE_MIN_EPOCHS": "10",
        "BIOCHEM_AE_PATIENCE": "6",
        "BIOCHEM_ODE_RXN_EPOCHS": "30",
        "BIOCHEM_ODE_MIN_EPOCHS": "18",
        "BIOCHEM_ODE_PATIENCE": "8",
        "BIOCHEM_ODE_RXN_MIN_EPOCHS_BEFORE_PLATEAU": "12",
        "BIOCHEM_TEACHER_VAL_EVERY": "1",
        "BIOCHEM_VAL_TIME_STRIDE": "5",
        "BIOCHEM_PRETRAIN_LOG_INTERVAL": "5",
        "BIOCHEM_WARMUP_EPOCHS": "24",
        "BIOCHEM_PHYSICS_PRECISION_RAMP_EPOCHS": "18",
        "BIOCHEM_RESIDUAL_SPARSE_RAMP_EPOCHS": "35",
        "BIOCHEM_TBPTT_MAX_WINDOW": "14",
        "BIOCHEM_TBPTT_WINDOW_CURRICULUM": "1",
        "BIOCHEM_TBPTT_ANCHOR_RANDOM_START": "0",
        "BIOCHEM_TEACHER_TBPTT_RANDOM_ANCHOR": "0",
        "BIOCHEM_COMPLEXITY_STEP": "2",
        "BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT": "1.0",
        "BIOCHEM_MU_SI_ANCHOR_AUX_EARLY_EPOCHS": "16",
        "BIOCHEM_MU_SI_ANCHOR_AUX_EARLY_MULT": "1.5",
        "BIOCHEM_DATA_ONLY_PHYS_TEMP": "1",
        "BIOCHEM_COMSOL_TEMPORAL_WEIGHT": "0.02",
        "BIOCHEM_ABORT_BAD_TEACHER_INIT": "1",
        "BIOCHEM_TEACHER_ODE_FREEZE_EPOCHS": "4",
    }
    for k, v in overnight.items():
        os.environ[k] = v
    os.environ["BIOCHEM_ODE_REACTION_EPOCHS"] = os.environ["BIOCHEM_ODE_RXN_EPOCHS"]
    _apply_biochem_mu_best_practice_env(only_if_missing=False)


_COMPREHENSIVE_MU_PRESET_ALIASES = frozenset({
    "comprehensive_mu",
    "mu_comprehensive",
    "step2_comprehensive",
    "biochem_comprehensive",
})


def _apply_biochem_preset_comprehensive_mu_if_requested() -> None:
    """Experimental preset: corona bundle + longer teacher/corrector + μ env defaults.

    **Unvalidated** for μ improvement (see training progress log). Still ``BIOCHEM_LOSS_DATA_ONLY=1``.
    Prefer ``run_biochem_mu_formulation_study.ps1`` until step-2 teacher is stable on patient007.
    On 4 GiB GPUs, lower ``BIOCHEM_TBPTT_MAX_WINDOW`` / ``BIOCHEM_ADJOINT_RK4_SUBSTEPS``.
    """
    preset = (os.environ.get("BIOCHEM_PRESET") or "").strip().lower()
    if preset not in _COMPREHENSIVE_MU_PRESET_ALIASES:
        return
    if (os.environ.get("BIOCHEM_STOCK_DEFAULTS", "") or "").strip().lower() in ("1", "true", "yes", "on"):
        return

    comprehensive: Dict[str, str] = {
        "BIOCHEM_GELATION_PRIOR_GATE": "1",
        "BIOCHEM_PRIOR_THROMBUS_CORONA_HOPS": "3",
        "BIOCHEM_LOSS_DATA_ONLY": "1",
        "BIOCHEM_DATA_ONLY_PHYS_TEMP": "1",
        "BIOCHEM_COMSOL_TEMPORAL_WEIGHT": "0.02",
        "BIOCHEM_COMPLEXITY_STEP": "2",
        "BIOCHEM_STOP_AFTER_TEACHER": "0",
        "BIOCHEM_NO_TEACHER_DEFAULTS": "1",
        "BIOCHEM_TEACHER_FORCE_MIN": "0.2",
        "BIOCHEM_TEACHER_TF_WARMUP_EPOCHS": "6",
        "BIOCHEM_TEACHER_EPOCHS": "36",
        "BIOCHEM_EPOCHS": "48",
        "BIOCHEM_AE_EPOCHS": "20",
        "BIOCHEM_AE_MIN_EPOCHS": "6",
        "BIOCHEM_AE_PATIENCE": "4",
        "BIOCHEM_ODE_RXN_EPOCHS": "16",
        "BIOCHEM_ODE_MIN_EPOCHS": "8",
        "BIOCHEM_ODE_PATIENCE": "5",
        "BIOCHEM_ODE_RXN_MIN_EPOCHS_BEFORE_PLATEAU": "10",
        "BIOCHEM_TEACHER_VAL_EVERY": "3",
        "BIOCHEM_VAL_EVERY": "5",
        "BIOCHEM_VAL_TIME_STRIDE": "5",
        "BIOCHEM_WARMUP_EPOCHS": "12",
        "BIOCHEM_PHYSICS_PRECISION_RAMP_EPOCHS": "14",
        "BIOCHEM_TBPTT_MAX_WINDOW": "10",
        "BIOCHEM_TBPTT_WINDOW_CURRICULUM": "1",
        "BIOCHEM_TEACHER_ODE_FREEZE_EPOCHS": "4",
        "BIOCHEM_PSEUDO_MIN_TEACHER_MU_SCORE": "-1.35",
        "BIOCHEM_DEBUG": "0",
        "BIOCHEM_PRETRAIN_LOG_INTERVAL": "5",
    }
    for k, v in comprehensive.items():
        os.environ[k] = v
    os.environ["BIOCHEM_ODE_REACTION_EPOCHS"] = os.environ["BIOCHEM_ODE_RXN_EPOCHS"]
    _apply_biochem_mu_best_practice_env(only_if_missing=False)
    print(
        "⚠️ BIOCHEM_PRESET=comprehensive_mu is experimental/unvalidated "
        "(see src/docs/BIOCHEM_TRAINING_PROGRESS.md).",
        flush=True,
    )


_TEACHER_VISC_BASELINE_PRESET_ALIASES = frozenset({
    "teacher_visc_baseline",
    "visc_teacher_baseline",
    "teacher_mu_base",
})


def _apply_biochem_preset_teacher_visc_baseline_if_requested() -> None:
    """Teacher-only viscosity baseline for incremental loss ablations.

    Goal: stable core objective for viscosity fidelity over time (global + wall + high-μ),
    while retaining essential flow/species supervision via ``L_Data_Kine`` + ``L_Data_Bio``.
    """
    preset = (os.environ.get("BIOCHEM_PRESET") or "").strip().lower()
    if preset not in _TEACHER_VISC_BASELINE_PRESET_ALIASES:
        return
    if (os.environ.get("BIOCHEM_STOCK_DEFAULTS", "") or "").strip().lower() in ("1", "true", "yes", "on"):
        return

    bundle: Dict[str, str] = {
        "BIOCHEM_COMPLEXITY_STEP": "2",
        "BIOCHEM_LOSS_DATA_ONLY": "1",
        "BIOCHEM_STOP_AFTER_TEACHER": "1",
        "BIOCHEM_NO_TEACHER_DEFAULTS": "1",
        "BIOCHEM_DATA_ONLY_PHYS_TEMP": "0",
        "BIOCHEM_TEACHER_EPOCHS": "18",
        "BIOCHEM_TEACHER_VAL_EVERY": "2",
        "BIOCHEM_VAL_TIME_STRIDE": "10",
        "BIOCHEM_TBPTT_MAX_WINDOW": "6",
        "BIOCHEM_TBPTT_WINDOW_CURRICULUM": "0",
        "BIOCHEM_DETACH_MACRO_STATE": "1",
        "BIOCHEM_TEACHER_FORCE_MIN": "0.0",
        "BIOCHEM_TEACHER_TF_WARMUP_EPOCHS": "4",
        "BIOCHEM_TEACHER_MU_RATIO_MAX": "80.0",
        "BIOCHEM_MU_LOG_ANCHOR_WEIGHT": "2.0",
        "BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT": "2.0",
        "BIOCHEM_MU_LOG_WALL_WEIGHT": "2.0",
        "BIOCHEM_MU_LOG_HIGH_WEIGHT": "1.0",
        "BIOCHEM_MU_LOG_WALL_RAMP_EPOCHS": "6",
        "BIOCHEM_MU_LOG_HIGH_RAMP_EPOCHS": "8",
        "BIOCHEM_MU_SI_MULTI_STEP": "1",
        "BIOCHEM_MU_ANCHOR_LATE_TIME_WEIGHT": "1",
        "BIOCHEM_MU_LATE_TIME_POWER": "2.0",
        "BIOCHEM_USE_MU_PATH_GROUP": "1",
        "BIOCHEM_TRAIN_MU_ENCODER": "1",
        "BIOCHEM_USE_DELTA_MU_HEAD": "1",
        "BIOCHEM_GELATION_PRIOR_GATE": "1",
        "BIOCHEM_ADJOINT_RK4_SUBSTEPS": "12",
        "BIOCHEM_DATALOADER_WORKERS": "0",
        "BIOCHEM_ABORT_BAD_TEACHER_INIT": "1",
    }
    for k, v in bundle.items():
        os.environ[k] = v
    os.environ.pop("BIOCHEM_LOSS_ISOLATE", None)
    _apply_biochem_mu_best_practice_env(only_if_missing=False)
    print(
        "✅ BIOCHEM_PRESET=teacher_visc_baseline: teacher-only step-2 baseline "
        "(Data_Kine/Data_Bio + μ log global/wall/high + μ SI) enabled.",
        flush=True,
    )


_TEACHER_MAX_COMPLEXITY_PRESET_ALIASES = frozenset({
    "teacher_max_complexity",
    "max_complexity_teacher",
    "teacher_step3_robust",
})


def _apply_biochem_preset_teacher_max_complexity_if_requested() -> None:
    """Robust teacher-only preset: full multitask (step 3) + viscosity-focused μ supervision.

    Intended for the teacher stage only (keeps ``BIOCHEM_STOP_AFTER_TEACHER=1``).
    This preset is for "all important losses active" runs where viscosity fidelity is
    the primary objective and single-loss isolate state must be cleared.
    """
    preset = (os.environ.get("BIOCHEM_PRESET") or "").strip().lower()
    if preset not in _TEACHER_MAX_COMPLEXITY_PRESET_ALIASES:
        return
    if (os.environ.get("BIOCHEM_STOCK_DEFAULTS", "") or "").strip().lower() in ("1", "true", "yes", "on"):
        return

    bundle: Dict[str, str] = {
        "BIOCHEM_COMPLEXITY_STEP": "3",
        "BIOCHEM_LOSS_DATA_ONLY": "0",
        "BIOCHEM_STOP_AFTER_TEACHER": "1",
        "BIOCHEM_NO_TEACHER_DEFAULTS": "0",
        "BIOCHEM_DATA_ONLY_PHYS_TEMP": "0",
        "BIOCHEM_TEACHER_EPOCHS": "24",
        "BIOCHEM_TEACHER_VAL_EVERY": "2",
        "BIOCHEM_VAL_TIME_STRIDE": "10",
        "BIOCHEM_TBPTT_MAX_WINDOW": "8",
        "BIOCHEM_TBPTT_WINDOW_CURRICULUM": "1",
        "BIOCHEM_DETACH_MACRO_STATE": "0",
        "BIOCHEM_TEACHER_FORCE_MIN": "0.2",
        "BIOCHEM_TEACHER_TF_WARMUP_EPOCHS": "6",
        "BIOCHEM_TEACHER_MU_RATIO_MAX": "80.0",
        "BIOCHEM_MU_LOG_ANCHOR_WEIGHT": "2.0",
        "BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT": "2.0",
        "BIOCHEM_MU_SI_MULTI_STEP": "1",
        "BIOCHEM_USE_MU_PATH_GROUP": "1",
        "BIOCHEM_TRAIN_MU_ENCODER": "1",
        "BIOCHEM_USE_DELTA_MU_HEAD": "1",
        "BIOCHEM_GELATION_PRIOR_GATE": "1",
        "BIOCHEM_ADJOINT_RK4_SUBSTEPS": "24",
        "BIOCHEM_ABORT_BAD_TEACHER_INIT": "1",
    }
    for k, v in bundle.items():
        os.environ[k] = v
    # Avoid accidental single-term smoke settings leaking into full-complexity runs.
    os.environ.pop("BIOCHEM_LOSS_ISOLATE", None)
    _apply_biochem_mu_best_practice_env(only_if_missing=False)
    print(
        "✅ BIOCHEM_PRESET=teacher_max_complexity: teacher-only step-3 multitask + μ-focused supervision enabled.",
        flush=True,
    )


_STEP2P5_PRESET_ALIASES = frozenset({"step2p5", "phys_temp_only", "compact_step2p5"})


def _apply_biochem_preset_step2p5_phys_temp_if_requested() -> None:
    """``BIOCHEM_PRESET=step2p5`` (or alias): enable COMSOL temporal Huber on anchors without long-run overnight.

    Still ``BIOCHEM_LOSS_DATA_ONLY=1``. Skipped when ``BIOCHEM_STOCK_DEFAULTS=1``. Use a different
    ``BIOCHEM_PRESET`` value than ``overnight_step2`` (that preset already enables phys temp).
    """
    preset = (os.environ.get("BIOCHEM_PRESET") or "").strip().lower()
    if preset not in _STEP2P5_PRESET_ALIASES:
        return
    if (os.environ.get("BIOCHEM_STOCK_DEFAULTS", "") or "").strip().lower() in ("1", "true", "yes", "on"):
        return
    os.environ["BIOCHEM_LOSS_DATA_ONLY"] = "1"
    os.environ["BIOCHEM_DATA_ONLY_PHYS_TEMP"] = "1"
    if "BIOCHEM_COMSOL_TEMPORAL_WEIGHT" not in os.environ:
        os.environ["BIOCHEM_COMSOL_TEMPORAL_WEIGHT"] = "0.02"


_THROMBUS_CORONA_PRESET_ALIASES = frozenset({"thrombus_corona", "corona_thrombus", "biochem_thrombus_corona"})


def _apply_biochem_preset_thrombus_corona_if_requested() -> None:
    """Experimental ``BIOCHEM_PRESET=thrombus_corona`` (unvalidated — not default for iteration).

    Bundles: ``BIOCHEM_GELATION_PRIOR_GATE=1``, ``BIOCHEM_PRIOR_THROMBUS_CORONA_HOPS=3``,
    data-only step 2, ``BIOCHEM_DATA_ONLY_PHYS_TEMP=1``, ``BIOCHEM_STOP_AFTER_TEACHER=0`` (corrector on).
    Slightly widens ``BIOCHEM_PRIOR_WALL_DECAY_ND`` when unset.

    Evidence: one 2026-05-16 run — teacher μ flat ~1.48, corrector 1.57→1.55; μ cap/TBPTT/TF confounds.
    Not rerun after MU_LOG + μ-path A0 (patient007 ~0.44). Test corona components individually later.

    Skipped when ``BIOCHEM_STOCK_DEFAULTS=1``. Conflicts with ``overnight_step2`` if both set — pick one preset.
    """
    preset = (os.environ.get("BIOCHEM_PRESET") or "").strip().lower()
    if preset not in _THROMBUS_CORONA_PRESET_ALIASES:
        return
    if (os.environ.get("BIOCHEM_STOCK_DEFAULTS", "") or "").strip().lower() in ("1", "true", "yes", "on"):
        return
    bundle: Dict[str, str] = {
        "BIOCHEM_GELATION_PRIOR_GATE": "1",
        "BIOCHEM_PRIOR_THROMBUS_CORONA_HOPS": "3",
        "BIOCHEM_LOSS_DATA_ONLY": "1",
        "BIOCHEM_DATA_ONLY_PHYS_TEMP": "1",
        "BIOCHEM_COMSOL_TEMPORAL_WEIGHT": "0.02",
        "BIOCHEM_STOP_AFTER_TEACHER": "0",
        "BIOCHEM_COMPLEXITY_STEP": "2",
    }
    for k, v in bundle.items():
        os.environ[k] = v
    if "BIOCHEM_PRIOR_WALL_DECAY_ND" not in os.environ:
        os.environ["BIOCHEM_PRIOR_WALL_DECAY_ND"] = "0.01"
    _apply_biochem_mu_best_practice_env(only_if_missing=True)
    print(
        "⚠️ BIOCHEM_PRESET=thrombus_corona is experimental/unvalidated "
        "(see src/docs/BIOCHEM_TRAINING_PROGRESS.md). Prefer mu formulation study for μ work.",
        flush=True,
    )


_SWEEP_WALL_SENTINEL_ALIASES = frozenset({"sweep_wall_sentinel"})


def _apply_biochem_preset_sweep_wall_sentinel_if_requested() -> None:
    if (os.environ.get("BIOCHEM_PRESET") or "").strip().lower() not in _SWEEP_WALL_SENTINEL_ALIASES:
        return
    bundle: Dict[str, str] = {
        "BIOCHEM_COMPLEXITY_STEP": "2",
        "BIOCHEM_LOSS_DATA_ONLY": "1",
        "BIOCHEM_LOSS_ISOLATE": "MU_LOG",
        "BIOCHEM_TEACHER_EPOCHS": "14",
        "BIOCHEM_TEACHER_VAL_EVERY": "2",
        "BIOCHEM_VAL_TIME_STRIDE": "10",
        "BIOCHEM_STOP_AFTER_TEACHER": "1",
        "BIOCHEM_MU_LOG_ANCHOR_WEIGHT": "1.0",
        "BIOCHEM_MU_LOG_WALL_WEIGHT": "3.6",
        "BIOCHEM_MU_LOG_HIGH_WEIGHT": "1.8",
        "BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT": "0.0",
        "BIOCHEM_USE_MU_PATH_GROUP": "1",
        "BIOCHEM_TRAIN_MU_ENCODER": "1",
        "BIOCHEM_USE_DELTA_MU_HEAD": "1",
        "BIOCHEM_USE_SPLIT_MU_HEAD": "1",
        "BIOCHEM_TEACHER_MU_RATIO_MAX": "80.0",
        "BIOCHEM_MU_TRIGGER_GATE_TEMP_START": "0.10",
        "BIOCHEM_MU_TRIGGER_GATE_TEMP_END": "0.04",
        "BIOCHEM_TRIGGER_GATE_MIN": "0.06",
        "BIOCHEM_WALL_GATE_MIN": "0.08",
        "BIOCHEM_MU_WALL_GATE_CENTER": "0.45",
        "BIOCHEM_MU_WALL_GATE_TEMP": "0.14",
        "BIOCHEM_MU_WALL_DELTA_GAIN": "0.85",
        "BIOCHEM_WALL_GATE_BIAS": "0.25",
        "BIOCHEM_WALL_MASK_LOGIT_BOOST": "0.60",
        "BIOCHEM_TBPTT_MAX_WINDOW": "5",
        "BIOCHEM_DETACH_MACRO_STATE": "1",
        "BIOCHEM_ADJOINT_RK4_SUBSTEPS": "8",
        "BIOCHEM_USE_BIO_GATE_SUPPRESSOR": "0",
        "BIOCHEM_USE_WALL_DELTA_HEAD": "1",
        "BIOCHEM_STOCK_DEFAULTS": "1",
    }
    for k, v in bundle.items():
        os.environ[k] = v
    print("✅ BIOCHEM_PRESET=sweep_wall_sentinel: Step-2 MU isolate + split μ architecture (fast probe).", flush=True)


_SWEEP_BIO_SUPPRESSOR_ALIASES = frozenset({"sweep_bio_suppressor"})


def _apply_biochem_preset_sweep_bio_suppressor_if_requested() -> None:
    if (os.environ.get("BIOCHEM_PRESET") or "").strip().lower() not in _SWEEP_BIO_SUPPRESSOR_ALIASES:
        return
    bundle: Dict[str, str] = {
        "BIOCHEM_COMPLEXITY_STEP": "2",
        "BIOCHEM_LOSS_DATA_ONLY": "1",
        "BIOCHEM_LOSS_ISOLATE": "MU_LOG",
        "BIOCHEM_TEACHER_EPOCHS": "14",
        "BIOCHEM_TEACHER_VAL_EVERY": "2",
        "BIOCHEM_VAL_TIME_STRIDE": "10",
        "BIOCHEM_STOP_AFTER_TEACHER": "1",
        "BIOCHEM_MU_LOG_ANCHOR_WEIGHT": "1.0",
        "BIOCHEM_MU_LOG_WALL_WEIGHT": "3.2",
        "BIOCHEM_MU_LOG_HIGH_WEIGHT": "1.8",
        "BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT": "0.0",
        "BIOCHEM_USE_MU_PATH_GROUP": "1",
        "BIOCHEM_TRAIN_MU_ENCODER": "1",
        "BIOCHEM_USE_DELTA_MU_HEAD": "1",
        "BIOCHEM_USE_SPLIT_MU_HEAD": "1",
        "BIOCHEM_TEACHER_MU_RATIO_MAX": "80.0",
        "BIOCHEM_MU_TRIGGER_GATE_TEMP_START": "0.10",
        "BIOCHEM_MU_TRIGGER_GATE_TEMP_END": "0.04",
        "BIOCHEM_TRIGGER_GATE_MIN": "0.06",
        "BIOCHEM_WALL_GATE_MIN": "0.08",
        "BIOCHEM_MU_WALL_GATE_CENTER": "0.45",
        "BIOCHEM_MU_WALL_GATE_TEMP": "0.14",
        "BIOCHEM_MU_WALL_DELTA_GAIN": "0.90",
        "BIOCHEM_WALL_GATE_BIAS": "0.25",
        "BIOCHEM_WALL_MASK_LOGIT_BOOST": "0.60",
        "BIOCHEM_TBPTT_MAX_WINDOW": "5",
        "BIOCHEM_DETACH_MACRO_STATE": "1",
        "BIOCHEM_ADJOINT_RK4_SUBSTEPS": "8",
        "BIOCHEM_USE_BIO_GATE_SUPPRESSOR": "1",
        "BIOCHEM_BIO_SUPPRESSOR_THRESHOLD_SI": "5e-5",
        "BIOCHEM_BIO_SUPPRESSOR_POWER": "0.85",
        "BIOCHEM_BIO_SUPPRESSOR_GATE_FLOOR": "0.10",
        "BIOCHEM_BIO_SUPPRESS_WALL_ALPHA": "0.25",
        "BIOCHEM_USE_WALL_DELTA_HEAD": "1",
        "BIOCHEM_STOCK_DEFAULTS": "1",
    }
    for k, v in bundle.items():
        os.environ[k] = v
    print("✅ BIOCHEM_PRESET=sweep_bio_suppressor: Step-2 MU isolate + bio suppressor (fast probe).", flush=True)


_SWEEP_WALL_OVERCOMP_ALIASES = frozenset({"sweep_wall_overcomp", "sweep_wall_overcompensate"})


def _apply_biochem_preset_sweep_wall_overcomp_if_requested() -> None:
    if (os.environ.get("BIOCHEM_PRESET") or "").strip().lower() not in _SWEEP_WALL_OVERCOMP_ALIASES:
        return
    bundle: Dict[str, str] = {
        "BIOCHEM_COMPLEXITY_STEP": "2",
        "BIOCHEM_LOSS_DATA_ONLY": "1",
        "BIOCHEM_LOSS_ISOLATE": "MU_LOG",
        "BIOCHEM_TEACHER_EPOCHS": "14",
        "BIOCHEM_TEACHER_VAL_EVERY": "2",
        "BIOCHEM_VAL_TIME_STRIDE": "10",
        "BIOCHEM_STOP_AFTER_TEACHER": "1",
        "BIOCHEM_MU_LOG_ANCHOR_WEIGHT": "1.0",
        "BIOCHEM_MU_LOG_WALL_WEIGHT": "6.5",
        "BIOCHEM_MU_LOG_HIGH_WEIGHT": "1.0",
        "BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT": "0.0",
        "BIOCHEM_USE_MU_PATH_GROUP": "1",
        "BIOCHEM_TRAIN_MU_ENCODER": "1",
        "BIOCHEM_USE_DELTA_MU_HEAD": "1",
        "BIOCHEM_USE_SPLIT_MU_HEAD": "1",
        "BIOCHEM_TEACHER_MU_RATIO_MAX": "80.0",
        "BIOCHEM_MU_TRIGGER_GATE_TEMP_START": "0.10",
        "BIOCHEM_MU_TRIGGER_GATE_TEMP_END": "0.04",
        "BIOCHEM_TRIGGER_GATE_MIN": "0.03",
        "BIOCHEM_WALL_GATE_MIN": "0.20",
        "BIOCHEM_MU_WALL_GATE_CENTER": "0.30",
        "BIOCHEM_MU_WALL_GATE_TEMP": "0.10",
        "BIOCHEM_MU_WALL_DELTA_GAIN": "1.30",
        "BIOCHEM_WALL_GATE_BIAS": "0.90",
        "BIOCHEM_WALL_MASK_LOGIT_BOOST": "1.50",
        "BIOCHEM_TBPTT_MAX_WINDOW": "5",
        "BIOCHEM_DETACH_MACRO_STATE": "1",
        "BIOCHEM_ADJOINT_RK4_SUBSTEPS": "8",
        "BIOCHEM_USE_BIO_GATE_SUPPRESSOR": "0",
        "BIOCHEM_USE_WALL_DELTA_HEAD": "1",
        "BIOCHEM_STOCK_DEFAULTS": "1",
    }
    for k, v in bundle.items():
        os.environ[k] = v
    print("✅ BIOCHEM_PRESET=sweep_wall_overcomp: intentional wall-overcompensation probe active.", flush=True)


def _apply_pycharm_biochem_optimal_defaults() -> None:
    """Apply set-if-missing env defaults: fast-iterate schedule + data-only step-2 supervision.

    Skipped if ``BIOCHEM_STOCK_DEFAULTS=1``. Any key can still be overridden in the OS
    environment or IDE Run Configuration (after this runs, for keys we only ``pop``).

    **Backward** uses supervised anchors (``BIOCHEM_LOSS_DATA_ONLY=1`` → ``L_Data_Kine+L_Data_Bio``,
    plus ``W_MuSI*L_MuSI`` when ``BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT>0``, defaulted >0 here).
    With ``BIOCHEM_DATA_ONLY_PHYS_TEMP=1`` and ``BIOCHEM_COMSOL_TEMPORAL_WEIGHT>0``, add
    ``w_pt * L_PhysTemp`` to that same backprop sum (overnight preset enables this).
    Clears ``BIOCHEM_MAX_LOAD_VESSELS`` if present so inherited one-graph IDE caps do not apply;
    re-export that variable before launch if you need a cap. For single-term loss smoke tests,
    use ``BIOCHEM_STOCK_DEFAULTS=1`` and set ``BIOCHEM_LOSS_ISOLATE`` (otherwise prefer data-only).
    """
    if (os.environ.get("BIOCHEM_STOCK_DEFAULTS", "") or "").strip().lower() in ("1", "true", "yes", "on"):
        return

    # Drop inherited Run Configuration caps so "Run" loads all graphs by default.
    os.environ.pop("BIOCHEM_MAX_LOAD_VESSELS", None)

    def _s(k: str, v: str) -> None:
        if k not in os.environ:
            os.environ[k] = v

    _s("BIOCHEM_DEBUG", "0")
    # Trace forces extra GPU sync + debug.log on first clot batch; keep opt-in (unset = off).
    _s("BIOCHEM_TRACE_CLOT_BATCH", "0")
    # ---------------------------------------------------------
    # Default Run: supervised data loss only for backward (verify L_Data_* path)
    # ---------------------------------------------------------
    _s("BIOCHEM_LOSS_DATA_ONLY", "1")
    _s("BIOCHEM_KINE_PRIOR_WEIGHT", "0.0")
    _s("BIOCHEM_FI_GATE_START_WEIGHT", "0.0")
    _s("BIOCHEM_RESIDUAL_SPARSE_LAMBDA_START", "0.0")
    _s("BIOCHEM_RESIDUAL_SPARSE_LAMBDA_END", "0.0")
    # Simple overfit defaults: trainable AE decoder, multi-step TBPTT, accum=1
    # Set-if-missing defaults so Run Configuration overrides still win.
    _s("BIOCHEM_NO_TEACHER_DEFAULTS", "1")
    # Intentionally omit _s("BIOCHEM_MAX_LOAD_VESSELS"): uncapped load; any inherited value was popped above.
    _s("BIOCHEM_MAX_LOAD_SHUFFLE", "0")
    _s("BIOCHEM_DATALOADER_WORKERS", "4")
    _s("BIOCHEM_SKIP_PRETRAIN", "0")
    _s("BIOCHEM_FREEZE_DECODER_PRETRAIN", "0")
    _s("BIOCHEM_TEACHER_FORCE_MIN", "0.35")
    _s("BIOCHEM_TEACHER_CLIP_NORM", "1.0")
    _s("BIOCHEM_TEACHER_LR", "2e-3")
    _s("BIOCHEM_TEACHER_ACCUMULATION_STEPS", "1")
    _s("BIOCHEM_TEACHER_PHYSICS_PRECISION_CEILING", "1e-8")
    _s("BIOCHEM_TEACHER_SKIP_VAL", "0")
    _s("BIOCHEM_STOP_AFTER_TEACHER", "1")
    _s("BIOCHEM_EPOCHS", "18")
    _s("BIOCHEM_TEACHER_EPOCHS", "15")
    _s("BIOCHEM_TEACHER_TF_WARMUP_EPOCHS", "5")
    _s("BIOCHEM_LR", "0.001")
    _s("BIOCHEM_LATENT_DIM", "256")
    # accum>1 scales gradients without extra graphs/epoch; safe with multi-anchor splits too.
    _s("BIOCHEM_ACCUMULATION_STEPS", "1")
    _s("BIOCHEM_AE_EPOCHS", "14")
    _s("BIOCHEM_AE_MIN_EPOCHS", "4")
    _s("BIOCHEM_AE_PATIENCE", "3")
    _s("BIOCHEM_AE_MIN_DELTA", "1e-4")
    _s("BIOCHEM_WARMUP_EPOCHS", "10")
    _s("BIOCHEM_PHYSICS_PRECISION_RAMP_EPOCHS", "10")
    _s("BIOCHEM_BIO_ENCODER_PRIOR_DIM", "2")
    # Dampen the ODE derivative: keep the "speed limit" above stock (1e-3 -> 5e-2)
    # without making the resting ODE penalty dominate the initial total loss.
    # Penalizes the L_Latent_Reg "derivative energy" term inside compute_biochem_loss.
    _s("BIOCHEM_LATENT_REG_SCALE", "5e-2")
    # Cap latent ODE macro-segment length (SI seconds, split inside ``GNODE_Phase3``).
    # Large default restores old-like fast validation rollouts.
    _s("BIOCHEM_ODE_MAX_STEP_S", "30000")
    # Fixed-grid RK4 internal steps per macro subsegment (see ``GNODE_Phase3``).
    # Very low counts (e.g. 4) save VRAM but can destabilize ``odeint_adjoint`` on stiff anchors;
    # 16 is a practical default on 4 GiB laptops; raise toward 32 if a single anchor still skips steps.
    _s("BIOCHEM_ADJOINT_RK4_SUBSTEPS", "16")
    _s("BIOCHEM_TEACHER_VAL_EVERY", "3")
    _s("BIOCHEM_VAL_EVERY", "10")
    _s("BIOCHEM_VAL_TIME_STRIDE", "10")
    _s("BIOCHEM_PRETRAIN_LOG_INTERVAL", "10")
    # TBPTT: short window saves adjoint VRAM; **keep DETACH_MACRO_STATE=0** so data-loss gradients
    # propagate through the rollout (detach=1 trades OOM safety for nearly frozen L_Data_* on TBPTT).
    # If you still OOM, set BIOCHEM_DETACH_MACRO_STATE=1 and/or lower BIOCHEM_TBPTT_MAX_WINDOW further.
    _s("BIOCHEM_DETACH_MACRO_STATE", "0")
    # Gelation (FI/Mat + learned head) × kinematic clot-risk prior so μ stays Carreau-like in bulk lumen.
    _s("BIOCHEM_GELATION_PRIOR_GATE", "1")
    # Step-2 footing: wider cap + epoch ramp (lower cap if 4 GiB OOM).
    _apply_biochem_mu_best_practice_env(only_if_missing=True)
    _s("BIOCHEM_TBPTT_MAX_WINDOW", "12")
    _s("BIOCHEM_TBPTT_WINDOW_CURRICULUM", "1")
    # Let micro-head weights be random (small) so gradients flow; resting state still in bias.
    _s("BIOCHEM_MICRO_HEAD_ZERO_INIT", "0")
    _s("BIOCHEM_RESIDUAL_SPARSE_RAMP_EPOCHS", "15")
    # Stronger early bio supervision scale (only affects L_Data_Bio magnitude in metrics when not data-only).
    _s("BIOCHEM_RAW_BIO_MAGNITUDE", "0.01")
    # Phase 3a.5: shorter default for iteration; raise for production-quality ODE mimic.
    if "BIOCHEM_ODE_RXN_EPOCHS" not in os.environ and "BIOCHEM_ODE_REACTION_EPOCHS" in os.environ:
        os.environ["BIOCHEM_ODE_RXN_EPOCHS"] = os.environ["BIOCHEM_ODE_REACTION_EPOCHS"]
    _s("BIOCHEM_ODE_RXN_EPOCHS", "12")
    _s("BIOCHEM_ODE_REACTION_EPOCHS", os.environ["BIOCHEM_ODE_RXN_EPOCHS"])
    _s("BIOCHEM_ODE_EMA_BETA", "0.9")
    _s("BIOCHEM_ODE_MIN_EPOCHS", "6")
    _s("BIOCHEM_ODE_PATIENCE", "4")
    _s("BIOCHEM_ODE_MIN_DELTA", "1e-4")
    _s("BIOCHEM_ODE_RXN_MIN_EPOCHS_BEFORE_PLATEAU", "10")
    # Explicit step label (also triggers ``_apply_biochem_complexity_step2_env`` if keys left unset).
    _s("BIOCHEM_COMPLEXITY_STEP", "2")
    _s("BIOCHEM_MU_SI_ANCHOR_AUX_EARLY_EPOCHS", "8")
    _s("BIOCHEM_MU_SI_ANCHOR_AUX_EARLY_MULT", "1.5")
    _apply_biochem_complexity_step2_env()
    _apply_biochem_preset_overnight_step2_if_requested()
    _apply_biochem_preset_step2p5_phys_temp_if_requested()
    _apply_biochem_preset_thrombus_corona_if_requested()
    _apply_biochem_preset_comprehensive_mu_if_requested()
    _apply_biochem_preset_teacher_visc_baseline_if_requested()
    _apply_biochem_preset_teacher_max_complexity_if_requested()
    _apply_biochem_preset_sweep_wall_sentinel_if_requested()
    _apply_biochem_preset_sweep_bio_suppressor_if_requested()
    _apply_biochem_preset_sweep_wall_overcomp_if_requested()
    _apply_biochem_complexity_step3_env()


def _teacher_stage_best_practice_defaults(max_epochs: int) -> None:
    """Apply ``os.environ.setdefault`` for teacher-only COMSOL-aligned training.

    Respects physics: keeps BiochemConfig / kernel SI parameters unchanged; only
    optimization knobs (forcing, physics precision cap) so COMSOL labels and PDE
    terms stay balanced.

    Skipped when ``BIOCHEM_NO_TEACHER_DEFAULTS=1`` (the usual default from
    ``_apply_pycharm_biochem_optimal_defaults``). Set ``BIOCHEM_NO_TEACHER_DEFAULTS=0``
    to opt in to this block.
    """
    if (os.environ.get("BIOCHEM_NO_TEACHER_DEFAULTS", "") or "").strip().lower() in ("1", "true", "yes", "on"):
        return

    def _setdef(key: str, val: str) -> None:
        if key not in os.environ:
            os.environ[key] = val

    ramp = str(max(6, min(20, max_epochs // 3)))
    _setdef("BIOCHEM_TEACHER_FORCE_MIN", "0.46")
    # Keep teacher-stage physics precision strongly muted so PDE residual spikes
    # cannot dominate biological regression gradients.
    _setdef("BIOCHEM_TEACHER_PHYSICS_PRECISION_CEILING", "1e-4")
    # Optional teacher early-stop threshold: set BIOCHEM_TEACHER_TARGET_MU_LOG_MAE explicitly
    # (no default — track best val mu_score / checkpoint instead).
    # Keep gradients unscaled in tiny-debug regimes (often 1 batch/epoch).
    _setdef("BIOCHEM_TEACHER_ACCUMULATION_STEPS", "1")
    # Lift clip norm high enough to avoid suppressing biology gradients during
    # teacher bootstrapping; PDE spikes are still constrained by dedicated
    # physics clipping and precision-ceiling controls.
    _setdef("BIOCHEM_TEACHER_CLIP_NORM", "100.0")
    # Keep teacher stage out of zero-weight stasis by using a stronger default
    # learning rate unless the shell explicitly overrides it.
    _setdef("BIOCHEM_TEACHER_LR", "0.01")
    # Strong kinematics-prior auxiliary: forces the network to respect the
    # baseline-shear "where a clot is physically plausible" map. Bumped to 10.0
    # to actively suppress clot predictions in high-velocity / low-residence regions.
    _setdef("BIOCHEM_KINE_PRIOR_WEIGHT", "10.0")
    _setdef("BIOCHEM_KINE_PRIOR_RAMP_EPOCHS", ramp)
    # Localise the kinematics prior with the v2 physics-derived formulation
    # (see ``src/core_physics/kinematics_clot_prior.py``): smooth
    # ``exp(-sdf_nd / lambda_w)`` boundary-layer gate on wall distance / stagnation.
    # ``BIOCHEM_PRIOR_BULK_SCALE`` is silently ignored by v2 (kept here only as
    # a no-op back-compat tombstone for older run configurations).
    _setdef("BIOCHEM_PRIOR_WALL_DECAY_ND", "0.006")
    _setdef("BIOCHEM_PRIOR_MIN_FLOOR", "1e-4")
    _setdef("BIOCHEM_PRIOR_W_PATHOLOGICAL", "1.0")
    _setdef("BIOCHEM_PRIOR_W_STAGNATION", "0.25")
    _setdef("BIOCHEM_PRIOR_STAGNATION_POWER", "1.5")
    _setdef("BIOCHEM_PRIOR_TOTAL_POWER", "1.5")
    _setdef("BIOCHEM_BIO_ENCODER_PRIOR_DIM", "2")
    _setdef("BIOCHEM_COMSOL_TEMPORAL_WEIGHT", "0.012")
    # Use a wider default rollout window so teacher batches see beyond t0->t1.
    _setdef("BIOCHEM_TBPTT_MAX_WINDOW", "15")
    _setdef("BIOCHEM_TEACHER_TBPTT_RANDOM_ANCHOR", "1")
    _setdef("BIOCHEM_TEACHER_ADAPTIVE_THRESHOLD_FLOOR_SCALE", "1.0")
    _setdef("BIOCHEM_TEACHER_CURRICULUM_BUFFER", "4")
    # Ground-truth teacher-forcing span before ramping toward autoregressive rollout (clamped to max_epochs in teacher stage).
    _setdef("BIOCHEM_TEACHER_TF_WARMUP_EPOCHS", "5")
    _setdef("BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT", "5.0")
    _setdef("BIOCHEM_MU_SI_ANCHOR_AUX_EARLY_EPOCHS", "4")
    _setdef("BIOCHEM_MU_SI_ANCHOR_AUX_EARLY_MULT", "2.0")
    _setdef("BIOCHEM_FI_GATE_START_EPOCHS", "6")
    _setdef("BIOCHEM_FI_GATE_START_WEIGHT", "0.0")
    _setdef("BIOCHEM_TEACHER_MU_RATIO_MAX", "80.0")
    # Compare against μ₂ magnitude (~mu_ratio_max when FI saturates); keep ε visible on [0, mu_ratio_max].
    _setdef("BIOCHEM_FI_GATE_START_EPS", "0.15")
    _setdef("BIOCHEM_TEACHER_ODE_FREEZE_EPOCHS", "3")
    # Overfit-debug default: start Data_Bio with stronger magnitude so biology
    # updates are visible immediately in one-anchor teacher runs.
    _setdef("BIOCHEM_RAW_BIO_MAGNITUDE", "0.01")
    # Fail fast when the first teacher forward is pathological vs GT (scale/ODE blow-up).
    _setdef("BIOCHEM_ABORT_BAD_TEACHER_INIT", "1")
    # Preflight uses continuous μ error on COMSOL truth nodes (t0→t1, TF=1), not threshold fractions.
    _setdef("BIOCHEM_PREFLIGHT_ABORT_MEDIAN_LOG_MAE", "2.5")
    _setdef("BIOCHEM_PREFLIGHT_ABORT_WORST_LOG_MAE", "4.0")
    _setdef("BIOCHEM_MICRO_HEAD_ZERO_INIT", "1")
    _setdef("BIOCHEM_RESIDUAL_SPARSE_LAMBDA_START", "12.0")
    _setdef("BIOCHEM_RESIDUAL_SPARSE_LAMBDA_END", "0.5")
    _setdef("BIOCHEM_RESIDUAL_SPARSE_RAMP_EPOCHS", str(max(8, max_epochs // 2)))
    print(
        "🧷 Teacher-stage defaults applied (COMSOL forcing + PDE cap + μ regression). "
        "Unset any var to inherit; skip entirely with BIOCHEM_NO_TEACHER_DEFAULTS=1 "
        "(default from PyCharm/simplest preset)."
    )


def _biochem_metrics_jsonl_path():
    run_dir = os.environ.get("KINEMATICS_TRAINING_RUN_DIR", "").strip()
    if run_dir:
        return Path(run_dir) / "metrics.jsonl"
    return reports_training_dir("biochem") / "metrics.jsonl"


def _biochem_append_jsonl(record: Dict[str, Any]) -> None:
    path = _biochem_metrics_jsonl_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except OSError:
        pass


def _graph_has_anchor_nodes(data) -> bool:
    """True if this graph carries any COMSOL-matched (patient/anchor) nodes."""
    ia = getattr(data, "is_anchor", None)
    if ia is None:
        return False
    if torch.is_tensor(ia):
        return bool(ia.any().item())
    return bool(ia)


# --- Optional diagnostics (compare with f958b74 ~2026-04-03: that revision mutated
# ``phys_cfg.re_target`` inside ``compute_biochem_loss``; current code keeps config fixed
# and passes ``re_ref`` from ``data.re_actual`` only into ``navier_stokes_residual``.)
def _biochem_debug_enabled() -> bool:
    return (os.environ.get("BIOCHEM_DEBUG", "0") or "").strip().lower() in ("1", "true", "yes", "on")


def _biochem_debug_batches_cap() -> int:
    try:
        return max(0, int(os.environ.get("BIOCHEM_DEBUG_BATCHES", "1")))
    except ValueError:
        return 1


def _biochem_should_log_batch(epoch: int, batch_idx: int) -> bool:
    if not _biochem_debug_enabled():
        return False
    return batch_idx < _biochem_debug_batches_cap()


_clot_batch_trace_emitted = False


def _per_node_mean_abs_edge_diff(
    nodal: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Graph ``Δ`` proxy: per-node mean |f_i - f_j| over incident edges (undirected ok)."""
    row = edge_index[0]
    col = edge_index[1]
    ediff = (nodal[row] - nodal[col]).abs()
    deg = torch.zeros(num_nodes, device=nodal.device, dtype=nodal.dtype)
    acc = torch.zeros(num_nodes, device=nodal.device, dtype=nodal.dtype)
    ones = torch.ones_like(ediff)
    deg.index_add_(0, row, ones)
    deg.index_add_(0, col, ones)
    acc.index_add_(0, row, ediff)
    acc.index_add_(0, col, ediff)
    return acc / deg.clamp(min=1.0)


def _biochem_trace_clot_batch_enabled() -> bool:
    return (os.environ.get("BIOCHEM_TRACE_CLOT_BATCH", "0") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _emit_clot_batch_trace(
    *,
    truth_mask: torch.Tensor,
    prior: torch.Tensor,
    mu_p_si: torch.Tensor,
    mu_g_si: torch.Tensor,
    delta_fi: torch.Tensor,
) -> None:
    """One-shot stdout + debug.log: FI spatial variation vs kinematics prior (``BIOCHEM_TRACE_CLOT_BATCH=1``)."""
    global _clot_batch_trace_emitted
    if _clot_batch_trace_emitted or not _biochem_trace_clot_batch_enabled():
        return
    m = truth_mask
    if not m.any():
        return
    _clot_batch_trace_emitted = True

    def _stat1d(t: torch.Tensor) -> str:
        x = t[m].detach().float().reshape(-1)
        if x.numel() == 0:
            return "(empty)"
        return (
            f"mean={x.mean().item():.4f} std={x.std(unbiased=False).item():.4f} "
            f"min={x.min().item():.4f} max={x.max().item():.4f}"
        )

    df = delta_fi[m].detach().float().reshape(-1)
    pr = prior[m].detach().float().reshape(-1)
    corr = torch.tensor(float("nan"), device=df.device)
    if df.numel() > 2 and df.std(unbiased=False) > 1e-8 and pr.std(unbiased=False) > 1e-8:
        df0 = df - df.mean()
        pr0 = pr - pr.mean()
        corr = (df0 * pr0).mean() / (df.std(unbiased=False) * pr.std(unbiased=False) + 1e-8)

    mu_p = mu_p_si[m].detach().float().reshape(-1)
    mu_g = mu_g_si[m].detach().float().reshape(-1)
    w_slow = (1.0 - pr).clamp(0.0, 1.0)
    l1_reg = (df * w_slow).mean().item()

    lines = [
        "📊 [BIOCHEM_TRACE_CLOT_BATCH] one anchor batch (first hit):",
        f"   anchors={int(m.sum().item())}",
        f"   |ΔFI| (edge-mean abs increment, SI): {_stat1d(delta_fi)}",
        f"   kinematics_prior: {_stat1d(prior)}",
        f"   Pearson(|ΔFI|,prior|anchors)={float(corr):.4f}  mean(|ΔFI|⊙(1-prior))={l1_reg:.4e}",
        f"   mu_pred_si(anchors): mean={mu_p.mean().item():.4e} p90={torch.quantile(mu_p, 0.9).item():.4e}",
        f"   mu_gt_si(anchors):   mean={mu_g.mean().item():.4e} p90={torch.quantile(mu_g, 0.9).item():.4e}",
    ]
    for ln in lines:
        _biochem_dbg_line(ln)


def _compute_mu_flow_debug_metrics(
    *,
    model,
    data,
    pred_final: torch.Tensor,
    target_final_mu_nd: Optional[torch.Tensor],
    phys_cfg,
    truth_mask: torch.Tensor,
    mu_ch: int,
) -> Dict[str, float]:
    """Strong μ diagnostics + simple inlet/outlet flow sanity (non-destructive debug only)."""
    out: Dict[str, float] = {}
    species_log = torch.clamp(pred_final[:, 4:16], min=-10.0, max=8.0)
    species_si = model.species_log_nd_to_si(species_log)
    fi_si = species_si[:, 8:9]
    mat_si = species_si[:, 11:12]

    mu1 = model.mu1_sigmoid(mat_si) if hasattr(model, "mu1_sigmoid") else torch.zeros_like(fi_si)
    mu2 = model.mu2_sigmoid(fi_si) if hasattr(model, "mu2_sigmoid") else torch.zeros_like(fi_si)
    learned = (
        model.learned_clot_penalty(species_log)
        if hasattr(model, "learned_clot_penalty")
        else torch.zeros_like(fi_si)
    )
    gel_total = mu1 + mu2 + learned

    mu_pred_si = phys_cfg.viscosity_nd_to_si(pred_final[:, mu_ch]).reshape(-1).float()
    m_all = torch.ones_like(mu_pred_si, dtype=torch.bool)
    m_truth = truth_mask.view(-1).bool().to(mu_pred_si.device)
    m_wall = (
        data.mask_wall.view(-1).bool().to(mu_pred_si.device)
        if hasattr(data, "mask_wall") and data.mask_wall is not None
        else torch.zeros_like(m_truth)
    )

    def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> float:
        if not bool(m.any().item()):
            return float("nan")
        return float(x[m].mean().item())

    def _masked_q(x: torch.Tensor, m: torch.Tensor, q: float) -> float:
        if not bool(m.any().item()):
            return float("nan")
        return float(torch.quantile(x[m], q).item())

    out["DBG_mu1_mean"] = _masked_mean(mu1.view(-1).float(), m_all)
    out["DBG_mu2_mean"] = _masked_mean(mu2.view(-1).float(), m_all)
    out["DBG_mu_learned_mean"] = _masked_mean(learned.view(-1).float(), m_all)
    out["DBG_mu_gel_total_mean"] = _masked_mean(gel_total.view(-1).float(), m_all)
    out["DBG_mu_pred_si_mean"] = _masked_mean(mu_pred_si, m_all)
    out["DBG_mu_pred_si_p90"] = _masked_q(mu_pred_si, m_all, 0.9)
    out["DBG_mu_pred_si_mean_truth"] = _masked_mean(mu_pred_si, m_truth)
    out["DBG_mu_pred_si_mean_wall"] = _masked_mean(mu_pred_si, m_truth & m_wall)

    if target_final_mu_nd is not None and bool(m_truth.any().item()):
        mu_gt_si = phys_cfg.viscosity_nd_to_si(target_final_mu_nd).reshape(-1).float().to(mu_pred_si.device)
        mp = mu_pred_si[m_truth].clamp(min=1e-8)
        mg = mu_gt_si[m_truth].clamp(min=1e-8)
        out["DBG_mu_log_mae_final_anchor"] = float((torch.log(mp) - torch.log(mg)).abs().mean().item())
        out["DBG_mu_gt_si_mean_truth"] = float(mu_gt_si[m_truth].mean().item())
        out["DBG_mu_gt_si_p90_truth"] = float(torch.quantile(mu_gt_si[m_truth], 0.9).item())
    else:
        out["DBG_mu_log_mae_final_anchor"] = float("nan")
        out["DBG_mu_gt_si_mean_truth"] = float("nan")
        out["DBG_mu_gt_si_p90_truth"] = float("nan")

    if hasattr(model, "_last_mu_trigger_gate") and model._last_mu_trigger_gate is not None:
        gate = model._last_mu_trigger_gate.view(-1).float()
        out["DBG_gate_mean_all"] = _masked_mean(gate, m_all)
        out["DBG_gate_mean_wall"] = _masked_mean(gate, m_truth & m_wall)
        if target_final_mu_nd is not None and bool(m_truth.any().item()):
            mu_gt_si = (
                phys_cfg.viscosity_nd_to_si(target_final_mu_nd).reshape(-1).float().to(mu_pred_si.device)
            )
            if mu_gt_si[m_truth].numel() > 0:
                high_thr = torch.quantile(mu_gt_si[m_truth], 0.9)
                m_high = m_truth & (mu_gt_si >= high_thr)
                out["DBG_gate_mean_clot"] = _masked_mean(gate, m_high)
            else:
                out["DBG_gate_mean_clot"] = float("nan")
        else:
            out["DBG_gate_mean_clot"] = float("nan")

    # Inlet volume flux vs FD prescription (Re = phys_cfg.re_target, Uav = u_ref × width).
    u = pred_final[:, 0].float()
    v = pred_final[:, 1].float()
    speed = torch.sqrt(u * u + v * v + 1e-20)
    out["DBG_speed_mean"] = float(speed.mean().item())
    vel_xy = torch.stack([u, v], dim=1)
    try:
        flux_dbg = flux_debug_from_graph_data(data, vel_xy, phys_cfg=phys_cfg)
        out.update(flux_debug_to_training_metrics(flux_dbg))
    except Exception as exc:
        out["DBG_flux_imbalance"] = float("nan")
        out["DBG_flux_debug_error"] = 1.0
        if _biochem_env_truthy("BIOCHEM_ZERO_LOSS_WARN", default=False):
            print(
                f"⚠️ flux_debug_from_graph_data failed ({type(exc).__name__}: {exc}); "
                "μ training continues.",
                flush=True,
            )
    return out


def _biochem_debug_log_path():
    run_dir = os.environ.get("KINEMATICS_TRAINING_RUN_DIR", "").strip()
    if run_dir:
        return Path(run_dir) / "debug.log"
    return reports_training_dir("biochem") / "debug.log"


def _biochem_dbg_line(msg: str) -> None:
    """Stdout + append to ``<reports_dir>/biochem_debug.log`` (tqdm often obscures raw prints)."""
    print(msg, flush=True)
    if not _biochem_debug_enabled():
        return
    try:
        path = _biochem_debug_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except OSError:
        pass


def _tensor_stat(x: torch.Tensor) -> str:
    d = x.detach()
    if not d.numel():
        return "empty"
    finite = torch.isfinite(d)
    n_fin = int(finite.sum().item())
    if n_fin == 0:
        return "all non-finite"
    d2 = d[finite]
    return f"min={d2.min().item():.4g} max={d2.max().item():.4g} finite={n_fin}/{d.numel()}"


def _scalar_fin(name: str, t: Union[torch.Tensor, float]) -> Tuple[bool, float]:
    v = float(t.detach().item() if torch.is_tensor(t) else t)
    ok = math.isfinite(v)
    return ok, v


def _debug_kendall_terms(
    loss_weighter: DynamicLossWeighter,
    losses: List,
    task_active: List[bool],
) -> None:
    """Print per-task precision, raw loss, and contribution precision*L + log_var (matches weighter)."""
    names = [
        "ADR_F", "ADR_S", "W_Bio", "W_Phy", "Bio_IO", "NS_mom", "Data_Kine", "Data_Bio",
    ]
    min_lv = loss_weighter.per_task_min_log_var
    max_lv = loss_weighter.per_task_max_log_var
    loss_weighter.clamped_log_vars().detach()
    _biochem_dbg_line(
        "   [Kendall] prec=exp(-log_var); contrib=prec*L_raw+log_var (per-task). "
        "Data_* L_raw is the same tensor as metrics L_Data_* (before summing with other loss terms)."
    )
    _biochem_dbg_line("   [Kendall breakdown] task | active | L_raw | prec=exp(-lv) | lv | contrib=prec*L+lv")
    with torch.no_grad():
        for i, loss in enumerate(losses):
            ta_i = task_active[i]
            act = bool(ta_i.item()) if torch.is_tensor(ta_i) else bool(ta_i)
            if not act:
                _biochem_dbg_line(f"      {names[i]:9} | off")
                continue
            li = loss.detach().item() if torch.is_tensor(loss) else float(loss)
            lv = float(
                torch.clamp(
                    loss_weighter.log_vars[i].detach(), min=min_lv[i], max=max_lv[i]
                ).item()
            )
            prec = math.exp(-lv)
            contrib = prec * li + lv
            flag = "" if math.isfinite(contrib) and math.isfinite(li) else " **NON-FINITE**"
            _biochem_dbg_line(
                f"      {names[i]:9} | on  | L={li:.6e} | prec={prec:.4g} | lv={lv:.4f} | contrib={contrib:.6e}{flag}"
            )


def _debug_biochem_batch(
    *,
    epoch: int,
    batch_idx: int,
    data,
    pred_series: torch.Tensor,
    all_losses: list,
    task_active: list,
    loss_weighter: DynamicLossWeighter,
    loss_total: torch.Tensor,
    l_latent_reg: torch.Tensor,
    metrics: dict,
    re_ref: Optional[float],
    r_lo: float,
    r_hi: float,
    evaluation_times: torch.Tensor,
    start_idx: int,
    end_idx: int,
    truth_count: int,
    tbptt_info: Optional[Dict[str, Any]] = None,
    bio_cfg: Optional[Any] = None,
    training: bool = True,
) -> None:
    src = getattr(data, "_biochem_path", None) or getattr(data, "_biochem_source", None)
    n_times = int(evaluation_times.shape[0])
    n_steps = max(0, n_times - 1)
    _biochem_dbg_line(
        f"\n[BIOCHEM_DEBUG] epoch={epoch} batch={batch_idx} src={src!r} "
        f"nodes={int(data.num_nodes)} "
        f"graph_times={int(data.y.shape[0])} eval_times={n_times} ode_steps={n_steps} "
        f"truth_nodes={truth_count}"
    )
    if _biochem_env_truthy("BIOCHEM_LOSS_DATA_ONLY"):
        w_mu_dbg = float(metrics.get("W_MuSI_aux_eff") or 0.0)
        w_mul_dbg = float(metrics.get("W_MuLog_aux_eff") or 0.0)
        mu_note = (
            f"+ {w_mu_dbg:g}*L_MuSI_anchor + {w_mul_dbg:g}*L_MuLog_anchor"
            if (w_mu_dbg > 0.0 or w_mul_dbg > 0.0)
            else "(no μ anchor aux: SI+LOG weights both 0)"
        )
        w_pt_dbg = float(metrics.get("W_DataOnlyPhysTemp_eff") or 0.0)
        pt_note = (
            f"+ {w_pt_dbg:g} * L_PhysTemp (COMSOL d/dt Huber on anchors)"
            if w_pt_dbg > 0.0
            else "(no temporal anchor term: unset BIOCHEM_DATA_ONLY_PHYS_TEMP or w_pt=0)"
        )
        _biochem_dbg_line(
            "   NOTE: BIOCHEM_LOSS_DATA_ONLY=1 — backward uses L_Data_Kine+L_Data_Bio "
            f"{mu_note} {pt_note} on anchors (or pseudo MSE); "
            "Kendall table below is diagnostic, not summed into loss_total."
        )
    if tbptt_info:
        cap = tbptt_info.get("cap")
        wlen = tbptt_info.get("window_len")
        cur = tbptt_info.get("curriculum")
        wcap = tbptt_info.get("window_cap")
        cur_s = "on (ramps min(5+epoch//4, cap))" if cur else "off (window=min(cap, graph_cap))"
        parts = [
            f"TBPTT: y_slice=[{tbptt_info.get('start_idx')}:{tbptt_info.get('end_idx')}) "
            f"len={wlen} BIOCHEM_TBPTT_MAX_WINDOW={cap} graph_cap={wcap} BIOCHEM_TBPTT_WINDOW_CURRICULUM={cur_s}",
        ]
        if "t_si_start" in tbptt_info and "t_si_end" in tbptt_info:
            parts.append(
                f"data.t_si≈[{tbptt_info['t_si_start']:.5g},{tbptt_info['t_si_end']:.5g}]s "
                f"(window endpoints)"
            )
        _biochem_dbg_line("   " + " | ".join(parts))
    elif not training:
        _biochem_dbg_line(
            f"   TBPTT: n/a (eval); full trajectory eval_times={n_times} "
            f"y=[0:{int(data.y.shape[0])})"
        )
    else:
        _biochem_dbg_line(
            f"   TBPTT: n/a (training, T_y≤2 or no slice); eval_times len={n_times}"
        )
    _biochem_dbg_line(
        f"   scales: u_ref={data.u_ref!r} d_bar={data.d_bar!r} "
        f"re_actual={getattr(data, 're_actual', None)!r} "
        f"re_ref_for_NS={re_ref!r} get_re(u,d)_range=[{r_lo:.4g},{r_hi:.4g}]"
    )
    if hasattr(data, "t") and data.t is not None:
        _biochem_dbg_line(
            f"   data.t: shape={tuple(data.t.shape)} min={data.t.min().item():.6g} max={data.t.max().item():.6g} [s]"
        )
    te = evaluation_times.detach().cpu()
    dt = te[1:] - te[:-1] if te.numel() > 1 else te
    t_ref = float(getattr(bio_cfg, "t_final", float("nan"))) if bio_cfg is not None else float("nan")
    dt_note = f"Δ_nd_min={(dt.min().item() if dt.numel() else float('nan')):.6g}"
    if math.isfinite(t_ref) and t_ref > 0:
        span_si = (te.max().item() - te.min().item()) * t_ref
        dt_note += f" | approx_Δt_si_min={(dt.min().item() * t_ref if dt.numel() else float('nan')):.6g}s window_span_si≈{span_si:.5g}s"
    _biochem_dbg_line(
        f"   eval_times_nd: min={te.min().item():.6g} max={te.max().item():.6g} {dt_note}"
    )
    raw_mag = max(float(os.environ.get("BIOCHEM_RAW_BIO_MAGNITUDE", "0.01")), 1e-12)
    _biochem_dbg_line(
        f"   supervised: L_Data_Bio={metrics.get('L_Data_Bio')} "
        f"L_Data_Kine={metrics.get('L_Data_Kine')} "
        f"(Huber; bio term scaled by 1/BIOCHEM_RAW_BIO_MAGNITUDE={raw_mag:g})"
    )
    ps = pred_series.detach()
    _biochem_dbg_line(f"   pred_series last step: {_tensor_stat(ps[-1])}")
    bad = []
    loss_names = [
        "L_ADR_F", "L_ADR_S", "L_W_Bio", "L_W_Phy", "L_B_IO", "L_mom", "L_Data_Kine", "L_Data_Bio",
    ]
    for j, ln in enumerate(loss_names):
        ok, v = _scalar_fin(ln, all_losses[j])
        if not ok:
            bad.append((ln, v))
    if bad:
        _biochem_dbg_line(f"   ** non-finite raw losses: {bad}")
    _biochem_dbg_line(
        f"   metrics TF_eff={metrics.get('TF_eff')} L_Latent_Reg={metrics.get('L_Latent_Reg')}"
    )
    lt = loss_total.detach().item()
    latent_scale = float(os.environ.get("BIOCHEM_LATENT_REG_SCALE", "1e-3"))
    lr = (
        (latent_scale * l_latent_reg).detach().item()
        if torch.is_tensor(l_latent_reg)
        else latent_scale * float(l_latent_reg)
    )
    if _biochem_env_truthy("BIOCHEM_LOSS_DATA_ONLY"):
        w_mu_dbg = float(metrics.get("W_MuSI_aux_eff") or 0.0)
        w_mul_dbg = float(metrics.get("W_MuLog_aux_eff") or 0.0)
        parts_mu = []
        if w_mu_dbg > 0.0:
            parts_mu.append(f"W_MuSI_eff={w_mu_dbg:g}*L_MuSI")
        if w_mul_dbg > 0.0:
            parts_mu.append(f"W_MuLog_eff={w_mul_dbg:g}*L_MuLog")
        mu_extra = (" " + ", ".join(parts_mu) + " in total") if parts_mu else ""
        _biochem_dbg_line(
            f"   scalar loss (backprop, data-only): total={lt:.6e} finite={math.isfinite(lt)}{mu_extra}; "
            f"latent_term={lr:.6e} not in total"
        )
    else:
        _biochem_dbg_line(
            f"   loss_weighter()+latent: total={lt:.6e} latent_scale={latent_scale:.4g} "
            f"latent_term={lr:.6e} finite={math.isfinite(lt)}"
        )
    _debug_kendall_terms(loss_weighter, all_losses, task_active)


class PatientDataset(Dataset):
    def __init__(self, root, file_list):
        super().__init__(root, transform=None, pre_transform=None)
        self.file_list = file_list

    def len(self):
        return len(self.file_list)

    def get(self, idx):
        path = self.file_list[idx]
        data = torch.load(path, weights_only=False)
        data = infer_missing_schema(data, phase_hint="biochem")
        assert_graph_schema(data, expected_y_schema=(BIO_Y_SCHEMA,))
        # Provenance for BIOCHEM_DEBUG=1 (PyG allows extra attributes on Data).
        data._biochem_path = str(path)
        # Also keep a public attribute name; some code paths / collate behavior are
        # more reliable with non-private keys.
        data.biochem_path = str(path)
        return data


def _biochem_src_for_log(data) -> str:
    """Normalize graph provenance for console lines (PyG ``Batch`` may wrap strings in length-1 lists)."""
    src = getattr(data, "biochem_path", None) or getattr(data, "_biochem_path", None)
    if src is None:
        return "<unknown>"
    if isinstance(src, (list, tuple)) and len(src) == 1:
        return str(src[0])
    if isinstance(src, (list, tuple)):
        return ", ".join(str(x) for x in src[:3]) + (f", +{len(src) - 3} more" if len(src) > 3 else "")
    return str(src)


def _biochem_data_source_key(data) -> Optional[str]:
    """Resolve a stable source-path key for pseudo-label lookup/debug."""
    s = _biochem_src_for_log(data)
    return None if s == "<unknown>" else s


def remap_stage_a_encoder_to_corrector(
    kinematics_weight: torch.Tensor,
    target_weight_template: torch.Tensor,
) -> torch.Tensor:
    """
    Remap Phase-2 encoder input channels to Phase-3 layout.

    Phase-2 encoded input width is 63; Phase-3 is 64 because one channel was
    inserted in the "rest" block before uv/mu/wss priors. Preserve the prior
    channels by shifting the Phase-2 tail by +1.
    """
    if kinematics_weight.shape == target_weight_template.shape:
        return kinematics_weight

    new_weight = torch.zeros_like(target_weight_template)
    old_out = int(kinematics_weight.shape[0])
    new_out = int(target_weight_template.shape[0])
    copy_out = min(old_out, new_out)
    old_in = int(kinematics_weight.shape[1])
    new_in = int(target_weight_template.shape[1])

    if old_in == 63 and new_in == 64:
        # Keep shared prefix, reserve one inserted channel at index 59,
        # then shift uv/mu/wss-related tail by +1 to preserve semantics.
        new_weight[:copy_out, :59] = kinematics_weight[:copy_out, :59]
        new_weight[:copy_out, 60:64] = kinematics_weight[:copy_out, 59:63]
        return new_weight

    # Safe fallback for unexpected shape pairs.
    copy_in = min(old_in, new_in)
    new_weight[:copy_out, :copy_in] = kinematics_weight[:copy_out, :copy_in]
    return new_weight


def _filter_compatible_state_dict(
    source_state_dict: Dict[str, torch.Tensor],
    target_state_dict: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """
    Keep only checkpoint tensors whose key exists and shape matches target model.
    """
    compatible: Dict[str, torch.Tensor] = {}
    skipped: List[str] = []
    for key, value in source_state_dict.items():
        target_value = target_state_dict.get(key, None)
        if target_value is None:
            skipped.append(key)
            continue
        if tuple(value.shape) != tuple(target_value.shape):
            skipped.append(key)
            continue
        compatible[key] = value
    return compatible, skipped


BIOCHEM_TEACHER_BEST_CKPT_NAME = "biochem_teacher_best.pth"


def _save_biochem_teacher_best_checkpoint(
    path: Path,
    teacher: torch.nn.Module,
    *,
    teacher_best_mu_score: float,
    best_epoch: int = -1,
    run_note: str = "",
) -> None:
    """Persist best teacher weights for inference / ``visualize_pipeline.py``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    val_log_mae = (
        float(-teacher_best_mu_score)
        if teacher_best_mu_score > -float("inf")
        else float("nan")
    )
    payload = {
        "model_state_dict": teacher.state_dict(),
        "teacher_best_mu_score": float(teacher_best_mu_score),
        "val_mu_log_mae": val_log_mae,
        "best_epoch": int(best_epoch),
        "run_note": (run_note or "").strip(),
        "checkpoint_role": "teacher_best",
    }
    torch.save(payload, path)
    ep_note = f" epoch={best_epoch:02d}" if best_epoch >= 0 else ""
    print(
        f"💾 Saved teacher-best checkpoint -> {path} "
        f"(val logMAE≈{val_log_mae:.4f}, mu_score={teacher_best_mu_score:.4f}{ep_note})",
        flush=True,
    )


def _try_load_biochem_post_pretrain(model: torch.nn.Module, path: Path, device: torch.device) -> bool:
    """Load ``biochem_post_pretrain.pth`` (full ``state_dict``) for warm-start; shape-filtered."""
    if not path.is_file():
        return False
    try:
        raw = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        raw = torch.load(path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state = raw["model_state_dict"]
    else:
        state = raw
    compatible, skipped = _filter_compatible_state_dict(state, model.state_dict())
    if not compatible:
        print(f"⚠️ Post-pretrain checkpoint {path.name} had no compatible parameter keys.")
        return False
    model.load_state_dict(compatible, strict=False)
    if skipped:
        print(f"   ℹ️ Post-pretrain load: skipped {len(skipped)} incompatible/shape-mismatch keys.")
    return True


def load_dataset():
    cfg_anchors = VesselConfig(phase="biochem_anchors")
    cfg_synthetic = VesselConfig(phase="biochem")

    anchor_dir = cfg_anchors.graph_output_dir
    synthetic_dir = cfg_synthetic.graph_output_dir

    anchor_files = sorted(list(anchor_dir.glob("*.pt"))) if anchor_dir.exists() else []
    synthetic_files = sorted(list(synthetic_dir.glob("*.pt"))) if synthetic_dir.exists() else []

    if not anchor_files and not synthetic_files:
        print(
            f"No Biochem graphs found in {anchor_dir} or {synthetic_dir}. "
            f"Please generate/extract Biochem data first."
        )
        return []

    file_list = anchor_files + synthetic_files
    max_load_raw = os.environ.get("BIOCHEM_MAX_LOAD_VESSELS", "").strip()
    if max_load_raw:
        try:
            max_load = max(1, int(max_load_raw))
        except ValueError:
            max_load = 0
        if max_load > 0 and len(file_list) > max_load:
            shuffle_before_cap = os.environ.get("BIOCHEM_MAX_LOAD_SHUFFLE", "1").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            if shuffle_before_cap:
                rng = random.Random(42)
                rng.shuffle(file_list)
            file_list = file_list[:max_load]
            print(
                f"✂️ Pre-load cap active (BIOCHEM_MAX_LOAD_VESSELS={max_load}): "
                f"using {len(file_list)} graph files before split."
            )
    print(
        f"📂 Found {len(anchor_files)} Biochem anchor graphs + "
        f"{len(synthetic_files)} Biochem synthetic graphs for lazy loading..."
    )

    # PyG ``Dataset`` requires a ``root``; loads use absolute paths in ``file_list``.
    return PatientDataset(root=str(data_root()), file_list=file_list)


def _print_biochem_anchor_file_split(train_anchors: List[Any], val_anchors: List[Any]) -> None:
    """Log anchor ``.pt`` basenames in train vs val so you can confirm disjoint splits."""
    if not train_anchors and not val_anchors:
        return
    ta = sorted({Path(p).name for p in train_anchors})
    va = sorted({Path(p).name for p in val_anchors})
    overlap = sorted(set(ta) & set(va))

    def _fmt(names: List[str], cap: int = 10) -> str:
        if len(names) <= cap:
            return ", ".join(names) if names else "(none)"
        return ", ".join(names[:cap]) + f", ... +{len(names) - cap} more"

    print(
        f"📂 Anchor train/val file lists: train_n={len(ta)} [{_fmt(ta)}] | "
        f"val_n={len(va)} [{_fmt(va)}] | overlap_basenames={len(overlap)}"
    )
    # ``ta`` is sorted for readability; DataLoader uses ``train_anchors`` list order (shuffle split).
    try:
        tr_paths = [Path(p) for p in train_anchors]
        order = ", ".join(f"{i}:{p.name}" for i, p in enumerate(tr_paths))
        print(f"   🧭 Teacher/DataLoader anchor order (batch_idx → file): {order}")
    except (TypeError, ValueError):
        pass
    if ta == va and ta:
        print(
            "   ⚠️ Train and val anchor **file lists are identical** (same .pt paths in both splits). "
            "Common with a single anchor graph or low-anchor mode — validation is not an independent holdout."
        )
    elif overlap:
        print(
            f"   ℹ️ Overlap (names appearing in both splits): {_fmt(overlap, 12)} — unusual unless "
            "paths are duplicated or the split logic was overridden."
        )


def initialize_biochem_priors(model):
    print("🧬 Injecting physical priors into biochemistry decoder biases...")
    target_layer = model.biochem_decoder.linear if hasattr(model.biochem_decoder, 'linear') else model.biochem_decoder

    # Micro-head init: default is small random weights (gradients flow); set
    # BIOCHEM_MICRO_HEAD_ZERO_INIT=1 for legacy exact-zero residual map at start.
    zero_init = (os.environ.get("BIOCHEM_MICRO_HEAD_ZERO_INIT", "0") or "").strip().lower() in (
        "1", "true", "yes", "on"
    )
    if zero_init:
        torch.nn.init.zeros_(target_layer.weight)
    else:
        torch.nn.init.normal_(target_layer.weight, std=1e-4)

    bias_vals = torch.zeros(12, dtype=torch.float32)
    # COMSOL-consistent resting blood chemistry in model ND/log1p space:
    # RP=1.0, AP=0.05, PT=1.0, AT=1.0, FG=1.0, others=0.0.
    # This preserves "delta learning" while matching initc semantics.
    bias_vals[0] = math.log1p(1.0)   # RP
    bias_vals[1] = math.log1p(0.05)  # AP
    bias_vals[4] = math.log1p(1.0)   # PT
    bias_vals[6] = math.log1p(1.0)   # AT
    bias_vals[7] = math.log1p(1.0)   # FG

    # Apply the biases
    with torch.no_grad():
        target_layer.bias.copy_(bias_vals)
    if zero_init:
        print("   micro-head init: zero weights + COMSOL resting-state bias vector.")
    else:
        print("   micro-head init: near-zero random weights + COMSOL resting-state bias vector.")

    print("🛑 Initializing ODE function to near-zero derivative...")

    def _init_linear_like_near_zero(module, eps=1e-3):
        linear = module.linear if hasattr(module, 'linear') else module
        if not isinstance(linear, torch.nn.Linear):
            return
        weight = getattr(linear, 'weight_orig', None)
        if weight is None:
            weight = linear.weight
        torch.nn.init.uniform_(weight, a=-eps, b=eps)
        if linear.bias is not None:
            torch.nn.init.zeros_(linear.bias)

    # Target terminal projection layers in the ODE network so dz/dt starts near zero.
    terminal_layers = []
    for name, module in model.ode_func.named_modules():
        if not isinstance(module, (torch.nn.Linear, type(model.biochem_decoder))):
            continue

        has_linear_child = any(
            child_name and isinstance(child, (torch.nn.Linear, type(model.biochem_decoder)))
            for child_name, child in module.named_modules()
        )
        if not has_linear_child:
            terminal_layers.append((name, module))

    # Fallback safety: if architecture inspection misses terminals, damp all ODE linear projections.
    if not terminal_layers:
        terminal_layers = [
            (name, module)
            for name, module in model.ode_func.named_modules()
            if isinstance(module, (torch.nn.Linear, type(model.biochem_decoder)))
        ]

    with torch.no_grad():
        for _, layer in terminal_layers:
            _init_linear_like_near_zero(layer)
        if hasattr(model.ode_func, 'derivative_scale'):
            model.ode_func.derivative_scale.fill_(1e-1)


def make_biochem_dynamic_loss_weighter(curriculum: CurriculumConfig, device) -> DynamicLossWeighter:
    """Per-task Kendall bounds: cap physics weights, floor supervised data weights."""
    # Hard cap the physics precision so PDE terms cannot be effectively muted.
    phys_ceiling = max(float(curriculum.biochem_physics_precision_ceiling), 1e-6)
    data_floor = max(float(curriculum.biochem_data_precision_floor), 1e-12)
    adr_s_floor = max(float(curriculum.biochem_adr_s_precision_floor), 1e-12)
    w_phys_floor = max(float(curriculum.biochem_w_phys_precision_floor), 1e-12)
    phys_min_lv = -math.log(phys_ceiling)
    data_max_lv = -math.log(data_floor)
    adr_s_max_lv = -math.log(adr_s_floor)
    w_phys_max_lv = -math.log(w_phys_floor)
    # 0–5: ADR_F, ADR_S, W_Bio, W_Phy, Bio_IO, NS_mom — 6–7: supervised Data_Kine, Data_Bio
    min_lv = [phys_min_lv] * 6 + [-8.0, -8.0]
    max_lv = [
        float("inf"),      # ADR_F
        adr_s_max_lv,      # ADR_S
        float("inf"),      # W_Bio
        w_phys_max_lv,     # W_Phy
        float("inf"),      # Bio_IO
        float("inf"),      # NS_mom
        data_max_lv,       # Data_Kine
        data_max_lv,       # Data_Bio
    ]
    print(
        f"⚖️ Biochem loss weighter: physics prec ≤ {phys_ceiling:g} (log_var ≥ {phys_min_lv:.3f}), "
        f"ADR_S prec ≥ {adr_s_floor:g} (log_var ≤ {adr_s_max_lv:.3f}), "
        f"W_Phys prec ≥ {w_phys_floor:g} (log_var ≤ {w_phys_max_lv:.3f}), "
        f"data prec ≥ {data_floor:g} (log_var ≤ {data_max_lv:.3f}), "
        f"freeze_in_warmup={curriculum.biochem_weighter_freeze_during_warmup}"
    )
    return DynamicLossWeighter(num_losses=8, min_log_var=min_lv, max_log_var=max_lv).to(device)


def _scheduled_biochem_huber_delta(bio_cfg: BiochemConfig, epoch: int) -> float:
    start = max(float(bio_cfg.biochem_huber_delta), 1e-8)
    end = max(float(getattr(bio_cfg, "biochem_huber_delta_final", start)), 1e-8)
    warmup = max(0, int(getattr(bio_cfg, "biochem_huber_delta_warmup_epochs", 0)))
    anneal = max(1, int(getattr(bio_cfg, "biochem_huber_delta_anneal_epochs", 1)))
    if epoch <= warmup:
        return start
    p = min(1.0, max(0.0, (epoch - warmup) / float(anneal)))
    p = _ease01(p, "smoothstep")
    return start + (end - start) * p


def _scheduled_residual_sparse_lambda(epoch: int, total_epochs: int) -> float:
    """Anneal residual sparsity regularization from strong -> permissive."""
    lam_start = max(float(os.environ.get("BIOCHEM_RESIDUAL_SPARSE_LAMBDA_START", "12.0")), 0.0)
    lam_end = max(float(os.environ.get("BIOCHEM_RESIDUAL_SPARSE_LAMBDA_END", "0.5")), 0.0)
    ramp_epochs_env = os.environ.get("BIOCHEM_RESIDUAL_SPARSE_RAMP_EPOCHS")
    if ramp_epochs_env is None or str(ramp_epochs_env).strip() == "":
        ramp_epochs = max(1, int(total_epochs // 2))
    else:
        ramp_epochs = max(1, int(ramp_epochs_env))
    p = min(1.0, max(0.0, float(epoch) / float(ramp_epochs)))
    p = _ease01(p, "smoothstep")
    return lam_start + (lam_end - lam_start) * p


def _validation_eval_times(data, bio_cfg, device) -> torch.Tensor:
    """Validation-time rollout grid with optional temporal decimation for speed."""
    full_times = to_t_nd(bio_cfg.resolve_biochem_times(data, device), bio_cfg.t_final)
    stride = max(1, int(os.environ.get("BIOCHEM_VAL_TIME_STRIDE", "1")))
    if stride <= 1 or full_times.numel() <= 2:
        return full_times
    decimated = full_times[::stride]
    # Ensure the final timestamp is always included for final-state metrics.
    if decimated[-1].item() != full_times[-1].item():
        decimated = torch.cat([decimated, full_times[-1:].clone()], dim=0)
    return decimated


def _teacher_start_decay(epoch: int, hold_epochs: int) -> float:
    """1 -> 0 smooth decay over teacher-start stabilization window."""
    h = max(1, int(hold_epochs))
    p = min(1.0, max(0.0, float(epoch) / float(h)))
    p = _ease01(p, "smoothstep")
    return 1.0 - p


def _apply_dynamic_physics_precision_ceiling(
    loss_weighter: DynamicLossWeighter,
    curriculum: CurriculumConfig,
    epoch: int,
) -> float:
    """Ramp Kendall physics precision ceiling after data-head warmup."""
    lo = max(float(curriculum.biochem_physics_precision_ceiling_warmup), 1e-6)
    hi = max(float(curriculum.biochem_physics_precision_ceiling), lo)
    warmup_end = max(0, int(curriculum.biochem_warmup_epochs))
    ramp_epochs = max(1, int(curriculum.biochem_physics_precision_ramp_epochs))
    if epoch <= warmup_end:
        ceiling = lo
    else:
        p = min(1.0, max(0.0, (epoch - warmup_end) / float(ramp_epochs)))
        p = _ease01(p, curriculum.biochem_curriculum_easing)
        ceiling = lo + (hi - lo) * p
    min_lv = -math.log(ceiling)
    with torch.no_grad():
        loss_weighter.per_task_min_log_var[:6].fill_(min_lv)
    return ceiling


def inject_biochem_kinematic_lora(model: GNODE_Phase3, rank: int = 4, alpha: float = 1.0) -> None:
    """Attach LoRA to SpectralLinear layers in the kinematic stack (call before ``setup_biochem_optimization``)."""
    n_enc = inject_lora_to_spectral_linears(model.kin_encoder, rank=rank, alpha=alpha)
    n_proc = inject_lora_to_spectral_linears(model.kin_processor, rank=rank, alpha=alpha)
    n_dec = inject_lora_to_spectral_linears(model.kinematics_decoder, rank=rank, alpha=alpha)
    print(
        f"💉 LoRA injected (SpectralLinear count): kin_encoder={n_enc}, "
        f"kin_processor={n_proc}, kinematics_decoder={n_dec} "
        f"(rank={rank}, alpha={alpha}); plain nn.Linear modules contribute 0."
    )


class SplitBiochemOptimizers:
    """Small compatibility wrapper around separate Biochem optimizers."""

    def __init__(
        self,
        *,
        physics_optimizer: Optional[optim.Optimizer],
        mu_optimizer: Optional[optim.Optimizer],
        bio_optimizer: optim.Optimizer,
        weighter_optimizer: optim.Optimizer,
        physics_params: List[torch.nn.Parameter],
        mu_params: List[torch.nn.Parameter],
        bio_params: List[torch.nn.Parameter],
    ) -> None:
        self.physics_optimizer = physics_optimizer
        self.mu_optimizer = mu_optimizer
        self.bio_optimizer = bio_optimizer
        self.weighter_optimizer = weighter_optimizer
        self.physics_params = physics_params
        self.mu_params = mu_params
        self.bio_params = bio_params

    @property
    def param_groups(self):
        # Preserve existing call sites that log ``optimizer.param_groups[0]["lr"]``.
        return self.bio_optimizer.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self.physics_optimizer is not None:
            self.physics_optimizer.zero_grad(set_to_none=set_to_none)
        if self.mu_optimizer is not None:
            self.mu_optimizer.zero_grad(set_to_none=set_to_none)
        self.bio_optimizer.zero_grad(set_to_none=set_to_none)
        self.weighter_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        if self.physics_optimizer is not None:
            self.physics_optimizer.step()
        if self.mu_optimizer is not None:
            self.mu_optimizer.step()
        self.bio_optimizer.step()
        self.weighter_optimizer.step()

    def clip_and_step(
        self,
        *,
        physics_clip: float,
        bio_clip: float,
        weighter_clip: Optional[float] = None,
    ) -> Tuple[float, float]:
        physics_norm = 0.0
        mu_norm = 0.0
        bio_norm = 0.0
        if self.physics_params and physics_clip > 0.0:
            physics_norm = float(torch.nn.utils.clip_grad_norm_(self.physics_params, max_norm=physics_clip))
        if self.mu_params and bio_clip > 0.0:
            mu_norm = float(torch.nn.utils.clip_grad_norm_(self.mu_params, max_norm=bio_clip))
        if self.bio_params and bio_clip > 0.0:
            bio_norm = float(torch.nn.utils.clip_grad_norm_(self.bio_params, max_norm=bio_clip))
        if weighter_clip is not None and weighter_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(self.weighter_optimizer.param_groups[0]["params"], max_norm=weighter_clip)
        self.step()
        self.zero_grad()
        return physics_norm, max(bio_norm, mu_norm)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "type": "SplitBiochemOptimizers",
            "physics": self.physics_optimizer.state_dict() if self.physics_optimizer is not None else None,
            "mu": self.mu_optimizer.state_dict() if self.mu_optimizer is not None else None,
            "bio": self.bio_optimizer.state_dict(),
            "weighter": self.weighter_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if state_dict.get("type") == "SplitBiochemOptimizers":
            if self.physics_optimizer is not None and state_dict.get("physics") is not None:
                self.physics_optimizer.load_state_dict(state_dict["physics"])
            if self.mu_optimizer is not None and state_dict.get("mu") is not None:
                self.mu_optimizer.load_state_dict(state_dict["mu"])
            self.bio_optimizer.load_state_dict(state_dict["bio"])
            self.weighter_optimizer.load_state_dict(state_dict["weighter"])
            return
        print(
            "⚠️ Legacy single-optimizer checkpoint detected; optimizer state is not shape-compatible "
            "with split optimizers, so optimizer moments are reinitialized."
        )


def _try_load_biochem_split_optimizer(optimizer: SplitBiochemOptimizers, state_dict: Any) -> bool:
    """Load split-optimizer moments when checkpoint matches current param groups.

    Returns False when the checkpoint was built with a different trainable split (e.g. teacher stage
    uses ``freeze_lora=True`` → no physics Adam group, different bio tensor count vs corrector).
    """
    if not isinstance(state_dict, dict) or state_dict.get("type") != "SplitBiochemOptimizers":
        print(
            "⚠️ Checkpoint optimizer_state_dict is missing SplitBiochemOptimizers metadata; "
            "keeping freshly initialized optimizer moments."
        )
        return False
    try:
        optimizer.load_state_dict(state_dict)
        return True
    except Exception as exc:
        print(
            "⚠️ Could not load optimizer state from checkpoint (Adam param groups / tensors differ from "
            f"this run: {type(exc).__name__}: {exc}). Common when resuming after teacher-only training "
            "(``freeze_lora=True``) into the corrector (LoRA in a separate AdamW). "
            "Continuing with reinitialized optimizer moments."
        )
        return False


def _biochem_split_opt_grads_finite(split: SplitBiochemOptimizers) -> bool:
    """True if every populated grad tensor is finite (no NaN/Inf before ``clip_and_step``)."""
    for p in split.bio_params:
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return False
    for p in split.physics_params:
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return False
    for p in split.mu_params:
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return False
    for group in split.weighter_optimizer.param_groups:
        for p in group["params"]:
            if p.grad is not None and not torch.isfinite(p.grad).all():
                return False
    return True


def _biochem_nonfinite_grad_detail(
    model: torch.nn.Module, split: SplitBiochemOptimizers, *, max_names: int = 12
) -> str:
    """Comma-separated model param names (then weighter slots) with non-finite grads."""
    bad: List[str] = []
    for name, p in model.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            bad.append(name)
            if len(bad) >= max_names:
                break
    if len(bad) >= max_names:
        return ", ".join(bad) + ", ..."
    for gi, group in enumerate(split.weighter_optimizer.param_groups):
        for pi, p in enumerate(group["params"]):
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                bad.append(f"weighter[{gi}].p[{pi}]")
                if len(bad) >= max_names:
                    return ", ".join(bad) + ", ..."
    return ", ".join(bad) if bad else ""


def _biochem_bio_grad_global_l2(split: SplitBiochemOptimizers) -> float:
    """Unclipped L2 norm of concatenated bio-parameter gradients (0.0 if none)."""
    sq = 0.0
    for p in split.bio_params:
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        sq += float((g * g).sum().item())
    for p in split.mu_params:
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        sq += float((g * g).sum().item())
    return math.sqrt(sq) if sq > 0.0 else 0.0


def setup_biochem_optimization(model, loss_weighter, base_lr=1e-3, freeze_lora=False):
    print("❄️  Verifying Kinematic Backbone is Frozen.")
    print("🔥 Activating split optimizers: kinematic LoRA isolated from biochemistry.")

    # Set the frozen kinematic backbone to eval mode!
    model.kin_encoder.eval()
    model.kin_processor.eval()
    model.kinematics_decoder.eval()

    # Freeze everything by default to be absolutely safe
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze specifically intended modules conditionally
    for name, param in model.named_parameters():
        if "lora" in name.lower() and not freeze_lora:
            param.requires_grad = True

    for param in model.bio_encoder.parameters():
        param.requires_grad = True

    for param in model.ode_func.parameters():
        param.requires_grad = True

    for name, param in model.biochem_decoder.named_parameters():
        if 'lora' not in name.lower():
            param.requires_grad = True

    if hasattr(model, "learned_clot_penalty"):
        for param in model.learned_clot_penalty.parameters():
            param.requires_grad = True

    if hasattr(model, "mu_delta_head"):
        for param in model.mu_delta_head.parameters():
            param.requires_grad = True
    if hasattr(model, "mu_delta_bulk_head"):
        for param in model.mu_delta_bulk_head.parameters():
            param.requires_grad = True
    if hasattr(model, "mu_delta_tail_head"):
        for param in model.mu_delta_tail_head.parameters():
            param.requires_grad = True
    if hasattr(model, "mu_trigger_gate_head"):
        for param in model.mu_trigger_gate_head.parameters():
            param.requires_grad = True
    if hasattr(model, "mu_delta_wall_head"):
        for param in model.mu_delta_wall_head.parameters():
            param.requires_grad = True

    train_mu_encoder = _biochem_env_truthy("BIOCHEM_TRAIN_MU_ENCODER", default=False)
    if train_mu_encoder and hasattr(model, "mu_encoder"):
        for param in model.mu_encoder.parameters():
            param.requires_grad = True

    physics_params: List[torch.nn.Parameter] = []
    mu_params: List[torch.nn.Parameter] = []
    bio_params: List[torch.nn.Parameter] = []
    physics_names: List[str] = []
    mu_names: List[str] = []
    bio_names: List[str] = []
    use_mu_path_group = _biochem_env_truthy("BIOCHEM_USE_MU_PATH_GROUP", default=True)
    mu_group_prefixes = (
        "mu_encoder.",
        "learned_clot_penalty.",
        "mu_delta_head.",
        "mu_delta_bulk_head.",
        "mu_delta_tail_head.",
        "mu_trigger_gate_head.",
        "mu_delta_wall_head.",
    )
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora" in name.lower():
            physics_params.append(param)
            physics_names.append(name)
        elif use_mu_path_group and name.startswith(mu_group_prefixes):
            mu_params.append(param)
            mu_names.append(name)
        else:
            # ``ode_func`` is biochemical reaction dynamics, not fluid physics;
            # keep it with the bio stack so rare clot gradients are not starved.
            bio_params.append(param)
            bio_names.append(name)

    if not bio_params:
        raise RuntimeError("setup_biochem_optimization found no trainable biology parameters.")

    physics_lr = base_lr * float(os.environ.get("BIOCHEM_PHYSICS_LR_MULT", "0.3"))
    mu_lr = base_lr * float(os.environ.get("BIOCHEM_MU_PATH_LR_MULT", "1.0"))
    bio_lr = base_lr * float(os.environ.get("BIOCHEM_BIO_LR_MULT", "1.0"))
    mu_bulk_lr = mu_lr * float(os.environ.get("BIOCHEM_MU_BULK_LR_MULT", "1.0"))
    mu_tail_lr = mu_lr * float(os.environ.get("BIOCHEM_MU_TAIL_LR_MULT", "1.0"))
    mu_gate_lr = mu_lr * float(os.environ.get("BIOCHEM_MU_GATE_LR_MULT", "1.0"))
    mu_wall_lr = mu_lr * float(os.environ.get("BIOCHEM_MU_WALL_LR_MULT", "1.0"))
    print(
        f"   μ trainability: BIOCHEM_TRAIN_MU_ENCODER={int(train_mu_encoder)} "
        f"BIOCHEM_USE_MU_PATH_GROUP={int(use_mu_path_group)}"
    )
    print(
        f"   trainable groups: physics_lora={len(physics_params)} tensors (lr={physics_lr:.3e}), "
        f"mu_path={len(mu_params)} tensors (lr={mu_lr:.3e}), "
        f"biology={len(bio_params)} tensors (lr={bio_lr:.3e}); "
        f"loss_weighter lr=5.000e-02"
    )

    opt_physics = (
        optim.AdamW(physics_params, lr=physics_lr, weight_decay=1e-4)
        if physics_params
        else None
    )
    split_mu = _biochem_env_truthy("BIOCHEM_USE_SPLIT_MU_HEAD", default=False)
    if mu_params:
        if split_mu:
            mu_group_bulk: List[torch.nn.Parameter] = []
            mu_group_tail: List[torch.nn.Parameter] = []
            mu_group_gate: List[torch.nn.Parameter] = []
            mu_group_wall: List[torch.nn.Parameter] = []
            mu_group_rest: List[torch.nn.Parameter] = []
            for n, p in zip(mu_names, mu_params):
                if n.startswith("mu_delta_bulk_head."):
                    mu_group_bulk.append(p)
                elif n.startswith("mu_delta_tail_head."):
                    mu_group_tail.append(p)
                elif n.startswith("mu_trigger_gate_head."):
                    mu_group_gate.append(p)
                elif n.startswith("mu_delta_wall_head."):
                    mu_group_wall.append(p)
                else:
                    mu_group_rest.append(p)
            mu_param_groups: List[Dict[str, Any]] = []
            if mu_group_rest:
                mu_param_groups.append({"params": mu_group_rest, "lr": mu_lr})
            if mu_group_bulk:
                mu_param_groups.append({"params": mu_group_bulk, "lr": mu_bulk_lr})
            if mu_group_tail:
                mu_param_groups.append({"params": mu_group_tail, "lr": mu_tail_lr})
            if mu_group_gate:
                mu_param_groups.append({"params": mu_group_gate, "lr": mu_gate_lr})
            if mu_group_wall:
                mu_param_groups.append({"params": mu_group_wall, "lr": mu_wall_lr})
            print(
                f"   μ split-head lrs: base={mu_lr:.3e}, bulk={mu_bulk_lr:.3e}, "
                f"tail={mu_tail_lr:.3e}, gate={mu_gate_lr:.3e}, wall={mu_wall_lr:.3e}"
            )
            opt_mu = optim.AdamW(mu_param_groups, weight_decay=1e-5)
        else:
            opt_mu = optim.AdamW(mu_params, lr=mu_lr, weight_decay=1e-5)
    else:
        opt_mu = None
    opt_bio = optim.AdamW(bio_params, lr=bio_lr, weight_decay=1e-5)
    opt_weighter = optim.AdamW(loss_weighter.parameters(), lr=5e-2, weight_decay=0.0)
    return SplitBiochemOptimizers(
        physics_optimizer=opt_physics,
        mu_optimizer=opt_mu,
        bio_optimizer=opt_bio,
        weighter_optimizer=opt_weighter,
        physics_params=physics_params,
        mu_params=mu_params,
        bio_params=bio_params,
    )


def pretrain_autoencoder(
    model,
    loader,
    optimizer,
    device,
    kernels,
    epochs=5,
    ode_reaction_epochs=8,
    post_pretrain_save_path: Optional[Union[str, Path]] = None,
):
    if _biochem_env_truthy("BIOCHEM_SKIP_PRETRAIN", default=False):
        skip_ae = True
        skip_ode_rxn = True
    else:
        skip_ae = _biochem_env_truthy("BIOCHEM_SKIP_AE_PRETRAIN", default=False)
        skip_ode_rxn = _biochem_env_truthy("BIOCHEM_SKIP_ODE_RXN_PRETRAIN", default=False)

    prior_requires_grad = {name: p.requires_grad for name, p in model.named_parameters()}

    freeze_decoder = _biochem_env_truthy("BIOCHEM_FREEZE_DECODER_PRETRAIN", default=False)
    if freeze_decoder:
        for p in model.biochem_decoder.parameters():
            p.requires_grad = False
        if hasattr(model, "learned_clot_penalty"):
            for p in model.learned_clot_penalty.parameters():
                p.requires_grad = False
        if not (skip_ae and skip_ode_rxn):
            print(
                "   🔒 Decoder freeze: biochem_decoder + learned_clot_penalty fixed during AE + ODE-RXN "
                "(BIOCHEM_FREEZE_DECODER_PRETRAIN=1)."
            )

    for param in model.ode_func.parameters():
        param.requires_grad = False

    model.train()
    ae_scales = kernels.cfg.get_species_scales(device=device)[:12].view(1, 12)
    latent_reg_weight = 1e-4
    ae_min_epochs = max(1, int(os.environ.get("BIOCHEM_AE_MIN_EPOCHS", "8")))
    ae_patience = max(1, int(os.environ.get("BIOCHEM_AE_PATIENCE", "4")))
    ae_min_delta = max(float(os.environ.get("BIOCHEM_AE_MIN_DELTA", "1e-4")), 0.0)
    pretrain_log_interval = max(1, int(os.environ.get("BIOCHEM_PRETRAIN_LOG_INTERVAL", "5")))

    if skip_ae:
        print(
            "\n⏭️  Phase 3a AE skipped (BIOCHEM_SKIP_AE_PRETRAIN=1 or BIOCHEM_SKIP_PRETRAIN=1). "
            "Encoder/decoder weights unchanged from load."
        )
    else:
        print("\n🚀 --- Phase 3a: Autoencoder Pre-Training (Freezing ODE) ---")
        best_ae_loss = float("inf")
        ae_bad_epochs = 0
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0

            for data in loader:
                data = data.to(device)
                if not hasattr(data, "y") or data.y is None or data.y.shape[0] < 1 or data.y.shape[-1] < 16:
                    continue
                mask = biochem_truth_node_mask(data, int(data.x.shape[0]), device)
                if not mask.any():
                    continue

                optimizer.zero_grad()

                actual_num_steps = int(data.y.shape[0])
                ti = int(torch.randint(0, actual_num_steps, (1,), device=device).item())
                targ_species = data.y[ti, :, 4:16]
                targ_uvp = data.y[ti, :, :3]

                prior_tail = model._kinematics_prior_tail(data, targ_uvp[:, 0], targ_uvp[:, 1])
                if prior_tail is None:
                    bio_in = torch.cat([targ_species, targ_uvp, data.x[:, :15]], dim=-1)
                else:
                    bio_in = torch.cat([targ_species, targ_uvp, data.x[:, :15], prior_tail], dim=-1)

                z = model.bio_encoder(bio_in)
                pred_species = model._decode_species_log1p(model.biochem_decoder(z))

                pred_si = torch.expm1(pred_species[mask]) * ae_scales
                targ_si = torch.expm1(targ_species[mask]) * ae_scales
                pred_norm = pred_si / ae_scales
                targ_norm = targ_si / ae_scales

                recon_loss = F.huber_loss(pred_norm, targ_norm, delta=1.0)
                latent_reg = latent_reg_weight * torch.mean(z ** 2)
                loss = recon_loss + latent_reg
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            if num_batches == 0:
                print(
                    f"AE Epoch {epoch:02d}: skipped (no graphs with COMSOL-labeled nodes — check is_anchor / re-extract)."
                )
                continue
            avg_loss = total_loss / num_batches
            if (best_ae_loss - avg_loss) > ae_min_delta:
                best_ae_loss = avg_loss
                ae_bad_epochs = 0
            else:
                ae_bad_epochs += 1
            will_early_stop = (epoch + 1) >= ae_min_epochs and ae_bad_epochs >= ae_patience
            if _should_log_pretrain_epoch(epoch, epochs, pretrain_log_interval) or will_early_stop:
                print(f"AE Epoch {epoch:02d}: Recon Loss = {avg_loss:.4e}")
            if will_early_stop:
                print(
                    f"🛑 AE early stop at epoch {epoch:02d}: no improvement > {ae_min_delta:.1e} "
                    f"for {ae_bad_epochs} epoch(s) after min_epochs={ae_min_epochs}."
                )
                break

    if skip_ode_rxn:
        print(
            "\n⏭️  Phase 3a.5 ODE-RXN skipped (BIOCHEM_SKIP_ODE_RXN_PRETRAIN=1 or BIOCHEM_SKIP_PRETRAIN=1). "
            "ODE/decoder snapshot restore skipped; restoring parameter requires_grad."
        )
        for name, param in model.named_parameters():
            param.requires_grad = prior_requires_grad.get(name, True)
        return

    print("\n🧪 --- Phase 3a.5: ODE Reaction-Rate Imitation Pre-Training ---")
    for _, param in model.named_parameters():
        param.requires_grad = False
    for param in model.ode_func.parameters():
        param.requires_grad = True
    if hasattr(model, "learned_clot_penalty"):
        for p in model.learned_clot_penalty.parameters():
            p.requires_grad = False
    decoder_params: List[torch.nn.Parameter] = []
    if not freeze_decoder:
        for name, param in model.biochem_decoder.named_parameters():
            if "lora" not in name.lower():
                param.requires_grad = True
                decoder_params.append(param)

    base_lr = float(optimizer.param_groups[0].get("lr", 1e-3))
    ode_rxn_lr = base_lr * 1.0
    ode_rxn_params = list(model.ode_func.parameters())
    if decoder_params:
        ode_rxn_params = ode_rxn_params + decoder_params
    ode_rxn_optimizer = optim.AdamW(
        ode_rxn_params,
        lr=ode_rxn_lr,
        weight_decay=0.0,
    )
    on_frac = float(os.environ.get("BIOCHEM_ODE_MANIFOLD_FRAC", "1.0"))
    synth_jitter_frac = float(os.environ.get("BIOCHEM_ODE_SYNTH_JITTER_FRAC", "0.0"))
    max_reaction_batches = max(1, int(os.environ.get("BIOCHEM_ODE_MAX_REACTION_BATCHES", "32")))
    symlog_scale = max(float(os.environ.get("BIOCHEM_ODE_SYMLOG_SCALE", "0.25")), 1e-8)
    rate_clip = max(float(os.environ.get("BIOCHEM_ODE_TARGET_RATE_CLIP", "5.0")), 1.0)
    ode_clip = max(float(os.environ.get("BIOCHEM_ODE_CLIP_NORM", "0.5")), 1e-8)
    dec_clip = max(float(os.environ.get("BIOCHEM_ODE_DEC_CLIP_NORM", "1.0")), 1e-8)
    ode_ema_beta = min(max(float(os.environ.get("BIOCHEM_ODE_EMA_BETA", "0.9")), 0.0), 0.9999)
    ode_min_epochs = max(1, int(os.environ.get("BIOCHEM_ODE_MIN_EPOCHS", "20")))
    ode_patience = max(1, int(os.environ.get("BIOCHEM_ODE_PATIENCE", "8")))
    ode_min_delta = max(float(os.environ.get("BIOCHEM_ODE_MIN_DELTA", "1e-4")), 0.0)
    _rxnf = (os.environ.get("BIOCHEM_ODE_RXN_MIN_EPOCHS_BEFORE_PLATEAU", "") or "").strip()
    if _rxnf:
        ode_rxn_min_before_plateau = max(1, int(_rxnf))
    else:
        # Default: do not count early-stop patience until at least this many epochs
        # (avoids "best stays epoch 0" plateau exits after a handful of steps).
        ode_rxn_min_before_plateau = max(ode_min_epochs, 8)
    print(
        f"🧪 ODE-RXN optimizer: lr={ode_rxn_lr:.3e} (base_lr x1.0) | "
        f"on-manifold COMSOL species frac ≈ {on_frac:.2f} (BIOCHEM_ODE_MANIFOLD_FRAC) | "
        f"symlog_scale={symlog_scale:g} | rate_clip={rate_clip:g} | "
        f"patience_counts_after_epoch={ode_rxn_min_before_plateau} (BIOCHEM_ODE_RXN_MIN_EPOCHS_BEFORE_PLATEAU)"
    )

    # Species ordering must match kinetics.compute_species_reactions inputs.
    rxn_keys = ['RP', 'AP', 'APR', 'APS', 'PT', 'T', 'AT', 'FG', 'FI']
    scales = kernels.cfg.get_species_scales(device=device)[:9].view(1, 9)
    prev_rxn_avg = None
    plateau_streak = 0
    ema_rxn_loss = None
    best_ema_rxn = float("inf")
    best_raw_rxn = float("inf")
    best_epoch_idx = -1
    ode_bad_epochs = 0
    best_ode_state = {
        "ode_func": copy.deepcopy(model.ode_func.state_dict()),
        "biochem_decoder": copy.deepcopy(model.biochem_decoder.state_dict()),
    }
    last_ode_state: Optional[dict] = None
    last_raw_rxn = float("nan")

    model.train()
    for epoch in range(ode_reaction_epochs):
        total_loss = 0.0
        num_batches = 0

        for batch_idx, data in enumerate(loader):
            if batch_idx >= max_reaction_batches:
                break
            data = data.to(device)
            n_nodes = int(data.x.shape[0])
            if n_nodes == 0:
                continue

            # Mix physically plausible synthetic states with on-manifold COMSOL trajectory
            # samples so reaction targets stay near states the encoder sees in training.
            ti = 0
            use_manifold = (
                hasattr(data, "y")
                and data.y is not None
                and data.y.shape[-1] >= 13
                and data.y.shape[0] >= 1
                and torch.rand(1, device=device).item() < on_frac
            )
            if use_manifold:
                ti = int(torch.randint(0, int(data.y.shape[0]), (1,), device=device).item())
                species_log = data.y[ti, :, 4:13].to(device=device, dtype=torch.float32)
            else:
                resting_state = torch.zeros(1, 9, device=device, dtype=scales.dtype)
                resting_state[:, rxn_keys.index('RP')] = scales[:, rxn_keys.index('RP')]
                resting_state[:, rxn_keys.index('FG')] = scales[:, rxn_keys.index('FG')]

                clotted_state = torch.zeros(1, 9, device=device, dtype=scales.dtype)
                clotted_state[:, rxn_keys.index('AP')] = scales[:, rxn_keys.index('AP')]
                clotted_state[:, rxn_keys.index('T')] = scales[:, rxn_keys.index('T')]
                clotted_state[:, rxn_keys.index('FI')] = scales[:, rxn_keys.index('FI')]

                alpha = torch.rand(n_nodes, 1, device=device)
                base_state = resting_state * (1.0 - alpha) + clotted_state * alpha
                jitter = torch.randn(n_nodes, 9, device=device) * (synth_jitter_frac * scales)
                species_lin_si = torch.clamp(base_state + jitter, min=0.0)
                species_log = torch.log1p(species_lin_si / scales)
            wall_species = torch.zeros(n_nodes, 3, device=device)
            random_species = torch.cat([species_log, wall_species], dim=1)

            if hasattr(data, "y") and data.y is not None and data.y.shape[-1] >= 3:
                ti_uv = int(ti) if use_manifold else 0
                ti_uv = min(ti_uv, int(data.y.shape[0]) - 1)
                u_v_p = data.y[ti_uv, :, :3]
            else:
                u_v = data.x[:, 11:13]
                p0 = torch.zeros(n_nodes, 1, device=device)
                u_v_p = torch.cat([u_v, p0], dim=1)
            u_det = u_v_p[:, 0]
            v_det = u_v_p[:, 1]

            bio_in = torch.cat([random_species, u_v_p, data.x[:, :15]], dim=-1)
            prior_tail = model._kinematics_prior_tail(data, u_det, v_det)
            if prior_tail is not None:
                bio_in = torch.cat([bio_in, prior_tail], dim=-1)
            with torch.no_grad():
                z0 = model.bio_encoder(bio_in)
            z0 = z0.detach()

            batch_idx_nodes = get_batch_tensor(data, n_nodes, device)
            edge_index = data.edge_index
            edge_attr = data.edge_attr

            ode_rxn_optimizer.zero_grad()

            dz_dt = model.ode_func(0.0, z0, edge_index, edge_attr, batch_idx_nodes)
            species_now = model._decode_species_log1p(model.biochem_decoder(z0))[:, :9]
            pred_dlog_dt = F.linear(dz_dt, model.biochem_decoder.linear.weight, bias=None)[:, :9]

            # Use the stable encoder input state for reaction targets.
            true_species_si = torch.clamp(torch.expm1(random_species[:, :9]), min=0.0) * scales
            true_species_dict = {k: true_species_si[:, i] for i, k in enumerate(rxn_keys)}
            props = kernels.core._get_geometric_props(data)
            if isinstance(data.u_ref, torch.Tensor) and data.u_ref.numel() == n_nodes:
                props['u_ref'] = data.u_ref
                props['d_bar'] = data.d_bar
            else:
                props['u_ref'] = data.u_ref[batch_idx_nodes]
                props['d_bar'] = data.d_bar[batch_idx_nodes]
            shear_rate = kernels._compute_shear_rate(u_det, v_det, props, data)
            reaction_terms = kernels.kinetics.compute_species_reactions(true_species_dict, shear_rate)
            t_ref = kernels.cfg.t_final
            # d(C_norm)/dt = d(log(1 + C_norm))/dt * (1 + C_norm_input)
            pred_norm_rate = pred_dlog_dt * torch.clamp(torch.exp(random_species[:, :9]), min=1e-8)
            target_norm_rate = torch.stack(
                [reaction_terms[k] * t_ref for k in rxn_keys],
                dim=1,
            ) / scales
            target_norm_rate = torch.clamp(target_norm_rate, min=-rate_clip, max=rate_clip)
            pred_norm_rate = torch.clamp(pred_norm_rate, min=-rate_clip, max=rate_clip)

            pred_symlog = torch.arcsinh(pred_norm_rate / symlog_scale)
            targ_symlog = torch.arcsinh(target_norm_rate / symlog_scale)
            loss = F.huber_loss(pred_symlog, targ_symlog, delta=0.25)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.ode_func.parameters(), max_norm=ode_clip)
            if decoder_params:
                torch.nn.utils.clip_grad_norm_(decoder_params, max_norm=dec_clip)
            ode_rxn_optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1

        if num_batches == 0:
            print(f"ODE-RXN Epoch {epoch:02d}: skipped (no usable batches).")
            continue
        avg_loss = total_loss / num_batches
        if ema_rxn_loss is None:
            ema_rxn_loss = avg_loss
        else:
            ema_rxn_loss = ode_ema_beta * ema_rxn_loss + (1.0 - ode_ema_beta) * avg_loss
        if prev_rxn_avg is not None:
            rel_change = abs(ema_rxn_loss - prev_rxn_avg) / max(abs(prev_rxn_avg), 1e-12)
            if rel_change < 1e-3:
                plateau_streak += 1
            else:
                plateau_streak = 0
            if plateau_streak >= 1:
                print(
                    f"⚠️ ODE-RXN plateau signal (EMA): rel_change={rel_change:.2e} "
                    f"(streak={plateau_streak + 1} epochs)"
                )
        prev_rxn_avg = ema_rxn_loss

        if (best_ema_rxn - ema_rxn_loss) > ode_min_delta:
            best_ema_rxn = ema_rxn_loss
            best_raw_rxn = avg_loss
            best_epoch_idx = epoch
            ode_bad_epochs = 0
            best_ode_state = {
                "ode_func": copy.deepcopy(model.ode_func.state_dict()),
                "biochem_decoder": copy.deepcopy(model.biochem_decoder.state_dict()),
            }
        else:
            if epoch + 1 >= ode_rxn_min_before_plateau:
                ode_bad_epochs += 1

        will_early_stop = (epoch + 1) >= ode_min_epochs and ode_bad_epochs >= ode_patience
        if _should_log_pretrain_epoch(epoch, ode_reaction_epochs, pretrain_log_interval) or will_early_stop:
            print(
                f"ODE-RXN Epoch {epoch:02d}: Reaction Mimic Loss = {avg_loss:.4e} "
                f"(ema={ema_rxn_loss:.4e}, beta={ode_ema_beta:.3f})"
            )

        last_ode_state = {
            "ode_func": copy.deepcopy(model.ode_func.state_dict()),
            "biochem_decoder": copy.deepcopy(model.biochem_decoder.state_dict()),
        }
        last_raw_rxn = float(avg_loss)

        if will_early_stop:
            print(
                f"🛑 ODE-RXN early stop at epoch {epoch:02d}: no EMA improvement > {ode_min_delta:.1e} "
                f"for {ode_bad_epochs} epoch(s) after min_epochs={ode_min_epochs}."
            )
            break

    prefer_last = _biochem_env_truthy("BIOCHEM_ODE_RXN_PREFER_LAST_IF_RAW_BETTER", default=True)
    used_last_over_best = False
    if (
        prefer_last
        and last_ode_state is not None
        and math.isfinite(last_raw_rxn)
        and best_epoch_idx >= 0
        and float(last_raw_rxn) + 1e-12 < float(best_raw_rxn)
    ):
        best_ode_state = last_ode_state
        used_last_over_best = True
        print(
            f"ℹ️ ODE-RXN: using last epoch weights (raw={last_raw_rxn:.4e}) over best-EMA epoch "
            f"{best_epoch_idx:02d} (raw={best_raw_rxn:.4e}). "
            "Set BIOCHEM_ODE_RXN_PREFER_LAST_IF_RAW_BETTER=0 to always restore the best-EMA snapshot.",
            flush=True,
        )

    model.ode_func.load_state_dict(best_ode_state["ode_func"])
    model.biochem_decoder.load_state_dict(best_ode_state["biochem_decoder"])
    if used_last_over_best:
        print(f"✅ Restored ODE-RXN weights from last epoch (raw-improvement policy; raw={last_raw_rxn:.4e}).")
    elif best_epoch_idx >= 0:
        print(
            f"✅ Restored best ODE-RXN weights from epoch {best_epoch_idx:02d} "
            f"(raw={best_raw_rxn:.4e}, ema={best_ema_rxn:.4e})."
        )
    else:
        print(
            "ℹ️ ODE-RXN: no epoch improved the EMA best (or every epoch had zero batches); "
            "keeping the initial ODE/decoder snapshot loaded at phase start.",
            flush=True,
        )

    if post_pretrain_save_path is not None:
        pp = Path(post_pretrain_save_path)
        pp.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), pp)
        print(f"💾 Saved post-pretrain warm-start for next run -> {pp}")

    for name, param in model.named_parameters():
        param.requires_grad = prior_requires_grad.get(name, True)


def _biochem_scale_for_isolate(w: float) -> float:
    """Use curriculum weight when >0 so isolate matches full objective scale; else 1.0 for grad signal."""
    return float(w) if float(w) > 0.0 else 1.0


def _biochem_linear_ramp_weight(base_w: float, epoch: int, ramp_epochs: int) -> float:
    """Linearly ramp a loss weight from 0 -> base_w over ``ramp_epochs`` epochs."""
    w = max(float(base_w), 0.0)
    r = max(int(ramp_epochs), 0)
    if w <= 0.0 or r <= 0:
        return w
    progress = min(max((float(epoch) + 1.0) / float(r), 0.0), 1.0)
    return w * progress


def _biochem_stage_blend(epoch: int, switch_epoch: int, transition_epochs: int) -> float:
    """Smoothly blend Stage-A -> Stage-B weights around switch epoch."""
    if switch_epoch <= 0:
        return 0.0
    tr = max(int(transition_epochs), 0)
    if epoch < switch_epoch:
        return 0.0
    if tr <= 0:
        return 1.0
    # Start blending at switch epoch and reach full Stage-B at switch+transition.
    p = (float(epoch) - float(switch_epoch) + 1.0) / float(tr)
    p = min(max(p, 0.0), 1.0)
    return p * p * (3.0 - 2.0 * p)  # smoothstep


def _anchor_mu_si_and_log_losses(
    pred_series: torch.Tensor,
    y_true: torch.Tensor,
    truth_mask: torch.Tensor,
    cfg_mu,
    mu_ch: int,
    *,
    d_mu: float,
    multi_step: bool,
    include_log: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SI μ supervision on COMSOL truth nodes over the TBPTT window.

    Returns Huber-on-SI (``l_mu_si``) and mean |log μ_pred − log μ_gt| in SI (``l_mu_log``), the
    same construction as val ``mu_log_mae``. When ``multi_step`` is False, only the **last**
    time index is used for Huber (legacy). Optional late-time weights via
    ``BIOCHEM_MU_ANCHOR_LATE_TIME_WEIGHT``.
    """
    device = pred_series.device
    dtype = pred_series.dtype
    ma = truth_mask
    z = torch.tensor(0.0, device=device, dtype=dtype)
    if (not ma.any()) or pred_series.shape[0] < 1 or y_true.shape[0] < 1:
        return z, z
    t_cap = min(int(pred_series.shape[0]), int(y_true.shape[0]))
    tw = _tbptt_mu_time_weights(t_cap, device, dtype)
    huber_terms: List[torch.Tensor] = []
    log_terms: List[torch.Tensor] = []
    for t in range(t_cap):
        mu_p = cfg_mu.viscosity_nd_to_si(pred_series[t, :, mu_ch])
        mu_g = cfg_mu.viscosity_nd_to_si(y_true[t, :, mu_ch])
        huber_terms.append(
            F.huber_loss(mu_p[ma], mu_g[ma], reduction="mean", delta=d_mu) * tw[t]
        )
        if include_log:
            mp = mu_p[ma].float().clamp(min=1e-8)
            mg = mu_g[ma].float().clamp(min=1e-8)
            log_terms.append(
                (torch.log(mp) - torch.log(mg)).abs().mean().to(dtype=dtype) * tw[t]
            )
    if not huber_terms:
        return z, z
    if multi_step:
        l_mu_si = torch.stack(huber_terms).sum()
    else:
        l_mu_si = huber_terms[-1]
    if include_log and log_terms:
        l_mu_log = torch.stack(log_terms).sum()
    else:
        l_mu_log = z
    return l_mu_si, l_mu_log


def _anchor_mu_subset_log_losses(
    pred_series: torch.Tensor,
    y_true: torch.Tensor,
    truth_mask: torch.Tensor,
    data,
    cfg_mu,
    mu_ch: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extra SI log-μ anchor losses on clinically important subsets.

    Returns:
    - ``l_mu_log_wall``: mean |log μ_pred − log μ_gt| on truth ∩ wall nodes
    - ``l_mu_log_high``: mean |log μ_pred − log μ_gt| on high-μ_gt tail nodes (per-step quantile)
    """
    device = pred_series.device
    dtype = pred_series.dtype
    z = torch.tensor(0.0, device=device, dtype=dtype)
    m_truth = truth_mask.view(-1).bool()
    if (not m_truth.any()) or pred_series.shape[0] < 1 or y_true.shape[0] < 1:
        return z, z

    has_wall_mask = hasattr(data, "mask_wall") and data.mask_wall is not None
    m_wall_base = (
        data.mask_wall.view(-1).bool().to(device)
        if has_wall_mask
        else torch.zeros_like(m_truth, device=device)
    )
    try:
        q_high = float(os.environ.get("BIOCHEM_MU_HIGH_QUANTILE", "0.9"))
    except ValueError:
        q_high = 0.9
    q_high = min(max(q_high, 0.5), 0.999)

    t_cap = min(int(pred_series.shape[0]), int(y_true.shape[0]))
    tw = _tbptt_mu_time_weights(t_cap, device, dtype)
    wall_terms: List[torch.Tensor] = []
    high_terms: List[torch.Tensor] = []

    for t in range(t_cap):
        mu_p = cfg_mu.viscosity_nd_to_si(pred_series[t, :, mu_ch])
        mu_g = cfg_mu.viscosity_nd_to_si(y_true[t, :, mu_ch])
        dlog = (torch.log(mu_p.float().clamp(min=1e-8)) - torch.log(mu_g.float().clamp(min=1e-8))).abs()

        if has_wall_mask:
            m_wall = m_truth & m_wall_base
            if m_wall.any():
                wall_terms.append(dlog[m_wall].mean().to(dtype=dtype) * tw[t])

        g_truth = mu_g[m_truth].float()
        if g_truth.numel() >= 8:
            high_thr = torch.quantile(g_truth, q_high)
            m_high = m_truth & (mu_g >= high_thr)
            if m_high.any():
                high_terms.append(dlog[m_high].mean().to(dtype=dtype) * tw[t])

    l_mu_log_wall = torch.stack(wall_terms).sum() if wall_terms else z
    l_mu_log_high = torch.stack(high_terms).sum() if high_terms else z
    return l_mu_log_wall, l_mu_log_high


def _biochem_resolve_isolated_loss(
    key: str,
    *,
    pred_final: torch.Tensor,
    l_adr_fast: torch.Tensor,
    l_adr_slow: torch.Tensor,
    l_wall_bio: torch.Tensor,
    l_wall_phys: torch.Tensor,
    l_bio_io: torch.Tensor,
    l_mom: torch.Tensor,
    l_data_kine: torch.Tensor,
    l_data_bio: torch.Tensor,
    l_pseudo: torch.Tensor,
    pseudo_loss_weight: float,
    l_latent_reg: torch.Tensor,
    latent_scale: float,
    l_visc_reg: torch.Tensor,
    visc_reg_w: float,
    l_kine_prior: torch.Tensor,
    w_kp: float,
    l_phys_temp: torch.Tensor,
    w_pt: float,
    l_mu_si_anchor: torch.Tensor,
    w_mu_aux: float,
    l_mu_log_anchor: torch.Tensor,
    w_mu_log: float,
    l_mu_log_wall: torch.Tensor,
    w_mu_log_wall: float,
    l_mu_log_high: torch.Tensor,
    w_mu_log_high: float,
    l_fi_gate_start: torch.Tensor,
    w_fi_gate_start_eff: float,
    l_residual_sparse: torch.Tensor,
    lambda_residual_sparse: float,
) -> torch.Tensor:
    """Single-term backprop scalar for ``BIOCHEM_LOSS_ISOLATE`` smoke tests."""
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
        "FI_GATE_START": "FI_GATE",
        "RESIDUAL_SPARSE": "RES_SPARSE",
    }
    k = aliases.get(k, k)
    if k == "ADR_F":
        return l_adr_fast
    if k == "ADR_S":
        return l_adr_slow
    if k == "W_BIO":
        return l_wall_bio
    if k == "W_PHY":
        return l_wall_phys
    if k == "BIO_IO":
        return l_bio_io
    if k == "NS_MOM":
        return l_mom
    if k == "DATA_KINE":
        return l_data_kine
    if k == "DATA_BIO":
        return l_data_bio
    if k == "PSEUDO":
        pw = max(float(pseudo_loss_weight), 0.0)
        return (pw if pw > 0.0 else 1.0) * l_pseudo
    if k == "LATENT":
        return _biochem_scale_for_isolate(latent_scale) * l_latent_reg
    if k in ("VISC", "VISC_REG"):
        return _biochem_scale_for_isolate(visc_reg_w) * l_visc_reg
    if k == "KINE_PRIOR":
        return _biochem_scale_for_isolate(w_kp) * l_kine_prior
    if k == "PHYS_TEMP":
        return _biochem_scale_for_isolate(w_pt) * l_phys_temp
    if k == "MU_SI":
        return (
            _biochem_scale_for_isolate(w_mu_aux) * l_mu_si_anchor
            + _biochem_scale_for_isolate(w_mu_log) * l_mu_log_anchor
            + _biochem_scale_for_isolate(w_mu_log_wall) * l_mu_log_wall
            + _biochem_scale_for_isolate(w_mu_log_high) * l_mu_log_high
        )
    if k == "MU_LOG":
        return (
            _biochem_scale_for_isolate(w_mu_log) * l_mu_log_anchor
            + _biochem_scale_for_isolate(w_mu_log_wall) * l_mu_log_wall
            + _biochem_scale_for_isolate(w_mu_log_high) * l_mu_log_high
        )
    if k == "MU_LOG_WALL":
        return _biochem_scale_for_isolate(w_mu_log_wall) * l_mu_log_wall
    if k == "MU_LOG_HIGH":
        return _biochem_scale_for_isolate(w_mu_log_high) * l_mu_log_high
    if k in ("FI_GATE", "FI_GATE_START"):
        return _biochem_scale_for_isolate(w_fi_gate_start_eff) * l_fi_gate_start
    if k in ("RES_SPARSE", "RESIDUAL_SPARSE"):
        return _biochem_scale_for_isolate(lambda_residual_sparse) * l_residual_sparse
    valid = (
        "ADR_F, ADR_S, W_BIO, W_PHY, BIO_IO, NS_MOM, DATA_KINE, DATA_BIO, PSEUDO, LATENT, "
        "VISC, KINE_PRIOR, PHYS_TEMP, MU_SI, MU_LOG, MU_LOG_WALL, MU_LOG_HIGH, FI_GATE, RES_SPARSE"
    )
    raise ValueError(f"Unknown BIOCHEM_LOSS_ISOLATE={key!r}; expected one of: {valid}")


def compute_biochem_loss(
    model,
    data,
    kernels,
    loss_weighter,
    device,
    bio_cfg,
    epoch=0,
    total_epochs=25,
    curriculum: Optional[CurriculumConfig] = None,
    debug_batch: Optional[Tuple[int, int]] = None,
    pseudo_target_trajectory: Optional[torch.Tensor] = None,
    pseudo_loss_weight: float = 0.0,
    train_cfg: Optional[BiochemTrainingConfig] = None,
):
    curriculum = curriculum or CurriculumConfig()
    train_cfg = train_cfg or BiochemTrainingConfig.from_env()
    kernels.set_biochem_huber_delta(_scheduled_biochem_huber_delta(bio_cfg, epoch))

    num_nodes_d = int(data.x.shape[0])
    truth_mask = biochem_truth_node_mask(data, num_nodes_d, device)

    re_ref = None
    if hasattr(data, 're_actual') and data.re_actual is not None:
        ra = data.re_actual
        re_ref = float(ra.mean().item()) if torch.is_tensor(ra) else float(ra)

    # NS momentum uses Re = get_re(u_ref, d_bar), not PhysicsConfig.re_target directly. Biochem can
    # override via ``re_actual`` (passed as re_ref). Fail fast with tensor dumps if Re would be <= 0.
    phys_cfg_ns = kernels.core.cfg
    u_s = data.u_ref.squeeze() if torch.is_tensor(data.u_ref) else data.u_ref
    d_s = data.d_bar.squeeze() if torch.is_tensor(data.d_bar) else data.d_bar
    Re_from_graph = phys_cfg_ns.get_re(u_s, d_s)
    if torch.is_tensor(Re_from_graph):
        r_lo = float(Re_from_graph.detach().min().item())
        r_hi = float(Re_from_graph.detach().max().item())
    else:
        r_lo = r_hi = float(Re_from_graph)
    Re_effective = float(re_ref) if re_ref is not None else r_lo
    if not math.isfinite(Re_effective) or Re_effective <= 0:
        raise ValueError(
            "Invalid Reynolds number for Navier–Stokes residual (1/Re blows up). "
            f"PhysicsConfig.re_target={phys_cfg_ns.re_target} only defines scaling when graphs are built; "
            f"runtime Re is get_re(u_ref, d_bar) unless overridden by data.re_actual → re_ref. "
            f"Effective Re={Re_effective}, re_ref={re_ref!r}, get_re(u_ref,d_bar) in [{r_lo}, {r_hi}], "
            f"u_ref={data.u_ref!r}, d_bar={data.d_bar!r}, re_actual={getattr(data, 're_actual', None)!r}"
        )

    # Pass non-dimensional time into the ODE integration window.
    t_ref = bio_cfg.t_final
    full_times = to_t_nd(bio_cfg.resolve_biochem_times(data, device), t_ref)

    actual_num_steps = int(data.y.shape[0])
    start_idx = 0
    end_idx = actual_num_steps
    y_true_trajectory = data.y
    teacher_forcing_ratio = 0.0
    tbptt_info: Optional[Dict[str, Any]] = None

    wu = curriculum.biochem_warmup_epochs
    teacher_force_min = min(
        max(float(os.environ.get("BIOCHEM_TEACHER_FORCE_MIN", "0.0")), 0.0),
        1.0,
    )
    if model.training:
        # Teacher stage often runs with total_epochs <= warmup; avoid 100% TF lock-in.
        if total_epochs <= wu:
            # Teacher-stage runs are often shorter than warmup; allow callers to keep a
            # non-zero floor so supervised COMSOL anchors remain the dominant signal.
            teacher_forcing_ratio = max(
                teacher_force_min,
                1.0 - (epoch / float(max(1, total_epochs))),
            )
        elif epoch < wu:
            # Warmup still decays to expose autoregressive errors early.
            teacher_forcing_ratio = 1.0 - 0.5 * (epoch / float(max(1, wu)))
        else:
            decay_progress = (epoch - wu) / float(curriculum.biochem_teacher_force_decay_epochs)
            decay_progress = _ease01(decay_progress, curriculum.biochem_curriculum_easing)
            # Continue decaying from the warmup endpoint (0.5) to 0.0.
            teacher_forcing_ratio = max(0.0, 0.5 * (1.0 - decay_progress))

        # Teacher forcing uses COMSOL labels only where ``biochem_truth_node_mask`` is True
        # (synthetic graphs: all False; patient graphs: spatially matched nodes only).
        if not truth_mask.any():
            teacher_forcing_ratio = 0.0

        # TBPTT: shorten the time window to limit pred_trajectory length and autograd memory.
        # Windows are in **index** space; the physical span is set by ``data.t`` / extraction
        # (TEMP DEBUG: ``BIOCHEM_T_MAX`` / ``biochem_physical_time_horizon_s()``).
        # Anchors may start at random start_idx (ground truth exists at every time). Synthetic graphs
        # must use start_idx=0 so species ICs stay the resting prior at physical t=0; only the window
        # length is capped (avoids simulating all ~60 steps in one graph when truth_mask is all False).
        if actual_num_steps > 2:
            window_cap = max(2, actual_num_steps - 1)
            # Keep windows small for stability/speed; random start_idx covers different trajectory regions.
            # ``BIOCHEM_TBPTT_MAX_WINDOW`` caps slice length in index space (default: use full cap from epoch 0).
            # Ramp early epochs with ``BIOCHEM_TBPTT_WINDOW_CURRICULUM=1`` (legacy ``min(5+epoch//4, cap)``).
            tbptt_cap = max(2, int(os.environ.get("BIOCHEM_TBPTT_MAX_WINDOW", "8")))
            use_tbptt_curriculum = (os.environ.get("BIOCHEM_TBPTT_WINDOW_CURRICULUM", "") or "").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            if use_tbptt_curriculum:
                proposed_window = 5 + (epoch // 4)
                window_size = min(proposed_window, tbptt_cap, window_cap)
            else:
                window_size = min(tbptt_cap, window_cap)
            if truth_mask.any():
                start_idx = _resolve_tbptt_anchor_start_idx(actual_num_steps, window_size, device)
            else:
                early_frac = float(os.environ.get("BIOCHEM_SYNTH_TBPTT_EARLY_FRAC", "0.4"))
                early_frac = min(max(early_frac, 0.0), 1.0)
                early_epochs = max(1, int(total_epochs * early_frac))
                if epoch < early_epochs:
                    half = max(1, early_epochs // 2)
                    # Keep synthetic TBPTT window >= 2 so ODE executes at least one step.
                    synth_window = 2 if epoch < half else 3
                    window_size = min(window_cap, max(2, synth_window))
                start_idx = 0
            end_idx = start_idx + window_size
            y_true_trajectory = data.y[start_idx:end_idx]
            evaluation_times = full_times[start_idx:end_idx]
            tbptt_info = {
                "cap": int(tbptt_cap),
                "curriculum": bool(use_tbptt_curriculum),
                "window_len": int(window_size),
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "window_cap": int(window_cap),
            }
            if hasattr(data, "t") and data.t is not None and int(data.t.numel()) >= int(end_idx):
                try:
                    t_cpu = data.t.detach().cpu()
                    tbptt_info["t_si_start"] = float(t_cpu[int(start_idx)].item())
                    tbptt_info["t_si_end"] = float(t_cpu[int(end_idx) - 1].item())
                except (IndexError, RuntimeError):
                    pass
        else:
            evaluation_times = full_times
    else:
        evaluation_times = full_times

    initial_species_for_window = None
    if model.training and start_idx > 0:
        skip_warmup_roll = _biochem_env_truthy("BIOCHEM_TBPTT_SKIP_WARMUP_ROLLFORWARD", default=False)
        if not skip_warmup_roll and _biochem_env_truthy("BIOCHEM_TBPTT_ANCHOR_END_BIAS", default=False):
            skip_warmup_roll = bool(truth_mask.any())
        if skip_warmup_roll and int(data.y.shape[0]) > int(start_idx):
            initial_species_for_window = data.y[start_idx, :, 4:16].to(device)
        else:
            with torch.no_grad():
                warmup_times = full_times[: start_idx + 1]
                warmup_series = model(
                    data,
                    warmup_times,
                    y_true_trajectory=data.y[: start_idx + 1],
                    teacher_forcing_ratio=1.0,
                    start_idx=0,
                )
                warmup_species = warmup_series[-1, :, 4:16]

            if truth_mask.any():
                gt_species_at_start = data.y[start_idx, :, 4:16].to(device)
                initial_species_for_window = torch.where(
                    truth_mask.unsqueeze(-1),
                    gt_species_at_start,
                    warmup_species,
                )
            else:
                initial_species_for_window = warmup_species

    # TBPTT + macro detach: never force detach_macro_state=True just because model.training —
    # that severs the species carry graph every macro step and removes grad to bio_encoder / ODE / decoder
    # for L_Data_Bio on multi-step windows. Use BIOCHEM_DETACH_MACRO_STATE=1 only for long-rollout VRAM caps.
    pred_series = model(
        data,
        evaluation_times,
        y_true_trajectory=y_true_trajectory,
        teacher_forcing_ratio=teacher_forcing_ratio,
        start_idx=start_idx,
        initial_species=initial_species_for_window,
        detach_macro_state=_biochem_env_truthy("BIOCHEM_DETACH_MACRO_STATE", default=False),
    )

    props = kernels.core._get_geometric_props(data)
    batch_idx_nodes = get_batch_tensor(data, data.num_nodes, device)
    if isinstance(data.u_ref, torch.Tensor) and data.u_ref.numel() == data.num_nodes:
        props['u_ref'] = data.u_ref
        props['d_bar'] = data.d_bar
    else:
        props['u_ref'] = data.u_ref[batch_idx_nodes]
        props['d_bar'] = data.d_bar[batch_idx_nodes]

    # 3. Supervised data loss (full supervised time window on anchor nodes)
    pred_final = pred_series[-1]
    l_data_kine = torch.tensor(0.0, device=device)
    l_data_bio = torch.tensor(0.0, device=device)
    t_eval_bio_for_metrics = 0.0
    has_anchor_supervision = bool(truth_mask.any().item())

    # FIX: Since we removed the 3x dense multiplier, the prediction frequency
    # perfectly matches the data frequency. No need to slice [::3] anymore!
    pred_series_data_freq = pred_series
    target_series = y_true_trajectory.to(device)

    # Supervised loss only on COMSOL-trusted nodes (entire trajectory window).
    # l_data_kine: Huber on [u,v,p,mu_nd] vs batch variance scale (anchors only).
    # l_data_bio: Huber on log1p species channels vs per-channel floors (bulk + wall).
    if has_anchor_supervision:
        node_is_anchor = truth_mask
        pred_kine = pred_series_data_freq[:, node_is_anchor, :4]
        targ_kine = target_series[:, node_is_anchor, :4]
        kine_var = torch.clamp(torch.var(targ_kine, dim=(0, 1), keepdim=True), min=1e-2)
        l_data_kine = torch.mean(F.huber_loss(pred_kine, targ_kine, reduction='none') / kine_var)

        # ---------------------------------------------------------
        # Native log1p Huber loss to prevent exponential gradient crushing.
        # ---------------------------------------------------------
        pred_bio = pred_series_data_freq[:, node_is_anchor, 4:16]
        targ_bio = target_series[:, node_is_anchor, 4:16]

        scales = bio_cfg.get_species_scales(device=device)
        target_ranges = scales.clone()
        target_ranges[1] = max(float(bio_cfg.c_AP0 * bio_cfg.bulk_scale), 1e-8)   # AP
        target_ranges[5] = max(float(bio_cfg.Tcrit * bio_cfg.bulk_scale), 1e-8)   # T
        target_ranges[8] = max(float(bio_cfg.viscosity_fi_crit), 1e-8)            # FI
        target_ranges[11] = max(float(bio_cfg.viscosity_mat_crit), 1e-8)          # Mat

        # Relative channel weighting without expm1-space amplification.
        channel_weights = (scales / target_ranges).view(1, 1, 12)
        channel_weights = torch.clamp(channel_weights, min=1.0, max=100.0)

        # Huber directly in bounded log1p output space.
        base_huber = F.huber_loss(pred_bio, targ_bio, reduction="none", delta=1.0)

        raw_bio_magnitude = max(float(os.environ.get("BIOCHEM_RAW_BIO_MAGNITUDE", "0.01")), 1e-12)
        # Time-normalized supervised bio loss: sum over (time × anchor nodes × channels),
        # scaled by 1/T_eval so TBPTT windows of different lengths stay comparable (per-step scale).
        t_eval = max(int(pred_bio.shape[0]), 1)
        n_anc = max(int(pred_bio.shape[1]), 1)
        n_ch = max(int(pred_bio.shape[2]), 1)
        weighted_huber = base_huber * channel_weights
        l_data_bio = weighted_huber.sum() / (float(t_eval) * float(n_anc) * float(n_ch)) / raw_bio_magnitude
        t_eval_bio_for_metrics = float(t_eval)

    l_pseudo = torch.tensor(0.0, device=device)
    has_pseudo_supervision = False
    if pseudo_target_trajectory is not None and (not has_anchor_supervision):
        pseudo_target = pseudo_target_trajectory.to(device)
        pseudo_target = pseudo_target[start_idx:end_idx]
        if pseudo_target.shape == pred_series.shape:
            has_pseudo_supervision = True
            l_pseudo = F.mse_loss(pred_series[:, :, 4:16], pseudo_target[:, :, 4:16])

    # 4. Physics PDE Loss (Evaluated over dense time sequence)
    num_steps = len(evaluation_times) - 1
    # Keep fallback zero losses attached to the current forward graph.
    # Some sparse/synthetic batches can yield no valid residual nodes; if every
    # active loss is a detached constant, backward() will fail. We initialise each
    # accumulator from a *fresh* ``pred_final.sum() * 0.0`` so the five tensors are
    # independent storage. A chained ``a = b = c = z`` would alias them; subsequent
    # in-place updates anywhere downstream would silently couple distinct losses.
    def _zero_loss():
        return pred_final.sum() * 0.0

    if num_steps <= 0:
        l_adr_fast = _zero_loss()
        l_adr_slow = _zero_loss()
        l_wall_bio = _zero_loss()
        l_wall_phys = _zero_loss()
        l_bio_io = _zero_loss()
    else:
        dt_intervals = (evaluation_times[1:] - evaluation_times[:-1]).view(-1, 1, 1)
        dt_intervals = torch.clamp(dt_intervals, min=1e-9)
        d_pred_dt = (pred_series[1:] - pred_series[:-1]) / dt_intervals
        l_adr_fast = _zero_loss()
        l_adr_slow = _zero_loss()
        l_wall_bio = _zero_loss()
        l_wall_phys = _zero_loss()
        l_bio_io = _zero_loss()

        for t_idx in range(num_steps):
            # Evaluate physics at step t+1 using finite difference gradient
            pred_t = pred_series[t_idx + 1]
            d_dt_t = d_pred_dt[t_idx]

            vel_t = pred_t[:, 0:2]
            # Relax upper bound so FI can exceed clot threshold while still
            # preventing unbounded autoregressive blow-ups.
            biochem_t = torch.clamp(pred_t[:, 4:13], min=-10.0, max=8.0)
            wall_t = torch.clamp(pred_t[:, 13:16], min=-10.0, max=8.0)

            dC_dt_t = d_dt_t[:, 4:13]
            dM_dt_t = d_dt_t[:, 13:16]

            l_af, l_as = kernels.biochem_adr_residual(biochem_t, vel_t, props, data, d_pred_dt=dC_dt_t)
            l_wb, l_wp = kernels.biochem_wall_residual(biochem_t, wall_t, vel_t, props, data, dM_dt_t)
            l_bi, l_bo = kernels.biochem_inlet_outlet_residual(biochem_t, props, data)

            l_adr_fast = l_adr_fast + l_af
            l_adr_slow = l_adr_slow + l_as
            l_wall_bio = l_wall_bio + l_wb
            l_wall_phys = l_wall_phys + l_wp
            l_bio_io = l_bio_io + (l_bi + l_bo)

        inv = 1.0 / float(num_steps)
        l_adr_fast = l_adr_fast * inv
        l_adr_slow = l_adr_slow * inv
        l_wall_bio = l_wall_bio * inv
        l_wall_phys = l_wall_phys * inv
        l_bio_io = l_bio_io * inv

    wall_surface_mult = max(float(curriculum.biochem_wall_surface_loss_multiplier), 1e-6)
    wall_flux_mult = max(float(curriculum.biochem_wall_flux_loss_multiplier), 1e-6)
    l_wall_bio = l_wall_bio * wall_surface_mult
    l_wall_phys = l_wall_phys * wall_flux_mult

    # Fluid Mechanics (pseudo-steady snapshot at final time in the window)
    l_mom = kernels.core.navier_stokes_residual(
        pred_final[:, 0:4], data, props=props, re_ref=re_ref
    )
    l_visc_reg = kernels.compute_dual_viscosity_penalty(
        pred_final[:, 13:14],  # M_wall proxy (first wall-species channel)
        pred_final[:, 12:13],  # FI_field proxy (FI bulk channel)
        props,
        data,
    )

    # --- Kinematics prior + COMSOL temporal derivative match (continuous supervision) ---
    l_kine_prior = torch.tensor(0.0, device=device)
    l_phys_temp = torch.tensor(0.0, device=device)
    w_kp_max = train_cfg.kine_prior_weight
    kp_ramp = train_cfg.kine_prior_ramp_epochs
    if model.training and w_kp_max > 0.0 and kp_ramp > 0:
        w_kp = w_kp_max * min(1.0, float(epoch + 1) / float(kp_ramp))
    else:
        w_kp = w_kp_max if model.training else 0.0
    w_pt = float(os.environ.get("BIOCHEM_COMSOL_TEMPORAL_WEIGHT", "0.02"))
    phys_cfg = kernels.core.cfg
    mu_ch = STATE_CHANNEL_MU_EFF_ND
    prior_sparse_tgt = None
    mu_excess_all = None
    mu_gt_excess_all = None
    if model.training and has_anchor_supervision:
        mu_p = phys_cfg.viscosity_nd_to_si(pred_final[:, mu_ch])
        mu_g = phys_cfg.viscosity_nd_to_si(y_true_trajectory[-1, :, mu_ch])
        thr = 20.0 * phys_cfg.mu_viscosity_nd_scale
        mu_floor = float(phys_cfg.mu_inf)
        mu_span = max(float(thr) - mu_floor, 1e-8)
        mu_excess = ((mu_p - mu_floor) / mu_span).clamp(0.0, 1.0)
        mu_gt_excess = ((mu_g - mu_floor) / mu_span).clamp(0.0, 1.0)
        mu_excess_all = mu_excess
        mu_gt_excess_all = mu_gt_excess

        if hasattr(data, "G_x") and hasattr(data, "G_y"):
            u_lab = target_series[-1, :, 0].detach()
            v_lab = target_series[-1, :, 1].detach()
            prior = clot_prior_score_flat(data, u_lab, v_lab, bio_cfg, props).detach()
            prior_sparse_tgt = prior
            delta_fi: Optional[torch.Tensor] = None
            if (
                w_kp > 0.0
                and hasattr(data, "edge_index")
                and data.edge_index is not None
                and data.edge_index.numel() >= 2
            ):
                sp_b = torch.clamp(pred_final[:, 4:16], min=-10.0, max=8.0)
                scales_b = bio_cfg.get_species_scales(device=device)
                fi_si = torch.expm1(sp_b[:, 8]) * scales_b[8]
                delta_fi = _per_node_mean_abs_edge_diff(fi_si, data.edge_index, num_nodes_d)
                w_slow = (1.0 - prior).clamp(0.0, 1.0)
                l_kine_prior = (delta_fi[truth_mask] * w_slow[truth_mask]).mean()
            if delta_fi is not None:
                _emit_clot_batch_trace(
                    truth_mask=truth_mask,
                    prior=prior,
                    mu_p_si=mu_p,
                    mu_g_si=mu_g,
                    delta_fi=delta_fi,
                )
    elif model.training and (not has_anchor_supervision) and w_kp > 0.0:
        _sw = os.environ.get("BIOCHEM_KINE_PRIOR_SYNTH_WEIGHT")
        w_syn = float(_sw) if _sw is not None else 0.15
        if (
            w_syn > 0.0
            and hasattr(data, "G_x")
            and hasattr(data, "G_y")
            and hasattr(data, "edge_index")
            and data.edge_index is not None
            and data.edge_index.numel() >= 2
        ):
            mu_p = phys_cfg.viscosity_nd_to_si(pred_final[:, mu_ch])
            thr_syn = 20.0 * phys_cfg.mu_viscosity_nd_scale
            mu_floor = float(phys_cfg.mu_inf)
            mu_span_syn = max(float(thr_syn) - mu_floor, 1e-8)
            mu_excess_syn = ((mu_p - mu_floor) / mu_span_syn).clamp(0.0, 1.0)
            mu_excess_all = mu_excess_syn
            prior_s = clot_prior_score_flat(
                data, pred_final[:, 0].detach(), pred_final[:, 1].detach(), bio_cfg, props
            ).detach()
            prior_sparse_tgt = prior_s
            sp_b = torch.clamp(pred_final[:, 4:16], min=-10.0, max=8.0)
            scales_b = bio_cfg.get_species_scales(device=device)
            fi_si = torch.expm1(sp_b[:, 8]) * scales_b[8]
            delta_fi = _per_node_mean_abs_edge_diff(fi_si, data.edge_index, num_nodes_d)
            w_slow = (1.0 - prior_s).clamp(0.0, 1.0)
            l_kine_prior = (delta_fi * w_slow).mean() * w_syn
    if (
        model.training
        and has_anchor_supervision
        and w_pt > 0.0
        and pred_series_data_freq.shape[0] >= 2
        and evaluation_times.numel() >= 2
    ):
        node_is_anchor = truth_mask
        dtv = (evaluation_times[1:] - evaluation_times[:-1]).view(-1, 1, 1).clamp(min=1e-9)
        pd = (pred_series_data_freq[1:, node_is_anchor] - pred_series_data_freq[:-1, node_is_anchor]) / dtv
        gd = (target_series[1:, node_is_anchor] - target_series[:-1, node_is_anchor]) / dtv
        l_phys_temp = F.huber_loss(pd, gd, reduction="mean", delta=1.0)

    # Direct SI effective-viscosity fit on COMSOL anchors (primary high-viscosity supervision).
    # Multi-step: mean Huber over the TBPTT window (not only the final state). Optional ``l_mu_log``
    # matches validation ``mu_log_mae`` (mean |log μ_pred − log μ_gt| in SI on truth nodes).
    l_mu_si_anchor = torch.tensor(0.0, device=device)
    l_mu_log_anchor = torch.tensor(0.0, device=device)
    l_mu_log_wall = torch.tensor(0.0, device=device)
    l_mu_log_high = torch.tensor(0.0, device=device)
    l_mu_log_boundary = torch.tensor(0.0, device=device)
    w_mu_aux = max(float(os.environ.get("BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT", "0.0")), 0.0)
    w_mu_log = max(float(os.environ.get("BIOCHEM_MU_LOG_ANCHOR_WEIGHT", "0.0")), 0.0)
    w_mu_log_wall_base = max(float(os.environ.get("BIOCHEM_MU_LOG_WALL_WEIGHT", "0.0")), 0.0)
    w_mu_log_high_base = max(float(os.environ.get("BIOCHEM_MU_LOG_HIGH_WEIGHT", "0.0")), 0.0)
    stage_switch_ep = max(0, int(os.environ.get("BIOCHEM_MU_STAGE_SWITCH_EPOCH", "0")))
    stage_transition_ep = max(0, int(os.environ.get("BIOCHEM_MU_STAGE_TRANSITION_EPOCHS", "0")))
    stage_blend = _biochem_stage_blend(epoch, stage_switch_ep, stage_transition_ep)
    if stage_switch_ep > 0:
        w_mu_aux_b = max(float(os.environ.get("BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT_STAGE_B", str(w_mu_aux))), 0.0)
        w_mu_log_b = max(float(os.environ.get("BIOCHEM_MU_LOG_ANCHOR_WEIGHT_STAGE_B", str(w_mu_log))), 0.0)
        w_mu_log_wall_b = max(
            float(os.environ.get("BIOCHEM_MU_LOG_WALL_WEIGHT_STAGE_B", str(w_mu_log_wall_base))),
            0.0,
        )
        w_mu_log_high_b = max(
            float(os.environ.get("BIOCHEM_MU_LOG_HIGH_WEIGHT_STAGE_B", str(w_mu_log_high_base))),
            0.0,
        )
        w_mu_aux = ((1.0 - stage_blend) * w_mu_aux) + (stage_blend * w_mu_aux_b)
        w_mu_log = ((1.0 - stage_blend) * w_mu_log) + (stage_blend * w_mu_log_b)
        w_mu_log_wall_base = ((1.0 - stage_blend) * w_mu_log_wall_base) + (stage_blend * w_mu_log_wall_b)
        w_mu_log_high_base = ((1.0 - stage_blend) * w_mu_log_high_base) + (stage_blend * w_mu_log_high_b)
    wall_ramp_epochs = max(0, int(os.environ.get("BIOCHEM_MU_LOG_WALL_RAMP_EPOCHS", "0")))
    high_ramp_epochs = max(0, int(os.environ.get("BIOCHEM_MU_LOG_HIGH_RAMP_EPOCHS", "0")))
    w_mu_log_wall = (
        _biochem_linear_ramp_weight(w_mu_log_wall_base, epoch, wall_ramp_epochs)
        if model.training
        else w_mu_log_wall_base
    )
    w_mu_log_high = (
        _biochem_linear_ramp_weight(w_mu_log_high_base, epoch, high_ramp_epochs)
        if model.training
        else w_mu_log_high_base
    )
    mu_aux_early_epochs = max(0, int(os.environ.get("BIOCHEM_MU_SI_ANCHOR_AUX_EARLY_EPOCHS", "0")))
    mu_aux_early_mult = max(1.0, float(os.environ.get("BIOCHEM_MU_SI_ANCHOR_AUX_EARLY_MULT", "1.0")))
    if model.training and mu_aux_early_epochs > 0 and epoch < mu_aux_early_epochs:
        w_mu_aux = w_mu_aux * mu_aux_early_mult
    _mu_si_multi_step = (os.environ.get("BIOCHEM_MU_SI_MULTI_STEP", "1") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if model.training and has_anchor_supervision and (w_mu_aux > 0.0 or w_mu_log > 0.0 or w_mu_log_wall > 0.0 or w_mu_log_high > 0.0):
        cfg_mu = kernels.core.cfg
        ma = truth_mask
        if ma.any():
            d_mu = max(float(os.environ.get("BIOCHEM_MU_SI_HUBER_DELTA", "0.25")), 1e-7)
            l_mu_si_anchor, l_mu_log_anchor = _anchor_mu_si_and_log_losses(
                pred_series_data_freq,
                target_series,
                ma,
                cfg_mu,
                mu_ch,
                d_mu=d_mu,
                multi_step=_mu_si_multi_step,
                include_log=(w_mu_log > 0.0),
            )
            if w_mu_log_wall > 0.0 or w_mu_log_high > 0.0:
                l_mu_log_wall, l_mu_log_high = _anchor_mu_subset_log_losses(
                    pred_series_data_freq,
                    target_series,
                    ma,
                    data,
                    cfg_mu,
                    mu_ch,
                )
            w_mu_log_boundary = max(float(os.environ.get("BIOCHEM_MU_LOG_BOUNDARY_WEIGHT", "0.0")), 0.0)
            if (
                w_mu_log_boundary > 0.0
                and hasattr(data, "edge_index")
                and data.edge_index is not None
                and data.edge_index.numel() >= 2
            ):
                pred_last = pred_series_data_freq[-1, :, mu_ch]
                targ_last = target_series[-1, :, mu_ch]
                pred_mu = cfg_mu.viscosity_nd_to_si(pred_last)
                targ_mu = cfg_mu.viscosity_nd_to_si(targ_last)
                mu_cut = torch.quantile(targ_mu[ma], _mu_val_high_quantile())
                high_nodes = ma & (targ_mu >= mu_cut)
                row, col = data.edge_index
                boundary_edge = high_nodes[row] ^ high_nodes[col]
                boundary_nodes = torch.zeros_like(ma)
                if bool(boundary_edge.any()):
                    boundary_nodes[row[boundary_edge]] = True
                    boundary_nodes[col[boundary_edge]] = True
                boundary_nodes = boundary_nodes & ma
                if bool(boundary_nodes.any()):
                    l_mu_log_boundary = torch.mean(
                        torch.abs(
                            torch.log(torch.clamp(pred_mu[boundary_nodes], min=1e-12))
                            - torch.log(torch.clamp(targ_mu[boundary_nodes], min=1e-12))
                        )
                    )
            if w_mu_aux <= 0.0:
                l_mu_si_anchor = torch.tensor(0.0, device=device, dtype=pred_series_data_freq.dtype)
            if w_mu_log <= 0.0:
                l_mu_log_anchor = torch.tensor(0.0, device=device, dtype=pred_series_data_freq.dtype)
            if w_mu_log_wall <= 0.0:
                l_mu_log_wall = torch.tensor(0.0, device=device, dtype=pred_series_data_freq.dtype)
            if w_mu_log_high <= 0.0:
                l_mu_log_high = torch.tensor(0.0, device=device, dtype=pred_series_data_freq.dtype)

    mu_dbg = _compute_mu_flow_debug_metrics(
        model=model,
        data=data,
        pred_final=pred_final,
        target_final_mu_nd=(target_series[-1, :, mu_ch] if has_anchor_supervision else None),
        phys_cfg=phys_cfg,
        truth_mask=truth_mask,
        mu_ch=mu_ch,
    )

    # Teacher-start FI gate suppression: prevents inherited broad FI activation from
    # immediately doubling viscosity before anchor supervision can pull states back.
    l_fi_gate_start = torch.tensor(0.0, device=device)
    w_fi_gate_start_eff = 0.0
    if model.training:
        fi_gate_epochs = max(0, int(os.environ.get("BIOCHEM_FI_GATE_START_EPOCHS", "0")))
        fi_gate_w = max(0.0, float(os.environ.get("BIOCHEM_FI_GATE_START_WEIGHT", "0.0")))
        fi_gate_eps = max(0.0, float(os.environ.get("BIOCHEM_FI_GATE_START_EPS", "0.03")))
        if (
            fi_gate_epochs > 0
            and fi_gate_w > 0.0
            and epoch < fi_gate_epochs
            and float(teacher_forcing_ratio) < 0.92
        ):
            decay = _teacher_start_decay(epoch, fi_gate_epochs)
            w_fi_gate_start_eff = fi_gate_w * decay
            sp_last = torch.clamp(pred_final[:, 4:16], min=-10.0, max=8.0)
            scales_last = bio_cfg.get_species_scales(device=device)
            fi_si_last = torch.expm1(sp_last[:, 8]) * scales_last[8]
            t_scale = max(float(getattr(model, "T_scale", 1.0)), 1e-5)
            fi_temp = max(float(bio_cfg.viscosity_gnode_temp_fi) * t_scale, 1e-8)
            fi_logits = torch.clamp((fi_si_last - float(bio_cfg.viscosity_fi_crit)) / fi_temp, min=-50.0, max=50.0)
            # Must track ``model.mu_ratio_max`` (teacher forces μ₂ saturation scale to 1).
            mu_ratio_eff = float(getattr(model, "mu_ratio_max", bio_cfg.mu_ratio_max))
            mu2_fi = mu_ratio_eff * torch.sigmoid(fi_logits)
            if has_anchor_supervision:
                mu2_fi = mu2_fi[truth_mask]
            l_fi_gate_start = torch.mean(torch.relu(mu2_fi - fi_gate_eps).pow(2))

    # Residual sparsity prior (best-practice macro->micro decomposition):
    # penalize positive high-viscosity residual away from anchor-evidenced / prior-supported regions.
    l_residual_sparse = torch.tensor(0.0, device=device)
    lambda_residual_sparse = 0.0
    if model.training:
        lambda_residual_sparse = _scheduled_residual_sparse_lambda(epoch, total_epochs)
        if lambda_residual_sparse > 0.0:
            if mu_excess_all is None:
                mu_p_all = phys_cfg.viscosity_nd_to_si(pred_final[:, mu_ch])
                thr_all = 20.0 * phys_cfg.mu_viscosity_nd_scale
                mu_floor_all = float(phys_cfg.mu_inf)
                mu_span_all = max(float(thr_all) - mu_floor_all, 1e-8)
                mu_excess_all = ((mu_p_all - mu_floor_all) / mu_span_all).clamp(0.0, 1.0)
            guide = torch.zeros_like(mu_excess_all)
            if prior_sparse_tgt is not None:
                guide = torch.maximum(guide, prior_sparse_tgt.detach().to(guide.device))
            if has_anchor_supervision:
                if mu_gt_excess_all is None:
                    mu_g_all = phys_cfg.viscosity_nd_to_si(y_true_trajectory[-1, :, mu_ch])
                    thr_all = 20.0 * phys_cfg.mu_viscosity_nd_scale
                    mu_floor_all = float(phys_cfg.mu_inf)
                    mu_span_all = max(float(thr_all) - mu_floor_all, 1e-8)
                    mu_gt_excess_all = ((mu_g_all - mu_floor_all) / mu_span_all).clamp(0.0, 1.0)
                guide = torch.where(truth_mask, torch.maximum(guide, mu_gt_excess_all), guide)
            residual_excess = torch.relu(mu_excess_all - guide)
            l_residual_sparse = torch.mean(residual_excess.pow(2))

    # Trigger anti-collapse priors (opt-in):
    # keep tail trigger/gelation pathway alive on high-μ anchor nodes even when
    # global losses favor collapsing back to near-pure Carreau.
    l_trigger_gate_floor = torch.tensor(0.0, device=device)
    l_trigger_learned_floor = torch.tensor(0.0, device=device)
    w_trigger_gate_floor = 0.0
    w_trigger_learned_floor = 0.0
    if model.training and has_anchor_supervision:
        w_trigger_gate_floor = max(float(os.environ.get("BIOCHEM_TRIGGER_GATE_FLOOR_WEIGHT", "0.0")), 0.0)
        w_trigger_learned_floor = max(float(os.environ.get("BIOCHEM_TRIGGER_LEARNED_FLOOR_WEIGHT", "0.0")), 0.0)
        if w_trigger_gate_floor > 0.0 or w_trigger_learned_floor > 0.0:
            ma = truth_mask
            if ma.any():
                mu_g_si_last = phys_cfg.viscosity_nd_to_si(target_series[-1, :, mu_ch])
                mu_truth_nodes = mu_g_si_last[ma]
                q_high = _mu_val_high_quantile()
                mu_cut = torch.quantile(mu_truth_nodes.detach(), q_high)
                high_mask = ma & (mu_g_si_last >= mu_cut)
                if bool(high_mask.any()):
                    if w_trigger_gate_floor > 0.0:
                        gate = getattr(model, "_last_mu_trigger_gate", None)
                        if torch.is_tensor(gate):
                            gate_h = gate.reshape(-1)[high_mask]
                            gate_target = max(
                                0.0,
                                min(1.0, float(os.environ.get("BIOCHEM_TRIGGER_GATE_MIN_HIGH", "0.35"))),
                            )
                            l_trigger_gate_floor = torch.relu(
                                torch.tensor(gate_target, device=device, dtype=gate_h.dtype) - gate_h.mean()
                            ).pow(2)
                    if w_trigger_learned_floor > 0.0 and hasattr(model, "learned_clot_penalty"):
                        sp_last = torch.clamp(pred_final[:, 4:16], min=-10.0, max=8.0)
                        learned_h = model.learned_clot_penalty(sp_last).reshape(-1)[high_mask]
                        learned_target = max(float(os.environ.get("BIOCHEM_TRIGGER_LEARNED_MIN_HIGH", "0.03")), 0.0)
                        l_trigger_learned_floor = torch.relu(
                            torch.tensor(learned_target, device=device, dtype=learned_h.dtype) - learned_h.mean()
                        ).pow(2)

    latent_scale = train_cfg.latent_reg_scale

    # Eight Kendall tasks: skip supervised heads on non-anchor batches.
    all_losses = [
        l_adr_fast, l_adr_slow, l_wall_bio, l_wall_phys, l_bio_io, l_mom,
        l_data_kine, l_data_bio,
    ]
    task_active = [True] * 6 + [has_anchor_supervision, has_anchor_supervision]
    l_latent_reg = torch.tensor(0.0, device=device)
    ode_eval_count = int(getattr(model.ode_func, "derivative_eval_count", 0))
    if model.training and ode_eval_count > 0:
        # Memory-safe detached metric from ODE evaluations in this forward pass.
        avg_deriv_energy = model.ode_func.derivative_energy_sum / max(model.ode_func.derivative_eval_count, 1)
        l_latent_reg = torch.tensor(avg_deriv_energy, dtype=torch.float32, device=device)
        model.ode_func.derivative_energy_sum = 0.0
        model.ode_func.derivative_eval_count = 0

    visc_reg_w = max(float(curriculum.biochem_viscosity_regularization_weight), 0.0)
    isolate_key = (os.environ.get("BIOCHEM_LOSS_ISOLATE") or "").strip().upper()
    data_only = _biochem_env_truthy("BIOCHEM_LOSS_DATA_ONLY") and (not isolate_key)
    data_only_phys_temp_wanted = (
        data_only
        and _biochem_env_truthy("BIOCHEM_DATA_ONLY_PHYS_TEMP")
        and w_pt > 0.0
    )

    if isolate_key:
        loss = _biochem_resolve_isolated_loss(
            isolate_key,
            pred_final=pred_final,
            l_adr_fast=l_adr_fast,
            l_adr_slow=l_adr_slow,
            l_wall_bio=l_wall_bio,
            l_wall_phys=l_wall_phys,
            l_bio_io=l_bio_io,
            l_mom=l_mom,
            l_data_kine=l_data_kine,
            l_data_bio=l_data_bio,
            l_pseudo=l_pseudo,
            pseudo_loss_weight=float(pseudo_loss_weight),
            l_latent_reg=l_latent_reg,
            latent_scale=float(latent_scale),
            l_visc_reg=l_visc_reg,
            visc_reg_w=float(visc_reg_w),
            l_kine_prior=l_kine_prior,
            w_kp=float(w_kp),
            l_phys_temp=l_phys_temp,
            w_pt=float(w_pt),
            l_mu_si_anchor=l_mu_si_anchor,
            w_mu_aux=float(w_mu_aux),
            l_mu_log_anchor=l_mu_log_anchor,
            w_mu_log=float(w_mu_log),
            l_mu_log_wall=l_mu_log_wall,
            w_mu_log_wall=float(w_mu_log_wall),
            l_mu_log_high=l_mu_log_high,
            w_mu_log_high=float(w_mu_log_high),
            l_fi_gate_start=l_fi_gate_start,
            w_fi_gate_start_eff=float(w_fi_gate_start_eff),
            l_residual_sparse=l_residual_sparse,
            lambda_residual_sparse=float(lambda_residual_sparse),
        )
    elif data_only:
        if has_anchor_supervision:
            w_mu_log_boundary = max(float(os.environ.get("BIOCHEM_MU_LOG_BOUNDARY_WEIGHT", "0.0")), 0.0)
            loss = (
                l_data_kine
                + l_data_bio
                + (w_mu_aux * l_mu_si_anchor)
                + (w_mu_log * l_mu_log_anchor)
                + (w_mu_log_wall * l_mu_log_wall)
                + (w_mu_log_high * l_mu_log_high)
                + (w_mu_log_boundary * l_mu_log_boundary)
            )
            if model.training and data_only_phys_temp_wanted:
                loss = loss + (w_pt * l_phys_temp)
        elif has_pseudo_supervision and float(pseudo_loss_weight) > 0.0:
            loss = float(pseudo_loss_weight) * l_pseudo
        else:
            loss = pred_final.sum() * 0.0
            if model.training and _biochem_env_truthy("BIOCHEM_ZERO_LOSS_WARN", default=False):
                print(
                    "⚠️ BIOCHEM_LOSS_DATA_ONLY=1: no anchor supervision and no pseudo targets on this batch; "
                    "scalar loss is 0 (grad tether may still attach). Use anchors, raise pseudo weight, or "
                    "unset BIOCHEM_LOSS_DATA_ONLY. (Per-batch prints: BIOCHEM_ZERO_LOSS_WARN=1.)",
                    flush=True,
                )
    else:
        loss = (
            loss_weighter(all_losses, task_active=task_active)
            + (float(pseudo_loss_weight) * l_pseudo)
            + (latent_scale * l_latent_reg)
            + (visc_reg_w * l_visc_reg)
            + (w_kp * l_kine_prior)
            + (w_pt * l_phys_temp)
            + (w_mu_aux * l_mu_si_anchor)
            + (w_mu_log * l_mu_log_anchor)
            + (w_mu_log_wall * l_mu_log_wall)
            + (w_mu_log_high * l_mu_log_high)
            + (max(float(os.environ.get("BIOCHEM_MU_LOG_BOUNDARY_WEIGHT", "0.0")), 0.0) * l_mu_log_boundary)
            + (w_fi_gate_start_eff * l_fi_gate_start)
            + (lambda_residual_sparse * l_residual_sparse)
        )
    if model.training and (w_trigger_gate_floor > 0.0 or w_trigger_learned_floor > 0.0):
        loss = (
            loss
            + (w_trigger_gate_floor * l_trigger_gate_floor)
            + (w_trigger_learned_floor * l_trigger_learned_floor)
        )

    # Guard for sparse pseudo-only windows where every active term can become
    # graph-disconnected (e.g., all-zero residual/mimic terms in low-anchor mode).
    # Tethering with a zero-valued parameter term keeps autograd valid without
    # altering the scalar objective value.
    grad_tether_active = False
    if model.training and (not loss.requires_grad):
        for p in model.parameters():
            if p.requires_grad:
                loss = loss + (p.reshape(-1)[0] * 0.0)
                grad_tether_active = True
                break

    metrics = {
        "L_mom": l_mom.item(),
        "L_ADR_F": l_adr_fast.item(),
        "L_ADR_S": l_adr_slow.item(),
        "L_W_Bio": l_wall_bio.item(),
        "L_W_Phy": l_wall_phys.item(),
        "L_B_IO": l_bio_io.item(),
        # Supervised COMSOL labels on anchor nodes only (Huber / variance-normalized).
        "L_Data_Kine": l_data_kine.item(),
        "L_Data_Bio": l_data_bio.item(),
        "T_eval_Bio": t_eval_bio_for_metrics,
        "L_Pseudo": l_pseudo.item(),
        "L_Latent_Reg": l_latent_reg.item(),
        "L_Visc_Reg": l_visc_reg.item(),
        "W_Wall_Bio": wall_surface_mult,
        "W_Wall_Phy": wall_flux_mult,
        "W_Visc_Reg": visc_reg_w,
        "Huber_Delta_Bio": float(kernels._biochem_huber_delta),
        "TF_eff": float(teacher_forcing_ratio),
        "ODE_Evals": ode_eval_count,
        "Has_Anchor_Supervision": float(has_anchor_supervision),
        "Has_Pseudo_Supervision": float(has_pseudo_supervision),
        "Grad_Tether_Active": float(grad_tether_active),
        "PDE_Steps": float(num_steps),
        "L_KinePrior": l_kine_prior.item(),
        "L_PhysTemp": l_phys_temp.item(),
        "L_MuSI_aux": l_mu_si_anchor.item(),
        "W_MuSI_aux_eff": float(w_mu_aux),
        "L_MuLog_aux": l_mu_log_anchor.item(),
        "W_MuLog_aux_eff": float(w_mu_log),
        "L_MuLog_wall": l_mu_log_wall.item(),
        "W_MuLog_wall_eff": float(w_mu_log_wall),
        "L_MuLog_high": l_mu_log_high.item(),
        "W_MuLog_high_eff": float(w_mu_log_high),
        "L_MuLog_boundary": l_mu_log_boundary.item(),
        "W_MuLog_boundary_eff": max(float(os.environ.get("BIOCHEM_MU_LOG_BOUNDARY_WEIGHT", "0.0")), 0.0),
        "DataOnlyPhysTempOn": float(data_only_phys_temp_wanted),
        "W_DataOnlyPhysTemp_eff": float(w_pt) if data_only_phys_temp_wanted else 0.0,
        "L_FIGateStart": l_fi_gate_start.item(),
        "W_FIGateStart": float(w_fi_gate_start_eff),
        "L_ResidualSparse": l_residual_sparse.item(),
        "W_ResidualSparse": float(lambda_residual_sparse),
        "L_TriggerGateFloor": l_trigger_gate_floor.item(),
        "W_TriggerGateFloor": float(w_trigger_gate_floor),
        "L_TriggerLearnedFloor": l_trigger_learned_floor.item(),
        "W_TriggerLearnedFloor": float(w_trigger_learned_floor),
        "BIOCHEM_LOSS_DATA_ONLY": float(data_only),
        "BIOCHEM_LOSS_ISOLATE": isolate_key or "",
        "L_Backprop": float(loss.detach().item()),
    }
    metrics.update(mu_dbg)
    if debug_batch is not None:
        de, dbi = debug_batch
        _debug_biochem_batch(
            epoch=de,
            batch_idx=dbi,
            data=data,
            pred_series=pred_series,
            all_losses=all_losses,
            task_active=task_active,
            loss_weighter=loss_weighter,
            loss_total=loss,
            l_latent_reg=l_latent_reg,
            metrics=metrics,
            re_ref=re_ref,
            r_lo=r_lo,
            r_hi=r_hi,
            evaluation_times=evaluation_times,
            start_idx=start_idx,
            end_idx=end_idx,
            truth_count=int(truth_mask.sum().item()),
            tbptt_info=tbptt_info,
            bio_cfg=bio_cfg,
            training=bool(model.training),
        )
    return loss, metrics


def _biochem_anchor_basename(data) -> str:
    """Short graph label for per-anchor μ tables (``patient007.pt``, etc.)."""
    src = _biochem_src_for_log(data)
    if src == "<unknown>":
        return src
    return Path(src).name


def _mu_val_high_quantile() -> float:
    try:
        q = float(os.environ.get("BIOCHEM_MU_VAL_HIGH_QUANTILE", "0.9"))
    except ValueError:
        q = 0.9
    return min(max(q, 0.5), 0.999)


def _mu_pair_metrics(mu_p: torch.Tensor, mu_g: torch.Tensor) -> Dict[str, float]:
    """Continuous μ errors on a masked subset (SI viscosity)."""
    empty = {
        "n": 0,
        "mu_mae_si": float("nan"),
        "mu_rmse_si": float("nan"),
        "mu_log_mae": float("nan"),
        "mu_pearson": float("nan"),
        "mu_r2": float("nan"),
    }
    if mu_p.numel() < 1 or mu_g.numel() < 1:
        return empty
    mp = mu_p.float().reshape(-1)
    mg = mu_g.float().reshape(-1)
    err = mp - mg
    out: Dict[str, float] = {
        "n": int(mp.numel()),
        "mu_mae_si": float(err.abs().mean().item()),
        "mu_rmse_si": float(torch.sqrt(err.pow(2).mean() + 1e-20).item()),
        "mu_log_mae": float(
            (
                torch.log(mp.clamp(min=1e-8)) - torch.log(mg.clamp(min=1e-8))
            )
            .abs()
            .mean()
            .item()
        ),
        "mu_pearson": float("nan"),
        "mu_r2": float("nan"),
    }
    if mp.numel() >= 2:
        std_p = mp.std(unbiased=False)
        std_g = mg.std(unbiased=False)
        if std_p > 1e-12 and std_g > 1e-12:
            c = torch.corrcoef(torch.stack([mp, mg]))[0, 1]
            out["mu_pearson"] = float(c.item()) if torch.isfinite(c) else float("nan")
        ss_res = err.pow(2).sum()
        ss_tot = (mg - mg.mean()).pow(2).sum()
        if ss_tot > 1e-20:
            out["mu_r2"] = float((1.0 - ss_res / ss_tot).item())
    return out


def _mu_debug_masks(
    data,
    truth_mask: torch.Tensor,
    mu_gt_si: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Subsets for μ diagnostics: all COMSOL truth, wall band, high-μ (gelation) tail."""
    device = truth_mask.device
    all_m = truth_mask
    wall_m = truth_mask.clone()
    if hasattr(data, "mask_wall"):
        wall_m = truth_mask & data.mask_wall.view(-1).bool().to(device)
    high_m = torch.zeros_like(truth_mask)
    mg = mu_gt_si[truth_mask].float()
    if mg.numel() >= 2:
        q = _mu_val_high_quantile()
        thresh = torch.quantile(mg, q)
        high_m = truth_mask & (mu_gt_si >= thresh)
    bulk_m = truth_mask & ~high_m
    return {"all": all_m, "wall": wall_m, "high_mu": high_m, "bulk": bulk_m}


def _compute_mu_debug_metrics(
    pred_last: torch.Tensor,
    y_last: torch.Tensor,
    data,
    kernels,
    device: torch.device,
) -> Dict[str, Any]:
    """Per-graph μ metrics on truth / wall / high-μ_gt subsets (final time slice)."""
    num_nodes = int(data.num_nodes)
    truth_mask = biochem_truth_node_mask(data, num_nodes, device)
    mu_ch = STATE_CHANNEL_MU_EFF_ND
    phys_cfg = kernels.core.cfg
    bundle: Dict[str, Any] = {
        "anchor": _biochem_anchor_basename(data),
        "n_truth": int(truth_mask.sum().item()),
        "subsets": {},
    }
    if (not truth_mask.any()) or y_last.shape[-1] <= mu_ch:
        return bundle
    mu_pred_si = phys_cfg.viscosity_nd_to_si(pred_last[:, mu_ch])
    mu_gt_si = phys_cfg.viscosity_nd_to_si(y_last[:, mu_ch])
    for name, mask in _mu_debug_masks(data, truth_mask, mu_gt_si).items():
        if int(mask.sum().item()) >= 1:
            bundle["subsets"][name] = _mu_pair_metrics(mu_pred_si[mask], mu_gt_si[mask])
        else:
            bundle["subsets"][name] = _mu_pair_metrics(
                mu_pred_si.new_zeros(0), mu_gt_si.new_zeros(0)
            )
    return bundle


def _aggregate_mu_subset_metrics(per_graph: List[Dict[str, Any]], subset: str) -> Dict[str, float]:
    """Mean subset metrics across graphs that have ``n >= 1`` for that subset."""
    keys = ("mu_mae_si", "mu_rmse_si", "mu_log_mae", "mu_pearson", "mu_r2")
    acc = {k: 0.0 for k in keys}
    n_acc = 0
    for g in per_graph:
        sub = (g.get("subsets") or {}).get(subset)
        if not sub or int(sub.get("n", 0)) < 1:
            continue
        for k in keys:
            v = float(sub.get(k, float("nan")))
            if math.isfinite(v):
                acc[k] += v
        n_acc += 1
    if n_acc == 0:
        return {k: float("nan") for k in keys}
    inv = 1.0 / float(n_acc)
    return {k: acc[k] * inv for k in keys}


def _format_mu_subset_line(label: str, m: Dict[str, float]) -> str:
    n = int(m.get("n", 0)) if "n" in m else -1
    n_s = f" n={n}" if n >= 0 else ""
    r = m.get("mu_pearson", float("nan"))
    r_s = f" r={r:.3f}" if math.isfinite(float(r)) else " r=n/a"
    log_m = m.get("mu_log_mae", float("nan"))
    mae = m.get("mu_mae_si", float("nan"))
    if not math.isfinite(float(log_m)):
        return f"   📐 {label}:{n_s} (no nodes)"
    return f"   📐 {label}:{n_s} logMAE={log_m:.4f} MAE_si={mae:.3e}{r_s}"


def _log_mu_validation_report(
    *,
    stage: str,
    epoch: int,
    per_graph: List[Dict[str, Any]],
    extra_lines: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Console + return flat aggregate dict (``all`` subset) for checkpoint scoring."""
    if not per_graph:
        print(f"   📐 [{stage} ep {epoch:02d}] μ validation: no evaluable graphs.")
        return {
            "mu_mae_si": float("inf"),
            "mu_rmse_si": float("inf"),
            "mu_log_mae": float("inf"),
            "mu_pearson": 0.0,
            "mu_r2": 0.0,
        }
    agg_all = _aggregate_mu_subset_metrics(per_graph, "all")
    agg_wall = _aggregate_mu_subset_metrics(per_graph, "wall")
    agg_high = _aggregate_mu_subset_metrics(per_graph, "high_mu")
    agg_bulk = _aggregate_mu_subset_metrics(per_graph, "bulk")
    q = _mu_val_high_quantile()
    print(f"   📐 [{stage} ep {epoch:02d}] μ subsets (SI; high_μ = GT ≥ p{q:.2f} on truth nodes):")
    print(_format_mu_subset_line("all truth", {**agg_all, "n": sum(
        int((g.get("subsets") or {}).get("all", {}).get("n", 0)) for g in per_graph
    )}))
    print(_format_mu_subset_line("wall ∩ truth", agg_wall))
    print(_format_mu_subset_line("high-μ_gt tail", agg_high))
    print(_format_mu_subset_line("bulk (rest)", agg_bulk))
    rows: List[Tuple[str, float, float, float]] = []
    for g in per_graph:
        name = str(g.get("anchor", "?"))
        sub = (g.get("subsets") or {}).get("all", {})
        log_m = float(sub.get("mu_log_mae", float("nan")))
        log_w = float((g.get("subsets") or {}).get("wall", {}).get("mu_log_mae", float("nan")))
        log_h = float((g.get("subsets") or {}).get("high_mu", {}).get("mu_log_mae", float("nan")))
        if math.isfinite(log_m):
            rows.append((name, log_m, log_w, log_h))
    if rows:
        print("   Per-anchor logMAE (all | wall | high-μ_gt):")
        for name, la, lw, lh in sorted(rows, key=lambda t: t[1], reverse=True):
            lw_s = f"{lw:.4f}" if math.isfinite(lw) else "n/a"
            lh_s = f"{lh:.4f}" if math.isfinite(lh) else "n/a"
            print(f"      {name:24s}  all={la:.4f}  wall={lw_s}  high={lh_s}")
    if extra_lines:
        for line in extra_lines:
            print(line)
    out = dict(agg_all)
    out["mu_log_mae_wall"] = float(agg_wall.get("mu_log_mae", float("nan")))
    out["mu_log_mae_high_mu"] = float(agg_high.get("mu_log_mae", float("nan")))
    out["mu_log_mae_bulk"] = float(agg_bulk.get("mu_log_mae", float("nan")))
    return out


def calculate_validation_metrics(pred, data, kernels, device):
    props = kernels.core._get_geometric_props(data)

    num_nodes = int(data.num_nodes)
    truth_mask = biochem_truth_node_mask(data, num_nodes, pred.device)
    has_truth = bool(truth_mask.any().item())

    if pred.shape[0] != num_nodes:
        raise ValueError(
            "calculate_validation_metrics: pred rows must equal data.num_nodes "
            f"({pred.shape[0]} != {num_nodes})."
        )
    if data.y.dim() != 3:
        raise ValueError(
            "calculate_validation_metrics expects data.y shaped [T, N, C] (phase-3 trajectories); "
            f"got {tuple(data.y.shape)}."
        )
    if data.y.shape[1] != num_nodes:
        raise ValueError(
            "calculate_validation_metrics: data.y spatial dim must match num_nodes "
            f"({data.y.shape[1]} != {num_nodes})."
        )

    y_last = data.y[-1]

    mu_ch = STATE_CHANNEL_MU_EFF_ND
    mu_eff_nd = pred[ :, mu_ch ]

    mu_pred_dimensional = kernels.core.cfg.viscosity_nd_to_si(mu_eff_nd)

    mu_mae_si = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    mu_rmse_si = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    mu_log_mae = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    mu_pearson = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    mu_r2 = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    mu_debug: Dict[str, Any] = {}

    has_mu_labels = data.y.shape[-1] > mu_ch + 1
    if has_truth and has_mu_labels:
        mu_debug = _compute_mu_debug_metrics(pred, y_last, data, kernels, pred.device)
        sub_all = (mu_debug.get("subsets") or {}).get("all", {})
        if int(sub_all.get("n", 0)) >= 1:
            mu_mae_si = torch.tensor(float(sub_all["mu_mae_si"]), device=pred.device, dtype=pred.dtype)
            mu_rmse_si = torch.tensor(float(sub_all["mu_rmse_si"]), device=pred.device, dtype=pred.dtype)
            mu_log_mae = torch.tensor(float(sub_all["mu_log_mae"]), device=pred.device, dtype=pred.dtype)
            mu_pearson = torch.tensor(
                float(sub_all["mu_pearson"]) if math.isfinite(float(sub_all["mu_pearson"])) else 0.0,
                device=pred.device,
                dtype=pred.dtype,
            )
            mu_r2 = torch.tensor(
                float(sub_all["mu_r2"]) if math.isfinite(float(sub_all["mu_r2"])) else 0.0,
                device=pred.device,
                dtype=pred.dtype,
            )

    # --- Hemodynamic Metric: WSS Pearson (patent lumen, COMSOL-trusted wall nodes only) ---
    mask_wall = data.mask_wall.view(-1).bool()
    has_wall = bool(mask_wall.any().item())
    zero_pearson = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    wss_diag: Dict[str, Any] = {
        "wss_pearson_reason": "unset",
        "patent_wall_count": 0,
        "std_wss_pred": None,
        "std_wss_targ": None,
        "mu_debug": mu_debug,
        "anchor": _biochem_anchor_basename(data),
    }
    pearson_corr = zero_pearson
    c_u_p = None
    c_v_p = None

    if not has_truth:
        wss_diag["wss_pearson_reason"] = "no_comsol_truth_nodes"
    elif not has_wall:
        wss_diag["wss_pearson_reason"] = "no_wall_mask"
    elif data.y.shape[-1] <= 1:
        wss_diag["wss_pearson_reason"] = "insufficient_label_channels"
    else:
        patent_wall_mask = mask_wall & truth_mask

        if not patent_wall_mask.any():
            wss_diag["wss_pearson_reason"] = "empty_patent_wall_mask"
        else:
            wss_diag["patent_wall_count"] = int(patent_wall_mask.sum().item())
            # Compute Predicted WSS ([N,1] fields for WLS — same contract as physics_kernels)
            c_u_p = kernels.core._compute_derivatives(pred[ :, 0:1 ], props)
            c_v_p = kernels.core._compute_derivatives(pred[ :, 1:2 ], props)
            dudx_p, dudy_p = c_u_p[ :, 0, 0 ], c_u_p[ :, 1, 0 ]
            dvdx_p, dvdy_p = c_v_p[ :, 0, 0 ], c_v_p[ :, 1, 0 ]

            mu_wall_p = pred[ patent_wall_mask, mu_ch ]
            tau_xx_p = 2.0 * mu_wall_p * dudx_p[ patent_wall_mask ]
            tau_yy_p = 2.0 * mu_wall_p * dvdy_p[ patent_wall_mask ]
            tau_xy_p = mu_wall_p * (dudy_p[ patent_wall_mask ] + dvdx_p[ patent_wall_mask ])

            nx = data.x[ patent_wall_mask, 3 ]
            ny = data.x[ patent_wall_mask, 4 ]

            tx_p, ty_p = tau_xx_p * nx + tau_xy_p * ny, tau_xy_p * nx + tau_yy_p * ny
            tn_p = tx_p * nx + ty_p * ny
            wss_pred = torch.sqrt((tx_p - tn_p * nx) ** 2 + (ty_p - tn_p * ny) ** 2 + 1e-8)

            # Ground-truth WSS from final timestep velocities (same [N, C] layout as pred)
            c_u_t = kernels.core._compute_derivatives(y_last[ :, 0:1 ], props)
            c_v_t = kernels.core._compute_derivatives(y_last[ :, 1:2 ], props)
            dudx_t, dudy_t = c_u_t[ :, 0, 0 ], c_u_t[ :, 1, 0 ]
            dvdx_t, dvdy_t = c_v_t[ :, 0, 0 ], c_v_t[ :, 1, 0 ]

            mu_wall_t = (
                y_last[ patent_wall_mask, mu_ch ]
                if data.y.shape[ -1 ] > mu_ch + 1
                else torch.ones_like(mu_wall_p)
            )
            tau_xx_t = 2.0 * mu_wall_t * dudx_t[ patent_wall_mask ]
            tau_yy_t = 2.0 * mu_wall_t * dvdy_t[ patent_wall_mask ]
            tau_xy_t = mu_wall_t * (dudy_t[ patent_wall_mask ] + dvdx_t[ patent_wall_mask ])

            tx_t, ty_t = tau_xx_t * nx + tau_xy_t * ny, tau_xy_t * nx + tau_yy_t * ny
            tn_t = tx_t * nx + ty_t * ny
            wss_targ = torch.sqrt((tx_t - tn_t * nx) ** 2 + (ty_t - tn_t * ny) ** 2 + 1e-8)

            min_std = 1e-12
            if wss_pred.numel() < 2:
                wss_diag["wss_pearson_reason"] = "too_few_patent_wall_points_for_correlation"
                pearson_corr = zero_pearson
            else:
                std_p = wss_pred.std(unbiased=False)
                std_t = wss_targ.std(unbiased=False)
                wss_diag["std_wss_pred"] = float(std_p.item())
                wss_diag["std_wss_targ"] = float(std_t.item())
                if std_p < min_std and std_t < min_std:
                    wss_diag["wss_pearson_reason"] = "both_wss_vectors_near_constant"
                    pearson_corr = zero_pearson
                elif std_p < min_std:
                    wss_diag["wss_pearson_reason"] = "pred_wss_near_constant_on_patent_wall"
                    pearson_corr = zero_pearson
                elif std_t < min_std:
                    wss_diag["wss_pearson_reason"] = "comsol_wss_near_constant_on_patent_wall"
                    pearson_corr = zero_pearson
                else:
                    stacked = torch.stack([wss_pred, wss_targ])
                    pearson_corr = torch.corrcoef(stacked)[ 0, 1 ]
                    if torch.isnan(pearson_corr):
                        wss_diag["wss_pearson_reason"] = "corrcoef_nan"
                        pearson_corr = zero_pearson
                    else:
                        wss_diag["wss_pearson_reason"] = "ok"

    # Species channels are stored as log1p(species_nd). Convert FI to SI for reporting.
    fi_log1p = pred[:, 12]
    fi_scale = kernels.cfg.get_species_scales(device=pred.device)
    # Back-compat: some configs return a vector of species scales (FI at index 8).
    if torch.is_tensor(fi_scale) and fi_scale.numel() > 1:
        fi_scale = fi_scale.view(-1)[8]
    max_fibrin_pred = torch.clamp(torch.expm1(fi_log1p), min=0.0).max().mul(fi_scale).item()

    # --- NEW: Kinematic / Fluid Physics & Cascade Metrics ---
    # 1. Continuity Error (Mass Conservation - Interior Only)
    if c_u_p is None or c_v_p is None:
        c_u_p = kernels.core._compute_derivatives(pred[:, 0:1], props)
        c_v_p = kernels.core._compute_derivatives(pred[:, 1:2], props)
    div_u = c_u_p[:, 0, 0] + c_v_p[:, 1, 0]
    interior = kernels.core.fluid_interior_mask(data)
    continuity_err = torch.tensor(0.0, device=pred.device)
    if interior.any():
        continuity_err = torch.abs(div_u.view(-1)[interior]).mean()

    # 1b. Wall Slip Error (No-Slip Explicit Check)
    wall_slip_err = torch.tensor(0.0, device=pred.device)
    if has_wall:
        wall_vel = torch.norm(pred[data.mask_wall, :2], p=2, dim=1)
        wall_slip_err = wall_vel.mean()

    # 2. Kinematic Relative L2 & Intermediate Species Errors
    rel_l2_kine = torch.tensor(0.0, device=pred.device)
    rp_mae = torch.tensor(0.0, device=pred.device)
    t_mae = torch.tensor(0.0, device=pred.device)

    if has_truth and has_mu_labels:
        # Fluid velocity Rel L2
        p_uv = pred[truth_mask, :2]
        t_uv = y_last[truth_mask, :2]
        rel_l2_kine = torch.norm(p_uv - t_uv, p=2) / (torch.norm(t_uv, p=2) + 1e-8)

        # Intermediate Species Errors (RP = channel 4, Thrombin T = channel 9)
        rp_mae = F.l1_loss(pred[truth_mask, 4], y_last[truth_mask, 4])
        t_mae = F.l1_loss(pred[truth_mask, 9], y_last[truth_mask, 9])

    return (
        pearson_corr.item(),
        max_fibrin_pred,
        wss_diag,
        continuity_err.item(),
        wall_slip_err.item(),
        rel_l2_kine.item(),
        rp_mae.item(),
        t_mae.item(),
        mu_mae_si.item(),
        mu_rmse_si.item(),
        mu_log_mae.item(),
        mu_pearson.item(),
        mu_r2.item(),
    )


def _biochem_save_val_debug_plot(
    out_dir,
    epoch: int,
    pred_last,
    v_data,
    kernels,
    device: torch.device,
) -> None:
    """Sparse validation figure: COMSOL vs predicted viscosity on truth nodes (1:1 scatter)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    truth_mask = biochem_truth_node_mask(v_data, int(v_data.num_nodes), device)
    if not truth_mask.any():
        return
    phys_cfg = kernels.core.cfg
    mu_ch = STATE_CHANNEL_MU_EFF_ND
    p_mu = phys_cfg.viscosity_nd_to_si(pred_last[:, mu_ch][truth_mask]).detach().float().cpu().numpy()
    y_last = v_data.y[-1].to(device)
    g_mu = phys_cfg.viscosity_nd_to_si(y_last[:, mu_ch][truth_mask]).detach().float().cpu().numpy()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.scatter(g_mu, p_mu, s=2, alpha=0.35, c="C0")
    lo = float(min(float(g_mu.min()), float(p_mu.min())))
    hi = float(max(float(g_mu.max()), float(p_mu.max())))
    ax.plot([lo, hi], [lo, hi], "r--", lw=1, alpha=0.8)
    ax.set_xlabel("COMSOL μ (SI)")
    ax.set_ylabel("Pred μ (SI)")
    ax.set_title(f"Val μ (truth nodes) epoch {epoch}")
    fig.tight_layout()
    fig.savefig(out_dir / f"biochem_val_mu_epoch_{epoch:04d}.png", dpi=120)
    plt.close(fig)


def _compute_anchor_mu_metrics(
    model,
    loader,
    kernels,
    bio_cfg,
    device,
    non_blocking: bool = False,
) -> Dict[str, Any]:
    """Anchor validation μ metrics: aggregate ``all`` subset + per-graph debug bundles."""
    empty: Dict[str, Any] = {
        "mu_mae_si": float("inf"),
        "mu_rmse_si": float("inf"),
        "mu_log_mae": float("inf"),
        "mu_pearson": 0.0,
        "mu_r2": 0.0,
        "per_graph": [],
    }
    if len(loader) == 0:
        return empty
    model.eval()
    per_graph: List[Dict[str, Any]] = []

    with torch.no_grad():
        for v_data in loader:
            v_data = v_data.to(device, non_blocking=non_blocking)
            val_eval_times = _validation_eval_times(v_data, bio_cfg, device)
            v_pred = model(v_data, val_eval_times)
            if isinstance(v_pred, tuple):
                v_pred = v_pred[0]
            pred_last = v_pred[-1]
            y_last = v_data.y[-1].to(device)
            mu_dbg = _compute_mu_debug_metrics(pred_last, y_last, v_data, kernels, device)
            if int((mu_dbg.get("subsets") or {}).get("all", {}).get("n", 0)) >= 1:
                per_graph.append(mu_dbg)

    if not per_graph:
        return empty

    agg = _aggregate_mu_subset_metrics(per_graph, "all")
    agg_wall = _aggregate_mu_subset_metrics(per_graph, "wall")
    agg_high = _aggregate_mu_subset_metrics(per_graph, "high_mu")
    out: Dict[str, Any] = {
        "mu_mae_si": float(agg["mu_mae_si"]),
        "mu_rmse_si": float(agg["mu_rmse_si"]),
        "mu_log_mae": float(agg["mu_log_mae"]),
        "mu_pearson": float(agg["mu_pearson"]) if math.isfinite(float(agg["mu_pearson"])) else 0.0,
        "mu_r2": float(agg["mu_r2"]) if math.isfinite(float(agg["mu_r2"])) else 0.0,
        "mu_log_mae_wall": float(agg_wall.get("mu_log_mae", float("nan"))),
        "mu_log_mae_high_mu": float(agg_high.get("mu_log_mae", float("nan"))),
        "per_graph": per_graph,
    }
    return out


def _teacher_anchor_preflight_metrics(teacher, data, kernels, bio_cfg, device) -> Optional[Dict[str, float]]:
    """
    One macro-step forward aligned with loss: GT species at t0, TF=1, compare μ/FI at t1 vs ``y[1]``.
    Returns None if this graph cannot be evaluated (missing y, too-short time grid, no truth nodes).
    """
    if not hasattr(data, "y") or data.y is None or data.y.dim() != 3:
        return None
    full_times = to_t_nd(bio_cfg.resolve_biochem_times(data, device), bio_cfg.t_final)
    n_time = int(data.y.shape[0])
    n_win = min(2, n_time, int(full_times.numel()))
    if n_win < 2:
        return None
    mu_ch = STATE_CHANNEL_MU_EFF_ND
    if data.y.shape[-1] <= mu_ch:
        return None
    eval_t = full_times[:n_win]
    dt_nd = float((eval_t[1] - eval_t[0]).detach().cpu().item())
    y_win = data.y[:n_win].to(device)
    truth_mask = biochem_truth_node_mask(data, int(data.num_nodes), device)
    if not truth_mask.any():
        return None

    with torch.no_grad():
        pred = teacher(
            data,
            eval_t,
            y_true_trajectory=y_win,
            teacher_forcing_ratio=1.0,
            start_idx=0,
            detach_macro_state=True,
        )
    if isinstance(pred, tuple):
        pred = pred[0]
    pred_last = pred[-1]
    gt_idx = n_win - 1
    mu_pred_si = kernels.core.cfg.viscosity_nd_to_si(pred_last[:, mu_ch])
    mp = mu_pred_si[truth_mask]

    mu_gt_si = kernels.core.cfg.viscosity_nd_to_si(data.y[gt_idx, :, mu_ch].to(device))
    mg = mu_gt_si[truth_mask]
    eps_mu = 1e-8
    mu_log_mae = float(
        (torch.log(mp.float().clamp(min=eps_mu)) - torch.log(mg.float().clamp(min=eps_mu)))
        .abs()
        .mean()
        .item()
    )
    mu_mae_si = float((mp.float() - mg.float()).abs().mean().item())

    fi_pred_si = teacher.species_log_nd_to_si(pred_last[:, 4:16])[:, 8]
    fi_gt_si = teacher.species_log_nd_to_si(data.y[gt_idx, :, 4:16].to(device))[:, 8]
    fi_p = fi_pred_si[truth_mask]
    fi_g = fi_gt_si[truth_mask]
    fi_gt_mean = float(fi_g.mean().item())
    fi_pred_mean = float(fi_p.mean().item())

    return {
        "dt_nd": dt_nd,
        "mu_log_mae": mu_log_mae,
        "mu_mae_si": mu_mae_si,
        "mu_pred_mean": float(mp.mean().item()),
        "mu_pred_p90": float(torch.quantile(mp, 0.9).item()),
        "fi_pred_mean": fi_pred_mean,
        "fi_pred_p99": float(torch.quantile(fi_p, 0.99).item()),
        "fi_gt_mean": fi_gt_mean,
        "fi_gt_p99": float(torch.quantile(fi_g, 0.99).item()),
    }


def _median(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    m = len(s) // 2
    return float(s[m]) if len(s) % 2 else 0.5 * (s[m - 1] + s[m])


def _teacher_run_train_anchor_preflight(
    teacher,
    train_anchor_dataset,
    kernels,
    bio_cfg,
    device,
) -> None:
    """Run t0→t1 sanity on every training anchor; abort uses continuous μ_log_MAE on truth nodes."""
    lim_med = max(float(os.environ.get("BIOCHEM_PREFLIGHT_ABORT_MEDIAN_LOG_MAE", "2.5")), 1e-6)
    lim_worst = max(float(os.environ.get("BIOCHEM_PREFLIGHT_ABORT_WORST_LOG_MAE", "4.0")), lim_med)
    policy = (os.environ.get("BIOCHEM_PREFLIGHT_POLICY", "median") or "median").strip().lower()
    fi_ratio_lim = max(float(os.environ.get("BIOCHEM_ABORT_FI_GT_RATIO", "200")), 1.0)

    log_maes: List[float] = []
    mae_sis: List[float] = []
    fi_means_p: List[float] = []
    fi_means_g: List[float] = []
    per_idx: List[Tuple[int, float, float]] = []
    dt0: Optional[float] = None

    for idx in range(len(train_anchor_dataset)):
        data = train_anchor_dataset[idx].to(device)
        m = _teacher_anchor_preflight_metrics(teacher, data, kernels, bio_cfg, device)
        if m is None:
            continue
        if dt0 is None:
            dt0 = m["dt_nd"]
        log_maes.append(m["mu_log_mae"])
        mae_sis.append(m["mu_mae_si"])
        fi_means_p.append(m["fi_pred_mean"])
        fi_means_g.append(m["fi_gt_mean"])
        per_idx.append((idx, m["mu_log_mae"], m["mu_mae_si"]))

    if not log_maes:
        print(
            "   🔎 Teacher preflight: skipped (no evaluable train anchors with y, time grid, truth nodes)."
        )
        return

    med_log = _median(log_maes)
    med_mae = _median(mae_sis)
    worst_log = max(log_maes)
    worst_mae = max(mae_sis)
    dt_note = f"Δt_nd≈{dt0:.4e}" if dt0 is not None else "Δt_nd=n/a"

    hard_cap_raw = (os.environ.get("BIOCHEM_PREFLIGHT_ABORT_MAX_LOG_MAE") or "").strip()
    hard_cap: Optional[float] = float(hard_cap_raw) if hard_cap_raw else None
    hard_note = (
        f" | hard cap: worst μ_log_MAE > {hard_cap:.4f} (BIOCHEM_PREFLIGHT_ABORT_MAX_LOG_MAE)"
        if hard_cap is not None
        else ""
    )

    pol_note = (
        f"median μ_log_MAE > {lim_med:.4f}"
        if policy != "max"
        else f"worst μ_log_MAE > {lim_worst:.4f}"
    )
    print(
        f"   🔎 Teacher preflight ({len(log_maes)} train anchor(s), IC=GT t0→t1, {dt_note}): "
        f"μ_log_MAE median={med_log:.4f} worst={worst_log:.4f} | "
        f"μ_MAE_si median={med_mae:.3e} worst={worst_mae:.3e} | policy={policy!r} "
        f"(abort if {pol_note}{hard_note})"
    )
    if _biochem_env_truthy("BIOCHEM_PREFLIGHT_VERBOSE", default=False):
        for idx, lm, ma in sorted(per_idx, key=lambda t: -t[1]):
            print(f"      anchor[{idx}] μ_log_MAE={lm:.4f}  μ_MAE_si={ma:.3e}")

    if not _biochem_env_truthy("BIOCHEM_ABORT_BAD_TEACHER_INIT", default=False):
        return

    if hard_cap is not None and worst_log > hard_cap:
        raise RuntimeError(
            "BIOCHEM_ABORT_BAD_TEACHER_INIT: at least one anchor exceeds BIOCHEM_PREFLIGHT_ABORT_MAX_LOG_MAE "
            f"(worst μ_log_MAE={worst_log:.4f} > {hard_cap:.4f}). "
            "Raise the cap, fix ODE/init/scale, or set BIOCHEM_ABORT_BAD_TEACHER_INIT=0."
        )

    if policy == "max":
        stat = worst_log
        lim_cmp = lim_worst
    else:
        stat = med_log
        lim_cmp = lim_med

    if stat > lim_cmp:
        raise RuntimeError(
            "BIOCHEM_ABORT_BAD_TEACHER_INIT: μ_log_MAE preflight threshold exceeded "
            f"(policy={policy!r}, stat={stat:.4f} > {lim_cmp:.4f}). "
            "Tune BIOCHEM_PREFLIGHT_ABORT_MEDIAN_LOG_MAE / BIOCHEM_PREFLIGHT_ABORT_WORST_LOG_MAE, "
            "or set BIOCHEM_ABORT_BAD_TEACHER_INIT=0 to skip."
        )

    # Mean-FI ratio: use median over anchors so one outlier graph does not dominate.
    # At t0→t₁ preflight, COMSOL FI on truth nodes is often tiny-but-nonzero; dividing by ~1e-12
    # SI yields meaningless ×1000 "ratios" that are not ODE scale blow-ups. Only compare when
    # GT mean FI clears a species-scale floor (override with BIOCHEM_PREFLIGHT_FI_GT_MIN_SI).
    raw_fi_floor = (os.environ.get("BIOCHEM_PREFLIGHT_FI_GT_MIN_SI") or "").strip()
    if raw_fi_floor:
        fi_floor_si = max(float(raw_fi_floor), 1e-30)
    else:
        fi_scale = float(bio_cfg.get_species_scales(device=device)[8].detach().float().cpu().item())
        frac_raw = (os.environ.get("BIOCHEM_PREFLIGHT_FI_GT_MIN_FRAC_OF_SCALE") or "1e-5").strip()
        frac = max(float(frac_raw or "1e-5"), 1e-12)
        fi_floor_si = max(1e-18, fi_scale * frac)
    ratios: List[float] = []
    for i in range(len(fi_means_g)):
        g = fi_means_g[i]
        if g < fi_floor_si:
            continue
        ratios.append(fi_means_p[i] / max(g, 1e-30))
    if ratios:
        med_ratio = _median(ratios)
        if med_ratio > fi_ratio_lim:
            raise RuntimeError(
                f"BIOCHEM_ABORT_BAD_TEACHER_INIT: median FI_pred/FI_gt ({med_ratio:.1f}×) > "
                f"{fi_ratio_lim:.0f}× (among anchors with mean GT FI ≥ {fi_floor_si:.3e} SI). "
                "Lower BIOCHEM_ABORT_FI_GT_RATIO, set BIOCHEM_PREFLIGHT_FI_GT_MIN_SI / "
                "BIOCHEM_PREFLIGHT_FI_GT_MIN_FRAC_OF_SCALE, or set BIOCHEM_ABORT_BAD_TEACHER_INIT=0 to skip."
            )


def train_teacher_on_anchors(
    student_model,
    train_anchor_dataset,
    val_anchor_dataset,
    kernels,
    bio_cfg,
    curriculum,
    device,
    base_lr,
    low_anchor_mode: bool = False,
):
    """Train a teacher only on anchor graphs for pseudo-label distillation."""
    if len(train_anchor_dataset) == 0:
        print("⚠️ Teacher stage skipped: no anchor graphs in training split.")
        return None, 0.0

    teacher = copy.deepcopy(student_model).to(device)
    dl_kw = _biochem_dataloader_kw(device)
    nb_xfer = _biochem_non_blocking_transfer(device, dl_kw)
    try:
        teacher_val_every = max(1, int(os.environ.get("BIOCHEM_TEACHER_VAL_EVERY", "5")))
    except ValueError:
        teacher_val_every = 2
    # Deterministic order: preflight scans all anchors; the guard uses median(μ_log_MAE),
    # not one random batch from shuffle=True (which falsely aborted on a single hard graph).
    teacher_loader = DataLoader(train_anchor_dataset, batch_size=1, shuffle=False, **dl_kw)
    teacher_val_loader = DataLoader(val_anchor_dataset, batch_size=1, shuffle=False, **dl_kw)
    train_anchor_keys = {str(p) for p in getattr(train_anchor_dataset, "file_list", [])}
    val_anchor_keys = {str(p) for p in getattr(val_anchor_dataset, "file_list", [])}
    overlap = train_anchor_keys & val_anchor_keys
    overlap_ratio = (
        len(overlap) / max(1, len(val_anchor_keys))
        if len(val_anchor_keys) > 0 else 0.0
    )
    if len(train_anchor_keys) == 1 and train_anchor_keys == val_anchor_keys:
        print(
            "⚠️ One-anchor regime: teacher train/val use the same anchor file. "
            "Treat teacher val μ metrics as pipeline-health signals, not generalization."
        )
    elif overlap_ratio > 0.0:
        print(
            f"⚠️ Teacher anchor split overlap: {len(overlap)}/{max(1, len(val_anchor_keys))} "
            f"({overlap_ratio:.1%}) shared between train and val."
        )

    if (
        "BIOCHEM_TEACHER_EPOCHS" not in os.environ
        and "BIOCHEM_TEACHER_MAX_EPOCHS" not in os.environ
        and not low_anchor_mode
    ):
        os.environ["BIOCHEM_TEACHER_EPOCHS"] = "40"

    # Backward-compatible env lookup: prefer BIOCHEM_TEACHER_EPOCHS when provided.
    teacher_epochs_env = os.environ.get("BIOCHEM_TEACHER_EPOCHS")
    if teacher_epochs_env is None:
        teacher_epochs_env = os.environ.get("BIOCHEM_TEACHER_MAX_EPOCHS", "100")
    max_epochs = max(1, int(teacher_epochs_env))
    _teacher_stage_best_practice_defaults(max_epochs)
    train_cfg = BiochemTrainingConfig.from_env()
    if "BIOCHEM_TEACHER_LR" in os.environ and os.environ["BIOCHEM_TEACHER_LR"].strip() != "":
        teacher_lr = float(os.environ["BIOCHEM_TEACHER_LR"].strip())
    else:
        # Default fallback when env is missing/blank.
        teacher_lr = 2e-3
    _target_mu_raw = (os.environ.get("BIOCHEM_TEACHER_TARGET_MU_LOG_MAE") or "").strip()
    target_mu_log_mae: Optional[float] = None
    if _target_mu_raw:
        target_mu_log_mae = float(_target_mu_raw)
    accumulation_steps = max(1, int(os.environ.get("BIOCHEM_TEACHER_ACCUMULATION_STEPS", "1")))
    clip_teacher = float(os.environ.get("BIOCHEM_TEACHER_CLIP_NORM", "1.0"))
    clip_teacher_phys = float(os.environ.get("BIOCHEM_TEACHER_PHYSICS_CLIP_NORM", "0.1"))
    teacher_curriculum = copy.deepcopy(curriculum)
    try:
        tf_wu = int(os.environ.get("BIOCHEM_TEACHER_TF_WARMUP_EPOCHS", "5"))
    except ValueError:
        tf_wu = 5
    tf_wu = max(1, min(tf_wu, max_epochs))
    teacher_curriculum.biochem_warmup_epochs = tf_wu
    # Finish the post-warmup TF decay by the last epoch (epoch index runs ``0 .. max_epochs-1``).
    rem = max(0, (max_epochs - 1) - tf_wu)
    teacher_curriculum.biochem_teacher_force_decay_epochs = max(1, rem)

    teacher_weighter = make_biochem_dynamic_loss_weighter(curriculum, device)
    # Freeze Kendall log_vars so data-side uncertainty does not mute COMSOL targets.
    teacher_weighter.log_vars.requires_grad_(False)
    teacher_optimizer = setup_biochem_optimization(
        teacher, teacher_weighter, base_lr=teacher_lr, freeze_lora=True
    )
    best_state = None
    best_mu_score = -float("inf")
    best_epoch = -1
    best_mu_log_all = float("inf")
    best_mu_log_high = float("inf")
    pareto_ckpt = _biochem_env_truthy("BIOCHEM_TEACHER_PARETO_CHECKPOINT", default=False)
    pareto_all_tol = max(float(os.environ.get("BIOCHEM_TEACHER_PARETO_ALL_TOL", "0.03")), 0.0)
    pareto_high_tol = max(float(os.environ.get("BIOCHEM_TEACHER_PARETO_HIGH_TOL", "0.05")), 0.0)
    pareto_all_gain = max(float(os.environ.get("BIOCHEM_TEACHER_PARETO_ALL_GAIN_MIN", "0.002")), 0.0)
    pareto_high_gain = max(float(os.environ.get("BIOCHEM_TEACHER_PARETO_HIGH_GAIN_MIN", "0.01")), 0.0)

    _early_stop_note = (
        f"early_stop@mu_log_mae<={target_mu_log_mae:.3f}"
        if target_mu_log_mae is not None
        else "early_stop=off (set BIOCHEM_TEACHER_TARGET_MU_LOG_MAE to enable)"
    )
    print(
        f"\n👩‍🏫 --- Teacher Stage (anchors only): max_epochs={max_epochs}, "
        f"{_early_stop_note}, teacher_lr={teacher_lr:.3e}, "
        f"accum={accumulation_steps}, bio_clip={clip_teacher:.2f}, phys_clip={clip_teacher_phys:.2f}, "
        f"tf_warmup_epochs={teacher_curriculum.biochem_warmup_epochs} ---"
    )
    if pareto_ckpt:
        print(
            "   🧭 Pareto checkpointing enabled: accept checkpoints only when "
            "all/high-μ tradeoff improves within configured tolerances."
        )
    if _biochem_env_truthy("BIOCHEM_LOSS_DATA_ONLY") and _biochem_env_truthy(
        "BIOCHEM_DETACH_MACRO_STATE", default=False
    ):
        print(
            "   ⚠️ BIOCHEM_DETACH_MACRO_STATE=1 with BIOCHEM_LOSS_DATA_ONLY=1: species state is detached "
            "each macro ODE step, so TBPTT gradients to bio_encoder/ODE/decoder are truncated — expect "
            "L_Data_Kine/L_Data_Bio to move slowly vs BIOCHEM_DETACH_MACRO_STATE=0 (VRAM trade-off)."
        )
    # Preflight must see the same rheology caps as epoch 0 training (the loop sets these each epoch).
    # Otherwise ``mu_ratio_max`` stays at ``bio_cfg.mu_ratio_max`` (~80) and μ blows up on every anchor.
    decay0 = 0.0
    t_scale0 = curriculum.biochem_t_scale_warmup_initial - decay0 * (
        curriculum.biochem_t_scale_warmup_initial - curriculum.biochem_t_scale_warmup_final
    )
    teacher.T_scale = t_scale0
    kernels.kinetics.T_scale = t_scale0
    teacher.mu_ratio_max = _biochem_teacher_mu_ratio_max(bio_cfg)
    teacher.train()
    _teacher_run_train_anchor_preflight(teacher, train_anchor_dataset, kernels, bio_cfg, device)
    early_stop_allowed = not low_anchor_mode and overlap_ratio == 0.0
    if not early_stop_allowed:
        print("   ℹ️ Low-anchor mode: disabling teacher μ early-stop to avoid misleading stop signals.")

    # TBPTT random anchor starts (default in ``compute_biochem_loss``) trigger a no_grad warmup
    # rollout from t0→start_idx on almost every batch — often tens of ODE macro-steps with no
    # console output until the first epoch ends. Teacher anchors rarely need that exploration;
    # opt back in with ``BIOCHEM_TEACHER_TBPTT_RANDOM_ANCHOR=1``.
    _prev_tbptt_anchor_rand = os.environ.get("BIOCHEM_TBPTT_ANCHOR_RANDOM_START")
    teacher_tbptt_random = (os.environ.get("BIOCHEM_TEACHER_TBPTT_RANDOM_ANCHOR", "0") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if not teacher_tbptt_random and not _biochem_env_truthy("BIOCHEM_TBPTT_ANCHOR_END_BIAS", default=False):
        os.environ["BIOCHEM_TBPTT_ANCHOR_RANDOM_START"] = "0"
    try:
        ema_beta = float(os.environ.get("BIOCHEM_TEACHER_LBIO_EMA_BETA", "0.9"))
        ema_beta = min(max(ema_beta, 0.0), 0.999999)
        ema_l_bio: Optional[float] = None
        for epoch in range(max_epochs):
            freeze_ode_epochs = max(0, int(os.environ.get("BIOCHEM_TEACHER_ODE_FREEZE_EPOCHS", "3")))
            ode_frozen = epoch < freeze_ode_epochs
            for p in teacher.ode_func.parameters():
                p.requires_grad = not ode_frozen
            if epoch == 0 and freeze_ode_epochs > 0:
                print(f"   🧊 Teacher startup: freezing ODE reaction path for first {freeze_ode_epochs} epoch(s).")
            if epoch == freeze_ode_epochs and freeze_ode_epochs > 0:
                print("   🔓 Teacher startup: unfreezing ODE reaction path.")
            # Decay softened kinetics across teacher stage to progressively sharpen boundaries.
            decay_progress = epoch / float(max(1, max_epochs - 1))
            current_T_scale = curriculum.biochem_t_scale_warmup_initial - decay_progress * (
                curriculum.biochem_t_scale_warmup_initial - curriculum.biochem_t_scale_warmup_final
            )
            teacher.T_scale = current_T_scale
            kernels.kinetics.T_scale = current_T_scale
            # Optional explicit trigger-gate sharpening schedule for split μ heads.
            gate_t_start = float(os.environ.get("BIOCHEM_MU_TRIGGER_GATE_TEMP_START", "0.20"))
            gate_t_end = float(os.environ.get("BIOCHEM_MU_TRIGGER_GATE_TEMP_END", "0.08"))
            if hasattr(teacher, "mu_trigger_gate_temp"):
                gate_temp_now = gate_t_start + (gate_t_end - gate_t_start) * decay_progress
                teacher.mu_trigger_gate_temp = max(gate_temp_now, 1e-5)
            teacher_phys_ceiling = float(
                os.environ.get(
                    "BIOCHEM_TEACHER_PHYSICS_PRECISION_CEILING",
                    str(curriculum.biochem_physics_precision_ceiling_warmup),
                )
            )
            if teacher_phys_ceiling > 0.0:
                teacher_phys_min_lv = -math.log(max(teacher_phys_ceiling, 1e-6))
                with torch.no_grad():
                    teacher_weighter.per_task_min_log_var[:6].fill_(teacher_phys_min_lv)

            teacher.mu_ratio_max = _biochem_teacher_mu_ratio_max(bio_cfg)

            teacher.train()
            teacher_optimizer.zero_grad()
            epoch_l_tot, epoch_l_bio = 0.0, 0.0
            epoch_l_kine, epoch_l_back = 0.0, 0.0
            epoch_mu_si_w, epoch_mu_log_w = 0.0, 0.0
            epoch_mu_log_wall_w, epoch_mu_log_high_w = 0.0, 0.0
            epoch_mu_batches = 0
            epoch_dbg_mu1_sum = 0.0
            epoch_dbg_mu2_sum = 0.0
            epoch_dbg_mu_learned_sum = 0.0
            epoch_dbg_flow_imb_sum = 0.0
            epoch_dbg_q_rel_err_sum = 0.0
            epoch_dbg_flow_trivial_sum = 0.0
            epoch_dbg_flux_n = 0
            epoch_dbg_mu_log_anchor_sum = 0.0
            epoch_dbg_gate_all_sum = 0.0
            epoch_dbg_gate_wall_sum = 0.0
            epoch_dbg_gate_clot_sum = 0.0
            epoch_dbg_gate_n = 0
            epoch_dbg_n = 0
            n_batches = 0
            for batch_idx, data in enumerate(teacher_loader):
                data = data.to(device, non_blocking=nb_xfer)
                # Supervised teacher loss does not optimize node features. Building the full
                # autograd graph through ``x`` (bio_encoder input + sparse ops) is unnecessary
                # and on some anchors produced non-finite adjoint grads on 4 GiB GPUs while
                # the scalar loss stayed finite. Opt in with BIOCHEM_NODE_FEAT_REQUIRES_GRAD=1.
                if _biochem_env_truthy("BIOCHEM_NODE_FEAT_REQUIRES_GRAD", default=False):
                    data.x.requires_grad_(True)
                else:
                    data.x.requires_grad_(False)
                loss, metrics = compute_biochem_loss(
                    teacher,
                    data,
                    kernels,
                    teacher_weighter,
                    device,
                    bio_cfg,
                    epoch=epoch,
                    total_epochs=max_epochs,
                    curriculum=teacher_curriculum,
                    pseudo_target_trajectory=None,
                    pseudo_loss_weight=0.0,
                    debug_batch=(epoch, batch_idx) if _biochem_should_log_batch(epoch, batch_idx) else None,
                    train_cfg=train_cfg,
                )
                if not bool(torch.isfinite(loss.detach()).all().item()):
                    src = _biochem_src_for_log(data)
                    print(
                        f"   ⚠️ Teacher ep={epoch} batch={batch_idx}: non-finite loss; skipping backward/step "
                        f"(src={src!r}). Try lower BIOCHEM_TEACHER_LR / BIOCHEM_TEACHER_CLIP_NORM, shorter "
                        f"BIOCHEM_TBPTT_MAX_WINDOW, or BIOCHEM_DETACH_MACRO_STATE=1 (VRAM trade-off).",
                        flush=True,
                    )
                    teacher_optimizer.zero_grad()
                    continue
                epoch_l_tot += float(loss.item())
                current_l_bio = float(metrics.get("L_Data_Bio", 0.0))
                epoch_l_bio += current_l_bio
                epoch_l_kine += float(metrics.get("L_Data_Kine", 0.0))
                epoch_l_back += float(metrics.get("L_Backprop", float(loss.detach().item())))
                if metrics.get("Has_Anchor_Supervision", 0.0) > 0.5:
                    epoch_mu_si_w += float(metrics.get("W_MuSI_aux_eff", 0.0)) * float(metrics.get("L_MuSI_aux", 0.0))
                    epoch_mu_log_w += float(metrics.get("W_MuLog_aux_eff", 0.0)) * float(metrics.get("L_MuLog_aux", 0.0))
                    epoch_mu_log_wall_w += float(metrics.get("W_MuLog_wall_eff", 0.0)) * float(metrics.get("L_MuLog_wall", 0.0))
                    epoch_mu_log_high_w += float(metrics.get("W_MuLog_high_eff", 0.0)) * float(metrics.get("L_MuLog_high", 0.0))
                    epoch_mu_batches += 1
                dbg_mu1 = float(metrics.get("DBG_mu1_mean", float("nan")))
                dbg_mu2 = float(metrics.get("DBG_mu2_mean", float("nan")))
                dbg_ml = float(metrics.get("DBG_mu_learned_mean", float("nan")))
                dbg_fi = float(metrics.get("DBG_Q_inlet_outlet_imbalance", float("nan")))
                dbg_qre = float(metrics.get("DBG_Q_inlet_rel_err", float("nan")))
                dbg_triv = float(metrics.get("DBG_flow_trivial_score", float("nan")))
                dbg_lm = float(metrics.get("DBG_mu_log_mae_final_anchor", float("nan")))
                dbg_gate_all = float(metrics.get("DBG_gate_mean_all", float("nan")))
                dbg_gate_wall = float(metrics.get("DBG_gate_mean_wall", float("nan")))
                dbg_gate_clot = float(metrics.get("DBG_gate_mean_clot", float("nan")))
                if (
                    math.isfinite(dbg_mu1)
                    and math.isfinite(dbg_mu2)
                    and math.isfinite(dbg_ml)
                    and math.isfinite(dbg_fi)
                    and math.isfinite(dbg_lm)
                ):
                    epoch_dbg_mu1_sum += dbg_mu1
                    epoch_dbg_mu2_sum += dbg_mu2
                    epoch_dbg_mu_learned_sum += dbg_ml
                    epoch_dbg_flow_imb_sum += dbg_fi
                    epoch_dbg_mu_log_anchor_sum += dbg_lm
                    epoch_dbg_n += 1
                if math.isfinite(dbg_qre) and math.isfinite(dbg_triv):
                    epoch_dbg_q_rel_err_sum += dbg_qre
                    epoch_dbg_flow_trivial_sum += dbg_triv
                    epoch_dbg_flux_n += 1
                if math.isfinite(dbg_gate_all) and math.isfinite(dbg_gate_wall) and math.isfinite(dbg_gate_clot):
                    epoch_dbg_gate_all_sum += dbg_gate_all
                    epoch_dbg_gate_wall_sum += dbg_gate_wall
                    epoch_dbg_gate_clot_sum += dbg_gate_clot
                    epoch_dbg_gate_n += 1
                n_batches += 1
                if ema_l_bio is None:
                    ema_l_bio = current_l_bio
                else:
                    ema_l_bio = ema_beta * ema_l_bio + (1.0 - ema_beta) * current_l_bio
                (loss / accumulation_steps).backward()
                if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(teacher_loader)):
                    if not _biochem_split_opt_grads_finite(teacher_optimizer):
                        src = _biochem_src_for_log(data)
                        detail = _biochem_nonfinite_grad_detail(teacher, teacher_optimizer)
                        tail = f" nonfinite_params={detail!r}" if detail else ""
                        print(
                            f"   ⚠️ Teacher ep={epoch} batch={batch_idx} src={src!r}: non-finite gradients "
                            f"after backward.{tail} "
                            "Try BIOCHEM_ADJOINT_RK4_SUBSTEPS=24–32, BIOCHEM_ODEINT_USE_ADJOINT=0 (dense backward, "
                            "more VRAM), or BIOCHEM_TBPTT_MAX_WINDOW=4. Skipping optimizer step (weights unchanged).",
                            flush=True,
                        )
                        teacher_optimizer.zero_grad()
                    else:
                        raw_gn = _biochem_bio_grad_global_l2(teacher_optimizer)
                        max_raw = float(os.environ.get("BIOCHEM_TEACHER_MAX_RAW_GRAD_L2", "5000.0"))
                        if not math.isfinite(raw_gn) or raw_gn > max_raw:
                            print(
                                f"   ⚠️ Teacher ep={epoch} batch={batch_idx}: bio grad L2={raw_gn:.3e} "
                                f"exceeds cap {max_raw:g} or non-finite; skipping optimizer step. "
                                "Lower BIOCHEM_TEACHER_LR or raise cap if this fires too often.",
                                flush=True,
                            )
                            teacher_optimizer.zero_grad()
                        else:
                            teacher_optimizer.clip_and_step(
                                physics_clip=clip_teacher_phys,
                                bio_clip=clip_teacher,
                            )

            if n_batches == 0:
                print(
                    "   🛑 Teacher epoch had zero finite-loss batches (all skipped or empty loader); "
                    "aborting teacher stage.",
                    flush=True,
                )
                break

            avg_tot = epoch_l_tot / max(1, n_batches)
            avg_bio = epoch_l_bio / max(1, n_batches)
            avg_kine = epoch_l_kine / max(1, n_batches)
            avg_back = epoch_l_back / max(1, n_batches)
            ema_bio_str = f"{ema_l_bio:.3e}" if ema_l_bio is not None else "n/a"
            skip_teacher_val = _biochem_env_truthy("BIOCHEM_TEACHER_SKIP_VAL", default=False)
            if skip_teacher_val:
                run_teacher_val = False
            else:
                run_teacher_val = (epoch % teacher_val_every == 0) or (epoch == max_epochs - 1)
            val_stats = None
            if run_teacher_val:
                val_t0 = time.perf_counter()
                v_stride = os.environ.get("BIOCHEM_VAL_TIME_STRIDE", "1")
                tbptt_cap = os.environ.get("BIOCHEM_TBPTT_MAX_WINDOW", "?")
                detach_m = os.environ.get("BIOCHEM_DETACH_MACRO_STATE", "0")
                w_mu = os.environ.get("BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT", "0")
                w_mul = os.environ.get("BIOCHEM_MU_LOG_ANCHOR_WEIGHT", "0")
                print(
                    f"   ⏳ Teacher Ep {epoch:02d} val (stride={v_stride}, TBPTT_cap={tbptt_cap}, "
                    f"DETACH_MACRO={detach_m}, W_MuSI={w_mu}, W_MuLog={w_mul})..."
                )
                val_bundle = _compute_anchor_mu_metrics(
                    teacher, teacher_val_loader, kernels, bio_cfg, device, non_blocking=nb_xfer
                )
                dbg_mu1_e = (epoch_dbg_mu1_sum / float(max(1, epoch_dbg_n)))
                dbg_mu2_e = (epoch_dbg_mu2_sum / float(max(1, epoch_dbg_n)))
                dbg_mlearn_e = (epoch_dbg_mu_learned_sum / float(max(1, epoch_dbg_n)))
                dbg_flow_imb_e = (epoch_dbg_flow_imb_sum / float(max(1, epoch_dbg_n)))
                dbg_q_rel_e = (epoch_dbg_q_rel_err_sum / float(max(1, epoch_dbg_n)))
                dbg_triv_e = (epoch_dbg_flow_trivial_sum / float(max(1, epoch_dbg_n)))
                dbg_mu_log_e = (epoch_dbg_mu_log_anchor_sum / float(max(1, epoch_dbg_n)))
                dbg_gate_all_e = (
                    epoch_dbg_gate_all_sum / float(max(1, epoch_dbg_gate_n))
                    if epoch_dbg_gate_n > 0
                    else float("nan")
                )
                dbg_gate_wall_e = (
                    epoch_dbg_gate_wall_sum / float(max(1, epoch_dbg_gate_n))
                    if epoch_dbg_gate_n > 0
                    else float("nan")
                )
                dbg_gate_clot_e = (
                    epoch_dbg_gate_clot_sum / float(max(1, epoch_dbg_gate_n))
                    if epoch_dbg_gate_n > 0
                    else float("nan")
                )
                val_stats = _log_mu_validation_report(
                    stage="teacher",
                    epoch=epoch,
                    per_graph=list(val_bundle.get("per_graph") or []),
                    extra_lines=[
                        f"   Train Ep {epoch:02d}: L_tot={avg_tot:.3e} L_kine={avg_kine:.3e} "
                        f"L_bio(avg)={avg_bio:.3e} L_bio(ema)={ema_bio_str} | Pceil={teacher_phys_ceiling:.2g}",
                        (
                            f"   μ weighted terms (anchor batches={epoch_mu_batches}): "
                            f"W·L_MuSI={epoch_mu_si_w / max(1, epoch_mu_batches):.3e}, "
                            f"W·L_MuLog={epoch_mu_log_w / max(1, epoch_mu_batches):.3e}, "
                            f"W·L_MuLogWall={epoch_mu_log_wall_w / max(1, epoch_mu_batches):.3e}, "
                            f"W·L_MuLogHigh={epoch_mu_log_high_w / max(1, epoch_mu_batches):.3e}"
                        ),
                        (
                            f"   μ debug (batch-mean): mu1={dbg_mu1_e:.3e} mu2={dbg_mu2_e:.3e} "
                            f"learned={dbg_mlearn_e:.3e} "
                            f"final_anchor_logMAE={dbg_mu_log_e:.3e} Q_imb={dbg_flow_imb_e:.3e} "
                            f"Q_rel_err={dbg_q_rel_e:.3e} flow_trivial={dbg_triv_e:.3f} "
                            f"gate_all={dbg_gate_all_e:.3e} gate_wall={dbg_gate_wall_e:.3e} "
                            f"gate_clot={dbg_gate_clot_e:.3e}"
                        ),
                    ],
                )
                val_mu_score = -float(val_stats["mu_log_mae"])
                print(
                    f"   ✅ Teacher Ep {epoch:02d} validation complete "
                    f"({time.perf_counter() - val_t0:.2f}s) | mu_score={val_mu_score:.4f} "
                    f"(best running {best_mu_score:.4f})."
                )
                cand_all = float(val_stats["mu_log_mae"])
                cand_high = float(val_stats.get("mu_log_mae_high_mu", float("inf")))
                take_ckpt = False
                ckpt_reason = ""
                if best_state is None:
                    take_ckpt = True
                    ckpt_reason = "first checkpoint"
                elif pareto_ckpt:
                    all_improves = cand_all <= (best_mu_log_all - pareto_all_gain)
                    high_improves = cand_high <= (best_mu_log_high - pareto_high_gain)
                    all_ok = cand_all <= (best_mu_log_all + pareto_all_tol)
                    high_ok = cand_high <= (best_mu_log_high + pareto_high_tol)
                    if all_improves and high_ok:
                        take_ckpt = True
                        ckpt_reason = "Pareto(all improved, high within tol)"
                    elif high_improves and all_ok:
                        take_ckpt = True
                        ckpt_reason = "Pareto(high improved, all within tol)"
                elif val_mu_score > best_mu_score:
                    take_ckpt = True
                    ckpt_reason = "mu_score improved"
                if take_ckpt:
                    best_mu_score = val_mu_score
                    best_mu_log_all = cand_all
                    best_mu_log_high = cand_high
                    best_state = copy.deepcopy(teacher.state_dict())
                    best_epoch = int(epoch)
                    print(
                        f"   💾 Teacher checkpoint updated ({ckpt_reason}) | "
                        f"all={best_mu_log_all:.4f}, high={best_mu_log_high:.4f}"
                    )
                if (
                    early_stop_allowed
                    and target_mu_log_mae is not None
                    and float(val_stats["mu_log_mae"]) <= target_mu_log_mae
                ):
                    print(
                        f"   ✅ Teacher reached BIOCHEM_TEACHER_TARGET_MU_LOG_MAE "
                        f"({float(val_stats['mu_log_mae']):.4f} <= {target_mu_log_mae:.4f}); stopping early."
                    )
                    break
            else:
                dbg_mu1_e = (epoch_dbg_mu1_sum / float(max(1, epoch_dbg_n)))
                dbg_mu2_e = (epoch_dbg_mu2_sum / float(max(1, epoch_dbg_n)))
                dbg_mlearn_e = (epoch_dbg_mu_learned_sum / float(max(1, epoch_dbg_n)))
                dbg_flow_imb_e = (epoch_dbg_flow_imb_sum / float(max(1, epoch_dbg_n)))
                dbg_q_rel_e = (epoch_dbg_q_rel_err_sum / float(max(1, epoch_dbg_flux_n)))
                dbg_triv_e = (epoch_dbg_flow_trivial_sum / float(max(1, epoch_dbg_flux_n)))
                dbg_mu_log_e = (epoch_dbg_mu_log_anchor_sum / float(max(1, epoch_dbg_n)))
                dbg_gate_all_e = (
                    epoch_dbg_gate_all_sum / float(max(1, epoch_dbg_gate_n))
                    if epoch_dbg_gate_n > 0
                    else float("nan")
                )
                dbg_gate_wall_e = (
                    epoch_dbg_gate_wall_sum / float(max(1, epoch_dbg_gate_n))
                    if epoch_dbg_gate_n > 0
                    else float("nan")
                )
                dbg_gate_clot_e = (
                    epoch_dbg_gate_clot_sum / float(max(1, epoch_dbg_gate_n))
                    if epoch_dbg_gate_n > 0
                    else float("nan")
                )
                print(
                    f"   Teacher Ep {epoch:02d} | Train [L_tot: {avg_tot:.3e}, L_Back: {avg_back:.3e}, "
                    f"L_Kine(Avg): {avg_kine:.3e}, L_Bio (EMA β={ema_beta:.2f}): {ema_bio_str}, "
                    f"L_Bio (Avg): {avg_bio:.3e}] | "
                    f"μdbg[mu1={dbg_mu1_e:.2e},mu2={dbg_mu2_e:.2e},learned={dbg_mlearn_e:.2e},"
                    f"logMAE_f={dbg_mu_log_e:.2e},Q_imb={dbg_flow_imb_e:.2e},"
                    f"Q_rel={dbg_q_rel_e:.2e},trivial={dbg_triv_e:.3f},"
                    f"gate_all={dbg_gate_all_e:.2e},gate_wall={dbg_gate_wall_e:.2e},"
                    f"gate_clot={dbg_gate_clot_e:.2e}] | "
                    f"Val skipped (BIOCHEM_TEACHER_VAL_EVERY={teacher_val_every})"
                )

            teacher_metrics_row: Dict[str, Any] = {
                "stage": "teacher",
                "epoch": int(epoch),
                "train_L_tot": float(avg_tot),
                "train_L_bio_avg": float(avg_bio),
                "train_L_kine_avg": float(avg_kine),
                "train_L_back_avg": float(avg_back),
                "ema_l_bio": float(ema_l_bio) if ema_l_bio is not None else None,
                "teacher_val_ran": bool(run_teacher_val),
                "teacher_val_every": int(teacher_val_every),
                "teacher_best_mu_score_running": float(best_mu_score),
                "dbg_mu1_mean": (epoch_dbg_mu1_sum / float(max(1, epoch_dbg_n))),
                "dbg_mu2_mean": (epoch_dbg_mu2_sum / float(max(1, epoch_dbg_n))),
                "dbg_mu_learned_mean": (epoch_dbg_mu_learned_sum / float(max(1, epoch_dbg_n))),
                "dbg_flux_imbalance_mean": (epoch_dbg_flow_imb_sum / float(max(1, epoch_dbg_n))),
                "dbg_Q_inlet_rel_err_mean": (
                    epoch_dbg_q_rel_err_sum / float(max(1, epoch_dbg_flux_n))
                    if epoch_dbg_flux_n > 0
                    else None
                ),
                "dbg_flow_trivial_score_mean": (
                    epoch_dbg_flow_trivial_sum / float(max(1, epoch_dbg_flux_n))
                    if epoch_dbg_flux_n > 0
                    else None
                ),
                "dbg_final_anchor_mu_log_mae_mean": (epoch_dbg_mu_log_anchor_sum / float(max(1, epoch_dbg_n))),
                "dbg_gate_mean_all": (
                    epoch_dbg_gate_all_sum / float(max(1, epoch_dbg_gate_n))
                    if epoch_dbg_gate_n > 0
                    else None
                ),
                "dbg_gate_mean_wall": (
                    epoch_dbg_gate_wall_sum / float(max(1, epoch_dbg_gate_n))
                    if epoch_dbg_gate_n > 0
                    else None
                ),
                "dbg_gate_mean_clot": (
                    epoch_dbg_gate_clot_sum / float(max(1, epoch_dbg_gate_n))
                    if epoch_dbg_gate_n > 0
                    else None
                ),
                "val_mu_mae_si": float(val_stats["mu_mae_si"]) if val_stats is not None else None,
                "val_mu_rmse_si": float(val_stats["mu_rmse_si"]) if val_stats is not None else None,
                "val_mu_log_mae": float(val_stats["mu_log_mae"]) if val_stats is not None else None,
                "val_mu_log_mae_wall": (
                    float(val_stats["mu_log_mae_wall"]) if val_stats is not None else None
                ),
                "val_mu_log_mae_high_mu": (
                    float(val_stats["mu_log_mae_high_mu"]) if val_stats is not None else None
                ),
                "val_mu_pearson": float(val_stats["mu_pearson"]) if val_stats is not None else None,
                "val_mu_r2": float(val_stats["mu_r2"]) if val_stats is not None else None,
                "val_mu_score": (-float(val_stats["mu_log_mae"])) if val_stats is not None else None,
                "val_time_stride": int(os.environ.get("BIOCHEM_VAL_TIME_STRIDE", "1") or "1"),
                "tbptt_max_window": int(os.environ.get("BIOCHEM_TBPTT_MAX_WINDOW", "0") or "0"),
            }
            if val_stats is not None and run_teacher_val:
                for g in val_bundle.get("per_graph") or []:
                    row = dict(teacher_metrics_row)
                    row["anchor"] = g.get("anchor")
                    row["per_anchor"] = True
                    sub = g.get("subsets") or {}
                    for sk, sm in sub.items():
                        if isinstance(sm, dict):
                            for mk, mv in sm.items():
                                row[f"val_{sk}_{mk}"] = mv
                    _biochem_append_jsonl(row)
            else:
                _biochem_append_jsonl(teacher_metrics_row)
    finally:
        if _prev_tbptt_anchor_rand is None:
            os.environ.pop("BIOCHEM_TBPTT_ANCHOR_RANDOM_START", None)
        else:
            os.environ["BIOCHEM_TBPTT_ANCHOR_RANDOM_START"] = _prev_tbptt_anchor_rand

    if best_state is not None:
        teacher.load_state_dict(best_state, strict=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    if _biochem_env_truthy("BIOCHEM_TEACHER_SKIP_VAL", default=False) and best_state is None:
        print("✅ Teacher frozen. Validation skipped (BIOCHEM_TEACHER_SKIP_VAL=1); using last training weights.")
    else:
        print(f"✅ Teacher frozen. Best anchor validation mu_score (-log_MAE): {best_mu_score:.4f}")
    return teacher, float(best_mu_score), int(best_epoch)


def build_synthetic_pseudo_labels(teacher, synthetic_dataset, bio_cfg, device):
    """Run frozen teacher on synthetic graphs and cache pseudo trajectories."""
    pseudo = {}
    if teacher is None or len(synthetic_dataset) == 0:
        return pseudo

    teacher.eval()
    dl_kw = _biochem_dataloader_kw(device)
    nb_xfer = _biochem_non_blocking_transfer(device, dl_kw)
    synth_loader = DataLoader(synthetic_dataset, batch_size=1, shuffle=False, **dl_kw)
    print(f"🧾 Building pseudo-label bank for {len(synthetic_dataset)} synthetic graphs...")
    with torch.no_grad():
        for data in synth_loader:
            src = _biochem_data_source_key(data)
            if src is None:
                continue
            data = data.to(device, non_blocking=nb_xfer)
            eval_times = to_t_nd(bio_cfg.resolve_biochem_times(data, device), bio_cfg.t_final)
            pred = teacher(data, eval_times)
            if isinstance(pred, tuple):
                pred = pred[0]
            pseudo[src] = pred.detach().cpu()
    print(f"✅ Pseudo-label bank ready: {len(pseudo)} synthetic trajectories.")
    return pseudo


def train_biochem_corrector(epochs=60, lr=1e-3):
    _apply_pycharm_biochem_optimal_defaults()
    _preset = (os.environ.get("BIOCHEM_PRESET") or "").strip().lower()
    if _preset in _OVERNIGHT_STEP2_PRESET_ALIASES:
        print(
            "🌙 BIOCHEM_PRESET overnight-step2: long AE/ODE/teacher, val every "
            f"{os.environ.get('BIOCHEM_TEACHER_VAL_EVERY', '?')} ep, TBPTT cap "
            f"{os.environ.get('BIOCHEM_TBPTT_MAX_WINDOW', '?')}, μ SI + COMSOL temporal "
            f"(w_pt={os.environ.get('BIOCHEM_COMSOL_TEMPORAL_WEIGHT', '?')}) in data-only backprop. "
            "BIOCHEM_STOCK_DEFAULTS=1 skips presets. Lower TBPTT cap if OOM."
        )
    else:
        print(
            "⚡ Default preset: fast-iterate (shorter pretrain/teacher, val every "
            f"{os.environ.get('BIOCHEM_TEACHER_VAL_EVERY', '?')} ep) + step-2 data-only "
            f"(μ SI w={os.environ.get('BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT', '?')}, TBPTT cap="
            f"{os.environ.get('BIOCHEM_TBPTT_MAX_WINDOW', '?')}, curriculum="
            f"{os.environ.get('BIOCHEM_TBPTT_WINDOW_CURRICULUM', '?')}). "
            "BIOCHEM_STOCK_DEFAULTS=1 for long-run stock env; BIOCHEM_DEBUG=1 for batch logs. "
            "Lower TBPTT cap or μ weight if OOM. Overnight same tier: BIOCHEM_PRESET=overnight_step2."
        )
    _preset_raw = (os.environ.get("BIOCHEM_PRESET") or "").strip()
    _win_ps = ""
    if sys.platform == "win32":
        _win_ps = (
            " | PowerShell: $env:BIOCHEM_PRESET='overnight_step2' "
            "(cmd.exe: set BIOCHEM_PRESET=overnight_step2 && python …)"
        )
    print(
        "   Env snapshot: BIOCHEM_PRESET="
        f"{_preset_raw!r} | BIOCHEM_TEACHER_EPOCHS={os.environ.get('BIOCHEM_TEACHER_EPOCHS', '(unset)')} | "
        f"BIOCHEM_DEBUG={os.environ.get('BIOCHEM_DEBUG', '(unset)')} | "
        f"BIOCHEM_DATA_ONLY_PHYS_TEMP={os.environ.get('BIOCHEM_DATA_ONLY_PHYS_TEMP', '(unset)')}"
        f"{_win_ps}"
    )
    _iso = (os.environ.get("BIOCHEM_LOSS_ISOLATE") or "").strip()
    if _iso:
        print(
            f"ℹ️ BIOCHEM_LOSS_ISOLATE={_iso!r} — backward uses only that term. "
            "Unset it to use BIOCHEM_LOSS_DATA_ONLY (default) or full multitask loss."
        )
    epochs = max(1, int(os.environ.get("BIOCHEM_EPOCHS", str(epochs))))
    lr = float(os.environ.get("BIOCHEM_LR", str(lr)))
    device = resolve_training_device()
    print(f"Device: {device}")
    if device.type == "cpu":
        print(
            "CPU: biochem ODE uses 32 RK4 substeps per macro subsegment by default "
            "(set BIOCHEM_ADJOINT_RK4_SUBSTEPS higher for fidelity, lower for speed)."
        )
    else:
        configure_cuda_for_training(device)
    _apply_biochem_matmul_precision()
    _apply_biochem_sparse_invariant_mode()

    phys_cfg = PhysicsConfig(phase="biochem")
    bio_cfg = BiochemConfig(phase="biochem")
    curriculum = CurriculumConfig()
    curriculum.biochem_warmup_epochs = int(
        os.environ.get("BIOCHEM_WARMUP_EPOCHS", str(curriculum.biochem_warmup_epochs))
    )
    curriculum.biochem_physics_precision_ramp_epochs = int(
        os.environ.get(
            "BIOCHEM_PHYSICS_PRECISION_RAMP_EPOCHS",
            str(curriculum.biochem_physics_precision_ramp_epochs),
        )
    )
    core_kernels = PhysicsKernels(phys_cfg=phys_cfg)
    kernels = BiochemPhysicsKernels(biochem_cfg=bio_cfg, core_physics_kernels=core_kernels)

    # PASS PHYS_CFG TO MODEL
    bio_enc_prior = max(0, int(os.environ.get("BIOCHEM_BIO_ENCODER_PRIOR_DIM", "2")))
    latent_dim = max(8, int(os.environ.get("BIOCHEM_LATENT_DIM", "256")))
    model = GNODE_Phase3(
        phys_cfg=phys_cfg,
        in_channels=12,
        spatial_channels=15,
        latent_dim=latent_dim,
        max_inner_iters=10,
        bio_encoder_prior_dim=bio_enc_prior,
        mu_ratio_max=bio_cfg.mu_ratio_max,
        mat_crit=bio_cfg.viscosity_mat_crit,
        fi_crit=bio_cfg.viscosity_fi_crit,
        temp_mat=bio_cfg.viscosity_gnode_temp_mat,
        temp_fi=bio_cfg.viscosity_gnode_temp_fi,
    ).to(device)
    if bio_enc_prior > 0:
        print(
            f"🧭 bio_encoder kinematics prior: {bio_enc_prior} extra channel(s) "
            f"(BIOCHEM_BIO_ENCODER_PRIOR_DIM)."
        )
    print(f"🧠 Biochem latent_dim={latent_dim} (BIOCHEM_LATENT_DIM).")

    # 1. Backbone weights: stage-A kinematics_best.pth is required; optional biochem_best_bio or full latest resume.
    root = get_project_root()
    model_dir = stage_b_dir()
    biochem_resume_path = resolve_checkpoint("b", "biochem_best_bio.pth")
    kinematics_path = resolve_checkpoint("a", "kinematics_best.pth")
    if not kinematics_path.is_file():
        raise FileNotFoundError(
            f"Required kinematics checkpoint missing: {kinematics_path}. "
            "Train stage A (kinematics predictor) until kinematics_best.pth exists, or fix checkpoint paths."
        )
    latest_ckpt_path = resolve_checkpoint("b", "biochem_latest_checkpoint.pth")
    resume_enabled = (os.environ.get("BIOCHEM_RESUME", "0").strip().lower() in ("1", "true", "yes", "on"))
    init_from_best = (os.environ.get("BIOCHEM_INIT_FROM_BEST", "0").strip().lower() in ("1", "true", "yes", "on"))
    will_resume_from_latest = bool(resume_enabled and latest_ckpt_path.exists())
    load_biochem_best_weights = (init_from_best or resume_enabled) and biochem_resume_path.exists()

    loaded_biochem_best_backbone = False
    if will_resume_from_latest:
        print(
            "ℹ️ BIOCHEM_RESUME: will restore full training state from biochem_latest_checkpoint.pth "
            "(skipping backbone load here)."
        )
    elif load_biochem_best_weights:
        resume_state = torch.load(biochem_resume_path, map_location=device, weights_only=True)
        compatible_resume, skipped_resume = _filter_compatible_state_dict(resume_state, model.state_dict())
        model.load_state_dict(compatible_resume, strict=False)
        loaded_biochem_best_backbone = True
        print(f"🔁 Initialized Biochem weights from {biochem_resume_path.name}")
        if skipped_resume:
            print(
                f"⚠️ Skipped {len(skipped_resume)} incompatible/missing tensor(s) from {biochem_resume_path.name}."
            )
    else:
        print(f"🔁 Initializing Biochem kinematics backbone from {kinematics_path.name}")
        state_dict = torch.load(kinematics_path, map_location=device, weights_only=True)

        mapped_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('encoder.'):
                mapped_state_dict[key.replace('encoder.', 'kin_encoder.')] = value
            elif key.startswith('core.'):
                mapped_state_dict[key.replace('core.', 'kin_processor.')] = value
            elif key.startswith('kinematics_decoder.'):
                mapped_state_dict[key] = value
            # Extract the frozen mu_encoder
            elif key.startswith('mu_encoder.'):
                mapped_state_dict[key] = value

        # --- Dynamic channel expansion surgery (Kinematics -> Biochem) ---
        if 'kin_encoder.0.weight' in mapped_state_dict:
            kinematics_weight = mapped_state_dict['kin_encoder.0.weight']
            model_weight = model.kin_encoder[0].weight
            if kinematics_weight.shape[1] != model_weight.shape[1]:
                print(f"🔧 Adapting Kinematics encoder weights ({kinematics_weight.shape[1]} -> {model_weight.shape[1]})...")
                new_weight = remap_stage_a_encoder_to_corrector(kinematics_weight, model_weight)
                mapped_state_dict['kin_encoder.0.weight'] = new_weight
        # ------------------------------------------------------------

        compatible_backbone, skipped_backbone = _filter_compatible_state_dict(
            mapped_state_dict,
            model.state_dict(),
        )
        model.load_state_dict(compatible_backbone, strict=False)
        print("✅ Successfully loaded Kinematics kinematic weights into Biochem backbone.")
        if skipped_backbone:
            print(
                f"⚠️ Skipped {len(skipped_backbone)} incompatible/missing kinematics tensor(s) "
                "(expected when hidden widths differ)."
            )

    if will_resume_from_latest:
        pass  # biochem + kinematics come from biochem_latest_checkpoint.pth
    elif loaded_biochem_best_backbone:
        print("⏭️ Skipping biochem prior initialization because Biochem best checkpoint was loaded.")
    else:
        initialize_biochem_priors(model)
    loss_weighter = make_biochem_dynamic_loss_weighter(curriculum, device)

    print("💉 Injecting LoRA into kinematic modules (SpectralLinear layers)...")
    inject_biochem_kinematic_lora(model)

    if _biochem_debug_enabled():
        cap = _biochem_debug_batches_cap()
        lp = _biochem_debug_log_path()
        try:
            lp.parent.mkdir(parents=True, exist_ok=True)
            lp.write_text("", encoding="utf-8")
        except OSError:
            pass
        _biochem_dbg_line(
            f"[BIOCHEM_DEBUG] file={lp} (truncated each run); first {cap} batch(es)/epoch "
            f"(set BIOCHEM_DEBUG_BATCHES). TBPTT: BIOCHEM_TBPTT_MAX_WINDOW caps y time slice; "
            f"BIOCHEM_TBPTT_WINDOW_CURRICULUM=1 restores legacy epoch ramp."
        )

    dataset = load_dataset()
    if len(dataset) == 0:
        return

    # Keep loading lazy: split by file path metadata instead of materializing all graphs.
    all_files = list(dataset.file_list)
    anchors, physics = [], []
    print("🔎 Indexing Biochem files by anchor flag (lazy split)...")
    for graph_path in all_files:
        graph = torch.load(graph_path, map_location="cpu", weights_only=False)
        graph = infer_missing_schema(graph, phase_hint="biochem")
        assert_graph_schema(graph, expected_y_schema=(BIO_Y_SCHEMA,))
        ia = getattr(graph, "is_anchor", None)
        if ia is None:
            is_anchor = False
        elif torch.is_tensor(ia):
            is_anchor = bool(ia.any().item())
        else:
            is_anchor = bool(ia)
        if is_anchor:
            anchors.append(graph_path)
        else:
            physics.append(graph_path)
        del graph
    gc.collect()

    random.seed(42)
    random.shuffle(anchors)
    random.shuffle(physics)
    n_anchors_total = len(anchors)
    low_anchor_threshold = max(1, int(os.environ.get("BIOCHEM_LOW_ANCHOR_THRESHOLD", "5")))
    force_low_anchor_mode = (os.environ.get("BIOCHEM_LOW_ANCHOR_MODE", "").strip().lower() in ("1", "true", "yes", "on"))
    low_anchor_mode = force_low_anchor_mode or (0 < n_anchors_total < low_anchor_threshold)
    if low_anchor_mode:
        print(
            f"🧪 Low-anchor mode enabled: anchors={n_anchors_total} (<{low_anchor_threshold}). "
            "Training emphasizes pipeline health/debug over generalization."
        )

    min_trust = int(curriculum.biochem_min_anchors_for_trusted_metrics)
    metrics_trustworthy = n_anchors_total >= min_trust
    if not metrics_trustworthy:
        print(
            f"⚠️ Validation high-μ overlap / WSS are **not** reliable generalization metrics with "
            f"{n_anchors_total} anchor graph(s) (< {min_trust}). Interpret as pipeline health only."
        )

    # Robust split: keep at least one anchor in training whenever anchors exist.
    train_anchors, val_anchors = [], []
    train_physics, val_physics = [], []
    if len(dataset) == 1:
        print("⚠️ Only one graph found. Using it for both Training and Validation.")
        only = [all_files[0]]
        if len(anchors) == 1:
            train_anchors = only[:]
            val_anchors = only[:]
        else:
            train_physics = only[:]
            val_physics = only[:]
        train_data = only
        val_data = only
    else:
        if len(anchors) <= 1:
            # Low-anchor regime: keep anchor in both splits so validation metrics remain meaningful.
            train_anchors = anchors[:]
            val_anchors = anchors[:]
        else:
            split_idx_a = int(0.9 * len(anchors))
            split_idx_a = max(1, min(split_idx_a, len(anchors) - 1))
            train_anchors = anchors[:split_idx_a]
            val_anchors = anchors[split_idx_a:]

        if len(physics) <= 1:
            train_physics = physics[:]
            val_physics = []
        else:
            split_idx_p = int(0.9 * len(physics))
            split_idx_p = max(1, min(split_idx_p, len(physics) - 1))
            train_physics = physics[:split_idx_p]
            val_physics = physics[split_idx_p:]

        train_data = train_anchors + train_physics
        val_data = val_anchors + val_physics

        # Safety fallback if split produced empty validation
        if len(val_data) == 0:
            val_data = train_data

    _ds_root = str(data_root())
    train_dataset = PatientDataset(root=_ds_root, file_list=train_data)
    val_dataset = PatientDataset(root=_ds_root, file_list=val_data)
    train_anchor_dataset = PatientDataset(root=_ds_root, file_list=train_anchors)
    val_anchor_dataset = PatientDataset(
        root=_ds_root,
        file_list=val_anchors if len(val_anchors) > 0 else train_anchors
    )
    train_synth_dataset = PatientDataset(root=_ds_root, file_list=train_physics)

    _print_biochem_anchor_file_split(train_anchors, val_anchors)

    dl_kw = _biochem_dataloader_kw(device)
    nb_xfer = _biochem_non_blocking_transfer(device, dl_kw)
    nw_dl = int(dl_kw.get("num_workers", 0))
    if nw_dl > 0 or bool(dl_kw.get("pin_memory")):
        print(
            f"⚡ DataLoader throughput: num_workers={nw_dl}, pin_memory={dl_kw.get('pin_memory')} "
            "(override with BIOCHEM_DATALOADER_WORKERS / BIOCHEM_PIN_MEMORY)"
        )

    # IMPORTANT:
    # Biochem graphs store trajectories as y: [T, N, 16]. With vanilla PyG batching,
    # x concatenates over nodes while y concatenates over time, which misaligns tensors.
    # Use batch_size=1 and gradient accumulation for stable/equivalent optimization.
    accumulation_steps = max(1, int(os.environ.get("BIOCHEM_ACCUMULATION_STEPS", "1")))

    # Use simple loaders with batch_size=1 to preserve [T, N, 16] integrity.
    train_anchor_count = len(train_anchors) if len(dataset) > 1 else 1
    if train_anchor_count == 0:
        print("⚠️ No anchors in training split; running physics-only updates.")

    # Anchor oversampling for sparse-anchor phases (especially 1-anchor runs).
    # Target a minimum anchor sampling fraction without changing graph contents.
    anchor_set = set(train_anchors)
    train_physics_count = max(0, len(train_data) - len(anchor_set))
    if len(anchor_set) > 0 and train_physics_count > 0:
        if low_anchor_mode:
            target_anchor_fraction = 0.70 if len(anchor_set) == 1 else 0.55
        else:
            target_anchor_fraction = 0.5 if len(anchor_set) == 1 else 0.35
        w_anchor = target_anchor_fraction / max(len(anchor_set), 1)
        w_phys = (1.0 - target_anchor_fraction) / max(train_physics_count, 1)
        sample_weights = [w_anchor if p in anchor_set else w_phys for p in train_data]
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=max(len(train_data), accumulation_steps * 2),
            replacement=True,
        )
        print(
            f"🎯 Weighted sampling enabled (anchor frac target ~{target_anchor_fraction:.2f}; "
            f"anchors={len(anchor_set)}, physics={train_physics_count})."
        )
        loader = DataLoader(
            train_dataset, batch_size=1, shuffle=False, sampler=train_sampler, **dl_kw
        )
    else:
        loader = DataLoader(train_dataset, batch_size=1, shuffle=True, **dl_kw)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, **dl_kw)

    optimizer = setup_biochem_optimization(model, loss_weighter, base_lr=lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer.bio_optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    start_epoch = 0
    best_composite = -1.0e9
    mu_score_ema: Optional[float] = None
    teacher_best_mu_score = 0.0
    latest_ckpt_save = model_dir / "biochem_latest_checkpoint.pth"
    try:
        val_every = max(1, int(os.environ.get("BIOCHEM_VAL_EVERY", "4")))
    except ValueError:
        val_every = 4
    ckpt_every = max(1, int(os.environ.get("BIOCHEM_CKPT_EVERY", "4")))
    resume_ema_state = None
    stop_after_teacher = _biochem_stop_after_teacher()

    if resume_enabled and latest_ckpt_path.exists():
        print(f"🔄 Resuming Biochem from checkpoint: {latest_ckpt_path}")
        ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        opt_ok = _try_load_biochem_split_optimizer(optimizer, ckpt.get("optimizer_state_dict"))
        if opt_ok:
            scheduler_state = ckpt.get("scheduler_state_dict")
            if scheduler_state is not None:
                try:
                    scheduler.load_state_dict(scheduler_state)
                except Exception as exc:
                    print(
                        f"⚠️ Could not load LR scheduler state ({type(exc).__name__}: {exc}); "
                        "scheduler left at default init."
                    )
        try:
            loss_weighter.load_state_dict(ckpt["loss_weighter_state_dict"])
        except Exception as exc:
            print(
                f"⚠️ Could not load loss weighter state ({type(exc).__name__}: {exc}); "
                "using freshly constructed weighter state."
            )
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_composite = float(ckpt.get("best_composite", best_composite))
        mu_score_ema = ckpt.get("mu_score_ema", mu_score_ema)
        if mu_score_ema is not None:
            mu_score_ema = float(mu_score_ema)
        teacher_best_mu_score = float(ckpt.get("teacher_best_mu_score", 0.0))
        pseudo_w = float(ckpt.get("pseudo_w", 0.0))
        resume_ema_state = ckpt.get("ema_model_state_dict")
        if stop_after_teacher:
            pseudo_bank = {}
            print(
                "ℹ️ Skipping pseudo-label bank rebuild "
                "(BIOCHEM_STOP_AFTER_TEACHER=1; corrector not run)."
            )
        else:
            print("🧾 Rebuilding pseudo-label bank from resumed Biochem weights...")
            temp_teacher = copy.deepcopy(model).to(device)
            temp_teacher.eval()
            for p in temp_teacher.parameters():
                p.requires_grad = False
            pseudo_bank = build_synthetic_pseudo_labels(
                teacher=temp_teacher,
                synthetic_dataset=train_synth_dataset,
                bio_cfg=bio_cfg,
                device=device,
            )
            del temp_teacher
            pseudo_cov = len(pseudo_bank) / max(1, len(train_physics))
            if len(train_physics) > 0:
                print(
                    f"🧾 Pseudo-label coverage after resume: {len(pseudo_bank)}/{len(train_physics)} "
                    f"({pseudo_cov:.1%}) synthetic graphs."
                )
        print(f"✅ Biochem resume complete at epoch {start_epoch}.")
    elif resume_enabled:
        print(f"ℹ️ BIOCHEM_RESUME is enabled but no checkpoint found at {latest_ckpt_path}. Continuing fresh.")
    else:
        post_pt = model_dir / "biochem_post_pretrain.pth"
        reused_post_pretrain = False
        if _biochem_env_truthy("BIOCHEM_REUSE_LAST_PRETRAIN", default=False):
            reused_post_pretrain = _try_load_biochem_post_pretrain(model, post_pt, device)
            if reused_post_pretrain:
                print(
                    f"🔁 Reused AE+ODE-RXN warm-start from {post_pt.name} (skipping Phase 3a / 3a.5). "
                    "Unset BIOCHEM_REUSE_LAST_PRETRAIN or delete the file to run pretrain again."
                )
            else:
                print(
                    f"⚠️ BIOCHEM_REUSE_LAST_PRETRAIN set but {post_pt.name} is missing or incompatible; "
                    "running Phase 3a + 3a.5 from scratch."
                )
        save_post_pt: Optional[Path] = None
        if not reused_post_pretrain:
            save_post_pt = post_pt
            if _biochem_env_truthy("BIOCHEM_SKIP_PRETRAIN", default=False) or _biochem_env_truthy(
                "BIOCHEM_SKIP_ODE_RXN_PRETRAIN", default=False
            ):
                save_post_pt = None
            pretrain_autoencoder(
                model,
                loader,
                optimizer,
                device,
                kernels,
                epochs=max(1, int(os.environ.get("BIOCHEM_AE_EPOCHS", "30"))),
                ode_reaction_epochs=max(
                    1,
                    int(
                        os.environ.get(
                            "BIOCHEM_ODE_RXN_EPOCHS",
                            os.environ.get("BIOCHEM_ODE_REACTION_EPOCHS", "25"),
                        )
                    ),
                ),
                post_pretrain_save_path=save_post_pt,
            )
        if device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
        teacher, teacher_best_mu_score, teacher_best_epoch = train_teacher_on_anchors(
            student_model=model,
            train_anchor_dataset=train_anchor_dataset,
            val_anchor_dataset=val_anchor_dataset,
            kernels=kernels,
            bio_cfg=bio_cfg,
            curriculum=curriculum,
            device=device,
            base_lr=lr,
            low_anchor_mode=low_anchor_mode,
        )
        _save_biochem_teacher_best_checkpoint(
            model_dir / BIOCHEM_TEACHER_BEST_CKPT_NAME,
            teacher,
            teacher_best_mu_score=float(teacher_best_mu_score),
            best_epoch=int(teacher_best_epoch),
            run_note=(os.environ.get("BIOCHEM_RUN_NOTE") or "").strip(),
        )
        if stop_after_teacher:
            pseudo_bank = {}
            pseudo_w = 0.0
            print(
                "ℹ️ Skipping synthetic pseudo-label bank "
                "(BIOCHEM_STOP_AFTER_TEACHER=1; corrector not run)."
            )
        else:
            pseudo_bank = build_synthetic_pseudo_labels(
                teacher=teacher,
                synthetic_dataset=train_synth_dataset,
                bio_cfg=bio_cfg,
                device=device,
            )
            pseudo_cov = len(pseudo_bank) / max(1, len(train_physics))
            if len(train_physics) > 0:
                print(
                    f"🧾 Pseudo-label coverage: {len(pseudo_bank)}/{len(train_physics)} "
                    f"({pseudo_cov:.1%}) synthetic graphs."
                )
            pseudo_w_base = float(os.environ.get("BIOCHEM_SYNTH_PSEUDO_WEIGHT", "0.5"))
            min_teacher_score = float(os.environ.get("BIOCHEM_PSEUDO_MIN_TEACHER_MU_SCORE", "-1.0"))
            if teacher_best_mu_score < min_teacher_score:
                pseudo_w = 0.0
                print(
                    f"🧷 Synthetic pseudo-label weight set to 0 "
                    f"(teacher mu_score {teacher_best_mu_score:.4f} < "
                    f"BIOCHEM_PSEUDO_MIN_TEACHER_MU_SCORE={min_teacher_score})."
                )
            else:
                ref_score = float(os.environ.get("BIOCHEM_PSEUDO_TEACHER_REF_MU_SCORE", "-0.25"))
                denom = max(ref_score - min_teacher_score, 1e-6)
                ramp = min(1.0, max(0.0, (teacher_best_mu_score - min_teacher_score) / denom))
                pseudo_w = pseudo_w_base * ramp
                print(
                    f"🧷 Synthetic pseudo-label loss weight: {pseudo_w:.3f} "
                    f"(base={pseudo_w_base:.3f}, teacher_mu_score={teacher_best_mu_score:.4f}, ramp={ramp:.3f})"
                )

        # Teacher optimizes a deep copy; merge back so Phase 3 trains the same weights that produced
        # pseudo trajectories (teacher LoRA stayed frozen — student LoRA tensors load from teacher where keys match).
        if teacher is not None:
            compatible_tm, skipped_tm = _filter_compatible_state_dict(
                teacher.state_dict(), model.state_dict()
            )
            if compatible_tm:
                model.load_state_dict(compatible_tm, strict=False)
            n_tm = len(skipped_tm) if skipped_tm else 0
            merge_note = (
                " for Phase 3 corrector."
                if not stop_after_teacher
                else " (teacher-only run; corrector skipped)."
            )
            print(
                "🔁 Merged teacher weights into student" + merge_note
                + (f" Skipped {n_tm} incompatible/missing key(s)." if n_tm else "")
            )

    if stop_after_teacher:
        print("🛑 BIOCHEM_STOP_AFTER_TEACHER enabled: stopping after teacher stage.")
        return

    train_cfg = BiochemTrainingConfig.from_env()
    physics_clip_norm = float(os.environ.get("BIOCHEM_PHYSICS_CLIP_NORM", "0.1"))
    bio_clip_norm = float(os.environ.get("BIOCHEM_BIO_CLIP_NORM", "1.0"))
    ema_decay = float(os.environ.get("BIOCHEM_EMA_DECAY", "0.999"))
    ema_enabled = (os.environ.get("BIOCHEM_EMA", "1") or "").strip().lower() in ("1", "true", "yes", "on")
    ema_model = None
    if ema_enabled:
        ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))
        if resume_ema_state is not None:
            try:
                ema_model.load_state_dict(resume_ema_state, strict=False)
                print(f"🫧 EMA model restored from checkpoint (decay={ema_decay:.4f}).")
            except Exception as exc:
                print(f"⚠️ Could not restore EMA state ({exc}); starting EMA from current model.")
        else:
            print(f"🫧 EMA model enabled for validation/checkpoints (decay={ema_decay:.4f}).")

    mu_score_ema_beta = float(os.environ.get("BIOCHEM_VAL_MU_SCORE_EMA", "0.25"))
    ckpt_pearson_w = float(os.environ.get("BIOCHEM_CKPT_WSS_PEARSON_WEIGHT", "0.02"))

    cfg_paths = VesselConfig(phase="biochem")
    diary = TrainingDiary("biochem")
    diary.log_run_start(
        device=str(device),
        re_target=float(phys_cfg.re_target),
        graph_dir=str(cfg_paths.graph_output_dir),
        n_graphs_total=len(dataset),
        n_files_indexed=len(all_files),
        n_anchors_total=int(n_anchors_total),
        n_train_graphs=len(train_data),
        n_val_graphs=len(val_data),
        n_train_anchors=len(train_anchors),
        n_val_anchors=len(val_anchors),
        n_train_physics=len(train_physics),
        n_val_physics=len(val_physics),
        low_anchor_mode=bool(low_anchor_mode),
        metrics_trustworthy=bool(metrics_trustworthy),
        accumulation_steps=int(accumulation_steps),
        epochs=int(epochs),
        lr=float(lr),
        start_epoch=int(start_epoch),
        best_composite_checkpoint=float(best_composite),
        mu_score_ema_checkpoint=float(mu_score_ema) if mu_score_ema is not None else None,
        teacher_best_mu_score=float(teacher_best_mu_score),
        pseudo_w=float(pseudo_w),
        pseudo_bank_size=len(pseudo_bank),
        biochem_warmup_epochs=int(curriculum.biochem_warmup_epochs),
        mu_score_ema_beta=float(mu_score_ema_beta),
        ckpt_wss_pearson_weight=float(ckpt_pearson_w),
        train_cfg=asdict(train_cfg),
        ema_enabled=bool(ema_enabled),
        ema_decay=float(ema_decay),
        resume_enabled=bool(resume_enabled),
        resumed_latest_checkpoint=bool(resume_enabled and latest_ckpt_path.exists()),
        ckpt_every=int(ckpt_every),
        env_biochem_kinematics=env_snapshot("BIOCHEM_", "KINEMATICS_"),
        run_dir=str(diary.run_dir) if diary.run_dir is not None else None,
        diary_main_path=str(diary.path) if diary.path is not None else None,
    )
    if diary.run_dir is not None:
        try:
            (diary.run_dir / "biochem_training_config.json").write_text(
                json.dumps(
                    {
                        "train_cfg": asdict(train_cfg),
                        "ema_enabled": bool(ema_enabled),
                        "ema_decay": float(ema_decay),
                        "env": env_snapshot("BIOCHEM_", "KINEMATICS_"),
                    },
                    indent=2,
                    sort_keys=True,
                    default=str,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass

    run_end_emitted = False
    last_epoch_completed: Optional[int] = None

    def _emit_biochem_run_end(interrupted: bool = False) -> None:
        nonlocal run_end_emitted
        if run_end_emitted or not diary.enabled:
            return
        run_end_emitted = True
        if interrupted:
            print("\n⚠️ Training interrupted; appending training diary run_end (JSONL report).")
        diary.log_run_end(
            best_composite=float(best_composite),
            teacher_best_mu_score=float(teacher_best_mu_score),
            pseudo_w=float(pseudo_w),
            mu_score_ema=float(mu_score_ema) if mu_score_ema is not None else None,
            diary_path=str(diary.path) if diary.path else None,
            biochem_best_bio=str(model_dir / "biochem_best_bio.pth"),
            biochem_latest_checkpoint=str(latest_ckpt_save),
            interrupted=bool(interrupted),
            last_epoch_completed=last_epoch_completed,
        )

    atexit.register(lambda: _emit_biochem_run_end(True))

    print("\n🚀 --- Starting Phase 3: Segregated Bio-Fluid Coupling ---")

    if start_epoch >= epochs:
        extra = max(1, int(os.environ.get("BIOCHEM_RESUME_EXTRA_EPOCHS", "30")))
        new_total = max(epochs, start_epoch + extra)
        print(
            f"⚠️ Resume start_epoch={start_epoch} is ≥ configured BIOCHEM_EPOCHS={epochs}; "
            f"Phase 3 would run 0 iterations. Extending total epochs to {new_total} "
            f"(raise BIOCHEM_EPOCHS explicitly or set BIOCHEM_RESUME_EXTRA_EPOCHS, default +{extra})."
        )
        epochs = new_total

    watchdog_sec = float(os.environ.get("BIOCHEM_BATCH_WATCHDOG_SEC", "300"))
    default_lora_unlock_epoch = max(int(curriculum.biochem_warmup_epochs) + 4, max(1, epochs // 2))
    lora_unlock_epoch = int(
        os.environ.get("BIOCHEM_LORA_UNLOCK_EPOCH", str(default_lora_unlock_epoch))
    )
    lora_unlock_epoch = max(1, min(lora_unlock_epoch, max(1, epochs - 1)))
    print(f"🔐 LoRA unlock schedule: frozen until epoch {lora_unlock_epoch}, then enabled for co-adaptation.")

    for epoch in range(start_epoch, epochs):
        last_epoch_completed = epoch
        wu = curriculum.biochem_warmup_epochs

        ease = curriculum.biochem_curriculum_easing
        if epoch < wu:
            # --- STAGE A: THE PREDICTOR ---
            current_mu_ratio = 1.0  # Force strictly neutral rheology
            span = max(float(wu - 1), 1.0)
            t_w = _ease01(epoch / span, ease)
            current_T_scale = curriculum.biochem_t_scale_warmup_initial - t_w * (
                curriculum.biochem_t_scale_warmup_initial - curriculum.biochem_t_scale_warmup_final
            )

            # Keep macro LoRA frozen during early training; let micro head learn first.
            if epoch == 0:
                print("🔒 Kine phase (Predictor): Freezing LoRA layers; training micro residual head first.")
            for _name, param in model.named_parameters():
                if "lora" in _name.lower():
                    param.requires_grad = False
        else:
            # --- STAGE B: THE CORRECTOR ---
            coupled_denom = max(1, epochs - wu - 1)
            progress = _ease01((epoch - wu) / float(coupled_denom), ease)
            current_mu_ratio = bio_cfg.mu_ratio_init + progress * (
                bio_cfg.mu_ratio_max - bio_cfg.mu_ratio_init
            )
            current_T_scale = curriculum.biochem_t_scale_coupled_initial - progress * (
                curriculum.biochem_t_scale_coupled_initial - curriculum.biochem_t_scale_coupled_final
            )

            # Macro LoRA unfreezes on its own schedule (typically later than warmup).
            if epoch == lora_unlock_epoch:
                print("🔥 Macro LoRA unlock: enabling kinematic co-adaptation / rheological feedback.")
            lo_ra_on = epoch >= lora_unlock_epoch
            for _name, param in model.named_parameters():
                if "lora" in _name.lower():
                    param.requires_grad = bool(lo_ra_on)

        # Push updates to the network and kernels
        model.mu_ratio_max = current_mu_ratio

        # Unify the curriculum temperature
        model.T_scale = current_T_scale
        kernels.kinetics.T_scale = current_T_scale

        # FIX: Capitalized 'T' here as well
        huber_delta_epoch = _scheduled_biochem_huber_delta(bio_cfg, epoch)
        print(
            f"\n⏳ Epoch {epoch:02d} | mu_ratio: {current_mu_ratio:.1f}x | "
            f"T_scale: {current_T_scale:.2f} | huber_delta: {huber_delta_epoch:.4f} | "
            f"res_sparse_w: {_scheduled_residual_sparse_lambda(epoch, epochs):.3f}"
        )

        if curriculum.biochem_weighter_freeze_during_warmup:
            phys_start = wu + int(curriculum.biochem_weighter_physics_grace_epochs)
            if epoch < wu:
                loss_weighter.log_vars.requires_grad_(False)
            elif epoch < phys_start:
                loss_weighter.log_vars.requires_grad_(False)
                loss_weighter.log_vars[6:].requires_grad_(True)
                if epoch == wu:
                    print(
                        "⚖️  Biochem warmup done: unfreezing **data** Kendall log_vars "
                        f"(indices 6–7); physics log_vars frozen until epoch {phys_start}."
                    )
            else:
                loss_weighter.log_vars.requires_grad_(True)
                if epoch == phys_start:
                    print("⚖️  Unfreezing **physics** Kendall log_vars after grace period.")
        else:
            loss_weighter.log_vars.requires_grad_(True)

        current_phys_ceiling = _apply_dynamic_physics_precision_ceiling(loss_weighter, curriculum, epoch)
        model.train()
        total_loss_epoch = 0.0

        # --- NEW Gradient Trackers ---
        total_grad_norm_epoch = 0.0
        grad_clip_count = 0
        optimizer_steps = 0

        optimizer.zero_grad()

        # Epoch-level TF schedule (matches compute_biochem_loss; per-batch TF_eff may be 0 without truth nodes).
        if epoch < wu:
            teacher_forcing_ratio = 1.0
        else:
            decay_progress = (epoch - wu) / float(curriculum.biochem_teacher_force_decay_epochs)
            decay_progress = _ease01(decay_progress, curriculum.biochem_curriculum_easing)
            teacher_forcing_ratio = max(0.0, 1.0 - decay_progress)

        # EMA-smoothed progress metrics for less noisy tqdm feedback.
        ema_metrics = None
        ema_alpha = 0.05
        anchor_supervised_batches = 0
        pseudo_supervised_batches = 0
        no_grad_skipped_batches = 0
        total_batches = 0
        ode_zero_batches = 0
        l_musi_anchor_sum = 0.0
        l_musi_weighted_sum = 0.0
        l_musi_anchor_batch_count = 0
        l_mulog_anchor_sum = 0.0
        l_mulog_weighted_sum = 0.0
        w_mu_log_ep = max(float(os.environ.get("BIOCHEM_MU_LOG_ANCHOR_WEIGHT", "0.0")), 0.0)

        pbar = tqdm(loader, desc=f"Biochem Ep {epoch:02d}")
        for batch_idx, data in enumerate(pbar):
            total_batches += 1
            batch_t0 = time.perf_counter()
            data = data.to(device, non_blocking=nb_xfer)
            data.x.requires_grad_(True)
            data_src = _biochem_data_source_key(data)
            pseudo_target = pseudo_bank.get(data_src) if (data_src is not None and data_src in pseudo_bank) else None

            dbg = (epoch, batch_idx) if _biochem_should_log_batch(epoch, batch_idx) else None
            loss, metrics = compute_biochem_loss(
                model,
                data,
                kernels,
                loss_weighter,
                device,
                bio_cfg,
                epoch=epoch,
                total_epochs=epochs,
                curriculum=curriculum,
                debug_batch=dbg,
                pseudo_target_trajectory=pseudo_target,
                pseudo_loss_weight=pseudo_w,
                train_cfg=train_cfg,
            )
            if metrics.get("Has_Anchor_Supervision", 0.0) > 0.5:
                anchor_supervised_batches += 1
                l_musi_anchor_sum += float(metrics["L_MuSI_aux"])
                w_mu_eff = float(metrics.get("W_MuSI_aux_eff") or 0.0)
                l_musi_weighted_sum += w_mu_eff * float(metrics["L_MuSI_aux"])
                l_musi_anchor_batch_count += 1
                if w_mu_log_ep > 0.0:
                    l_mulog_anchor_sum += float(metrics["L_MuLog_aux"])
                    w_mul_eff = float(metrics.get("W_MuLog_aux_eff") or 0.0)
                    l_mulog_weighted_sum += w_mul_eff * float(metrics["L_MuLog_aux"])
            if metrics.get("Has_Pseudo_Supervision", 0.0) > 0.5:
                pseudo_supervised_batches += 1
            if int(metrics.get("ODE_Evals", 0)) == 0 and float(metrics.get("PDE_Steps", 0.0)) > 0.5:
                ode_zero_batches += 1
            loss = loss / accumulation_steps

            if torch.isnan(loss):
                print(f"\n⚠️ NaN detected in loss at epoch {epoch}! Skipping micro-batch.")
                continue

            if not loss.requires_grad:
                src = getattr(data, "_biochem_path", "<unknown>")
                no_grad_skipped_batches += 1
                _biochem_dbg_line(
                    "⚠️ Phase3 loss has no grad_fn before backward(); skipping micro-batch. "
                    f"epoch={epoch} batch={batch_idx} src={src} "
                    f"TF_eff={metrics.get('TF_eff')} "
                    f"Has_Anchor_Supervision={metrics.get('Has_Anchor_Supervision')} "
                    f"Has_Pseudo_Supervision={metrics.get('Has_Pseudo_Supervision')} "
                    f"L_ADR_F={metrics.get('L_ADR_F'):.3e} "
                    f"L_W_Phy={metrics.get('L_W_Phy'):.3e} "
                    f"L_Data_Kine={metrics.get('L_Data_Kine'):.3e} "
                    f"L_Data_Bio={metrics.get('L_Data_Bio'):.3e}"
                )
                continue
            loss.backward()
            if _biochem_should_log_batch(epoch, batch_idx) and _biochem_env_truthy(
                "BIOCHEM_DEBUG_LOG_GRAD_L2", default=False
            ):
                sq = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        g = p.grad.detach().data
                        sq += float((g * g).sum().item())
                _biochem_dbg_line(
                    f"[BIOCHEM_DEBUG] epoch={epoch} batch={batch_idx} grad_L2={math.sqrt(sq):.4e} (micro-batch)"
                )

            if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(loader)):
                physics_grad_norm, bio_grad_norm = optimizer.clip_and_step(
                    physics_clip=physics_clip_norm,
                    bio_clip=bio_clip_norm,
                )
                grad_norm = max(physics_grad_norm, bio_grad_norm)
                total_grad_norm_epoch += float(grad_norm)
                if physics_grad_norm > physics_clip_norm or bio_grad_norm > bio_clip_norm:
                    grad_clip_count += 1
                optimizer_steps += 1
                if ema_model is not None:
                    ema_model.update_parameters(model)

            batch_dt = time.perf_counter() - batch_t0
            if batch_dt > watchdog_sec:
                _biochem_dbg_line(
                    f"⏱️ [Watchdog] slow batch epoch={epoch} batch={batch_idx} "
                    f"dt={batch_dt:.2f}s ODE_evals={int(metrics.get('ODE_Evals', 0))}"
                )

            current_l_tot = loss.item() * accumulation_steps
            total_loss_epoch += current_l_tot

            if ema_metrics is None:
                ema_metrics = {
                    "L_tot": current_l_tot,
                    "L_Data_Kine": metrics["L_Data_Kine"],
                    "L_Data_Bio": metrics["L_Data_Bio"],
                    "L_ADR_F": metrics['L_ADR_F'],
                    "L_W_Bio": metrics['L_W_Bio'],
                    "L_W_Phy": metrics['L_W_Phy'],
                }
                if metrics.get("Has_Anchor_Supervision", 0.0) > 0.5:
                    ema_metrics["L_MuSI_aux"] = float(metrics["L_MuSI_aux"])
                    if w_mu_log_ep > 0.0:
                        ema_metrics["L_MuLog_aux"] = float(metrics["L_MuLog_aux"])
            else:
                ema_metrics["L_tot"] = (1 - ema_alpha) * ema_metrics["L_tot"] + ema_alpha * current_l_tot
                ema_metrics["L_Data_Kine"] = (1 - ema_alpha) * ema_metrics["L_Data_Kine"] + ema_alpha * metrics["L_Data_Kine"]
                ema_metrics["L_Data_Bio"] = (1 - ema_alpha) * ema_metrics["L_Data_Bio"] + ema_alpha * metrics["L_Data_Bio"]
                ema_metrics["L_ADR_F"] = (1 - ema_alpha) * ema_metrics["L_ADR_F"] + ema_alpha * metrics['L_ADR_F']
                ema_metrics["L_W_Bio"] = (1 - ema_alpha) * ema_metrics["L_W_Bio"] + ema_alpha * metrics['L_W_Bio']
                ema_metrics["L_W_Phy"] = (1 - ema_alpha) * ema_metrics["L_W_Phy"] + ema_alpha * metrics['L_W_Phy']
                if metrics.get("Has_Anchor_Supervision", 0.0) > 0.5:
                    v_musi = float(metrics["L_MuSI_aux"])
                    if "L_MuSI_aux" not in ema_metrics:
                        ema_metrics["L_MuSI_aux"] = v_musi
                    else:
                        ema_metrics["L_MuSI_aux"] = (1 - ema_alpha) * ema_metrics["L_MuSI_aux"] + ema_alpha * v_musi
                    if w_mu_log_ep > 0.0:
                        v_mul = float(metrics["L_MuLog_aux"])
                        if "L_MuLog_aux" not in ema_metrics:
                            ema_metrics["L_MuLog_aux"] = v_mul
                        else:
                            ema_metrics["L_MuLog_aux"] = (1 - ema_alpha) * ema_metrics["L_MuLog_aux"] + ema_alpha * v_mul

            pbar_post = {
                "L_tot": f"{ema_metrics['L_tot']:.2e}",
                "L_Kine": f"{ema_metrics['L_Data_Kine']:.2e}",
                "L_Bio": f"{ema_metrics['L_Data_Bio']:.2e}",
                "L_ADR_F": f"{ema_metrics['L_ADR_F']:.2e}",
                "L_W_Bio": f"{ema_metrics['L_W_Bio']:.2e}",
                "L_W_Phy": f"{ema_metrics['L_W_Phy']:.2e}",
                "TF_eff": f"{metrics['TF_eff']:.2f}",
                "ODE": f"{int(metrics.get('ODE_Evals', 0))}",
                "Pceil": f"{current_phys_ceiling:.0f}",
                "t_batch": f"{batch_dt:.2f}s",
                "A_sup": f"{anchor_supervised_batches}/{total_batches}",
                "P_sup": f"{pseudo_supervised_batches}/{total_batches}",
            }
            if "L_MuSI_aux" in ema_metrics:
                pbar_post["L_MuSI"] = f"{ema_metrics['L_MuSI_aux']:.2e}"
            if w_mu_log_ep > 0.0 and "L_MuLog_aux" in ema_metrics:
                pbar_post["L_MuLog"] = f"{ema_metrics['L_MuLog_aux']:.2e}"
            pbar.set_postfix(pbar_post)

        scheduler.step()
        if total_batches > 0:
            frac = anchor_supervised_batches / float(total_batches)
            print(
                f"📌 Anchor-supervised batches: {anchor_supervised_batches}/{total_batches} "
                f"({frac:.1%})"
            )
            pfrac = pseudo_supervised_batches / float(total_batches)
            print(
                f"🧾 Pseudo-supervised batches: {pseudo_supervised_batches}/{total_batches} "
                f"({pfrac:.1%})"
            )
            if l_musi_anchor_batch_count > 0:
                mean_lm = l_musi_anchor_sum / float(l_musi_anchor_batch_count)
                mean_wlm = l_musi_weighted_sum / float(l_musi_anchor_batch_count)
                print(
                    f"📐 μ SI anchor (anchor-supervised batches only): mean L_MuSI_aux={mean_lm:.4e}, "
                    f"mean (W_eff·L_MuSI)={mean_wlm:.4e} over {l_musi_anchor_batch_count} batch(es)"
                )
            if w_mu_log_ep > 0.0 and l_musi_anchor_batch_count > 0:
                mean_ll = l_mulog_anchor_sum / float(l_musi_anchor_batch_count)
                mean_wll = l_mulog_weighted_sum / float(l_musi_anchor_batch_count)
                print(
                    f"📐 μ log anchor (val-aligned |log pred−gt| in SI): mean L_MuLog_aux={mean_ll:.4e}, "
                    f"mean (W_eff·L_MuLog)={mean_wll:.4e} over {l_musi_anchor_batch_count} batch(es)"
                )
            if low_anchor_mode:
                print(
                    f"🩺 Low-anchor health: anchor_frac={frac:.1%}, pseudo_frac={pfrac:.1%}, "
                    f"pseudo_graph_cov={len(pseudo_bank)}/{max(1, len(train_physics))} ({pseudo_cov:.1%})"
                )
            if no_grad_skipped_batches > 0:
                print(
                    f"🛟 No-grad micro-batches skipped: {no_grad_skipped_batches}/{total_batches} "
                    f"({(no_grad_skipped_batches / float(total_batches)):.1%})"
                )
            if total_batches > 0 and ode_zero_batches / float(total_batches) > 0.5:
                print(
                    f"⚠️ Sanity: ODE_Evals==0 on {ode_zero_batches}/{total_batches} batches "
                    "despite PDE_Steps>0 — check TBPTT windows / synthetic batches."
                )

        train_loss_mean = float(total_loss_epoch) / max(float(total_batches), 1.0)

        epoch_l_musi_log: Dict[str, Any] = {}
        if l_musi_anchor_batch_count > 0:
            epoch_l_musi_log["mean_L_MuSI_aux"] = float(
                l_musi_anchor_sum / float(l_musi_anchor_batch_count)
            )
            epoch_l_musi_log["mean_W_eff_L_MuSI_aux"] = float(
                l_musi_weighted_sum / float(l_musi_anchor_batch_count)
            )
            epoch_l_musi_log["n_batches_L_MuSI_anchor"] = int(l_musi_anchor_batch_count)
            if w_mu_log_ep > 0.0:
                epoch_l_musi_log["mean_L_MuLog_aux"] = float(
                    l_mulog_anchor_sum / float(l_musi_anchor_batch_count)
                )
                epoch_l_musi_log["mean_W_eff_L_MuLog_aux"] = float(
                    l_mulog_weighted_sum / float(l_musi_anchor_batch_count)
                )

        _iso_ep = (os.environ.get("BIOCHEM_LOSS_ISOLATE") or "").strip()
        if (
            total_batches > 0
            and ema_metrics is not None
            and _biochem_env_truthy("BIOCHEM_LOSS_DATA_ONLY")
            and not _iso_ep
        ):
            print(
                f"📊 Ep {epoch:02d} data-only (EMA α={ema_alpha:g}): "
                f"L_tot≈{ema_metrics['L_tot']:.3e} L_Data_Kine≈{ema_metrics['L_Data_Kine']:.3e} "
                f"L_Data_Bio≈{ema_metrics['L_Data_Bio']:.3e} | mean L_tot/μbatch {train_loss_mean:.3e}"
            )

        # Validation & Metrics
        val_log: Optional[Dict[str, Any]] = None
        run_phase3_val = (epoch % val_every == 0) or (epoch == epochs - 1)
        if run_phase3_val:
            val_model = ema_model if ema_model is not None else model
            val_model.eval()
            val_pearson_total, val_fibrin_total = 0.0, 0.0

            # --- NEW Accumulators ---
            val_cont_total, val_wall_slip_total, val_kine_l2_total = 0.0, 0.0, 0.0
            val_rp_mae_total, val_t_mae_total = 0.0, 0.0
            val_mu_mae_total, val_mu_rmse_total, val_mu_log_mae_total = 0.0, 0.0, 0.0
            val_mu_pearson_total, val_mu_r2_total = 0.0, 0.0

            n_val_anchor = 0
            wss_reason_hist: Dict[str, int] = {}
            val_mu_per_graph: List[Dict[str, Any]] = []

            with torch.no_grad():
                safe_vars = loss_weighter.clamped_log_vars()
                weights = torch.exp(-safe_vars)

                print(
                    f"⚖️ Learned Weights -> ADR_F: {weights[0]:.2f} | ADR_S: {weights[1]:.2f} | "
                    f"W_Bio: {weights[2]:.2f} | W_Phys: {weights[3]:.2f} | Bio_IO: {weights[4]:.2f} | "
                    f"NS_mom: {weights[5]:.2f} | Data_Kine: {weights[6]:.2f} | Data_Bio: {weights[7]:.2f}"
                )
                learned_w = {
                    "w_ADR_F": float(weights[0].item()),
                    "w_ADR_S": float(weights[1].item()),
                    "w_W_Bio": float(weights[2].item()),
                    "w_W_Phys": float(weights[3].item()),
                    "w_Bio_IO": float(weights[4].item()),
                    "w_NS_mom": float(weights[5].item()),
                    "w_Data_Kine": float(weights[6].item()),
                    "w_Data_Bio": float(weights[7].item()),
                }

                for v_data in val_loader:
                    v_data = v_data.to(device, non_blocking=nb_xfer)
                    val_eval_times = _validation_eval_times(v_data, bio_cfg, device)
                    v_pred = val_model(v_data, val_eval_times)
                    if isinstance(v_pred, tuple):
                        v_pred = v_pred[0]

                    (
                        wss_p,
                        f,
                        wss_diag,
                        cont_err,
                        wall_slip,
                        kine_l2,
                        rp_err,
                        t_err,
                        mu_mae_si,
                        mu_rmse_si,
                        mu_log_mae,
                        mu_pearson,
                        mu_r2,
                    ) = calculate_validation_metrics(v_pred[-1], v_data, kernels, device)
                    val_pearson_total += wss_p
                    val_fibrin_total += f
                    val_cont_total += cont_err  # Computed on all graphs
                    val_wall_slip_total += wall_slip  # Computed on all graphs
                    val_mu_mae_total += mu_mae_si
                    val_mu_rmse_total += mu_rmse_si
                    val_mu_log_mae_total += mu_log_mae
                    val_mu_pearson_total += mu_pearson
                    val_mu_r2_total += mu_r2
                    rk = str(wss_diag.get("wss_pearson_reason", "unset"))
                    wss_reason_hist[rk] = wss_reason_hist.get(rk, 0) + 1
                    mu_dbg = wss_diag.get("mu_debug")
                    if isinstance(mu_dbg, dict) and int(
                        (mu_dbg.get("subsets") or {}).get("all", {}).get("n", 0)
                    ) >= 1:
                        val_mu_per_graph.append(mu_dbg)
                    is_anc = _graph_has_anchor_nodes(v_data)
                    if is_anc:
                        val_kine_l2_total += kine_l2  # Computed on anchors
                        val_rp_mae_total += rp_err    # Computed on anchors
                        val_t_mae_total += t_err      # Computed on anchors
                        n_val_anchor += 1

            model.train()
            n_val = max(len(val_loader), 1)
            n_val_anchor_safe = max(n_val_anchor, 1)
            avg_pearson = val_pearson_total / n_val
            avg_fibrin = val_fibrin_total / n_val
            avg_mu_mae = val_mu_mae_total / n_val
            avg_mu_rmse = val_mu_rmse_total / n_val
            avg_mu_log_mae = val_mu_log_mae_total / n_val
            avg_mu_pearson = val_mu_pearson_total / n_val
            avg_mu_r2 = val_mu_r2_total / n_val
            legacy_mu_log_mae = float(avg_mu_log_mae)

            trust_tag = "" if metrics_trustworthy else " [HEALTH-ONLY: few anchors]"
            wss_top = sorted(wss_reason_hist.items(), key=lambda kv: -kv[1])[0][0] if wss_reason_hist else "n/a"

            v_stride = os.environ.get("BIOCHEM_VAL_TIME_STRIDE", "1")
            print(
                f"📊 [Validation]{trust_tag} (val_stride={v_stride}, graphs={n_val}, "
                f"anchors={n_val_anchor})"
            )
            mu_agg = _log_mu_validation_report(
                stage="corrector",
                epoch=epoch,
                per_graph=val_mu_per_graph,
                extra_lines=[
                    f"   Legacy mean over all val graphs: logMAE={legacy_mu_log_mae:.4f} "
                    f"μPearson={avg_mu_pearson:.4f} (includes non-anchor graphs with empty μ)",
                    f"   Patent WSS Pearson: {avg_pearson:.4f} | Max Fibrin (SI): {avg_fibrin:.2e}",
                    f"   Fluid Physics: Continuity Err={val_cont_total/n_val:.2e} | "
                    f"Wall Slip Err={val_wall_slip_total/n_val:.2e} | "
                    f"Kinematic Rel_L2={val_kine_l2_total/n_val_anchor_safe:.4f}",
                    f"   Cascade MAE: RP={val_rp_mae_total/n_val_anchor_safe:.4f} | "
                    f"T={val_t_mae_total/n_val_anchor_safe:.4f}",
                    f"   WSS Pearson: top reason '{wss_top}' (hist={dict(wss_reason_hist)}).",
                ],
            )
            avg_mu_log_mae = float(mu_agg.get("mu_log_mae", legacy_mu_log_mae))
            avg_mu_mae = float(mu_agg.get("mu_mae_si", avg_mu_mae))
            avg_mu_rmse = float(mu_agg.get("mu_rmse_si", avg_mu_rmse))
            avg_mu_pearson = float(mu_agg.get("mu_pearson", avg_mu_pearson))
            avg_mu_r2 = float(mu_agg.get("mu_r2", avg_mu_r2))
            mu_score = -float(avg_mu_log_mae)

            if mu_score_ema is None:
                mu_score_ema_local = mu_score
            else:
                mu_score_ema_local = (1.0 - mu_score_ema_beta) * mu_score_ema + mu_score_ema_beta * mu_score
            composite_score = float(mu_score_ema_local) + ckpt_pearson_w * float(avg_pearson)
            mu_score_ema = mu_score_ema_local

            if composite_score > best_composite:
                best_composite = composite_score
                model_dir.mkdir(parents=True, exist_ok=True)
                best_state_to_save = (
                    ema_model.module.state_dict()
                    if ema_model is not None and hasattr(ema_model, "module")
                    else model.state_dict()
                )
                torch.save(best_state_to_save, model_dir / "biochem_best_bio.pth")
                print(
                    f"⭐ Saved best checkpoint (composite={composite_score:.4f} = mu_score_ema + "
                    f"{ckpt_pearson_w:g}*wss_pearson; mu_score_ema={mu_score_ema_local:.4f})"
                )

            val_log = {
                "avg_mu_mae_si": avg_mu_mae,
                "avg_mu_rmse_si": avg_mu_rmse,
                "avg_mu_log_mae": avg_mu_log_mae,
                "avg_mu_log_mae_wall": float(mu_agg.get("mu_log_mae_wall", float("nan"))),
                "avg_mu_log_mae_high_mu": float(mu_agg.get("mu_log_mae_high_mu", float("nan"))),
                "avg_mu_pearson": avg_mu_pearson,
                "avg_mu_r2": avg_mu_r2,
                "avg_pearson": avg_pearson,
                "avg_fibrin": avg_fibrin,
                "mu_score_ema": mu_score_ema_local,
                "composite_score": composite_score,
                "wss_reason_hist": dict(wss_reason_hist),
                "metrics_trustworthy": metrics_trustworthy,
                # --- NEW Validation Logging ---
                "avg_continuity": float(val_cont_total / n_val),
                "avg_wall_slip": float(val_wall_slip_total / n_val),
                "avg_kine_rel_l2": float(val_kine_l2_total / n_val_anchor_safe),
                "avg_rp_mae": float(val_rp_mae_total / n_val_anchor_safe),
                "avg_t_mae": float(val_t_mae_total / n_val_anchor_safe),
            }
            diary.log_validation(epoch, val_log, **learned_w)

        diary.log_epoch_end(
            epoch,
            train_loss_mean=float(train_loss_mean),
            lr=float(optimizer.param_groups[0]["lr"]),
            mu_ratio=float(current_mu_ratio),
            T_scale=float(current_T_scale),
            teacher_forcing_ratio=float(teacher_forcing_ratio),
            best_composite_so_far=float(best_composite),
            mu_score_ema=float(mu_score_ema) if mu_score_ema is not None else None,
            total_batches=int(total_batches),
            anchor_supervised_batches=int(anchor_supervised_batches),
            pseudo_supervised_batches=int(pseudo_supervised_batches),
            low_anchor_mode=bool(low_anchor_mode),
            pseudo_w=float(pseudo_w),
            teacher_best_mu_score=float(teacher_best_mu_score),
            # --- NEW Gradient Logging ---
            avg_grad_norm=float(total_grad_norm_epoch / max(1, optimizer_steps)),
            grad_clip_rate=float(grad_clip_count / max(1, optimizer_steps)),
            **epoch_l_musi_log,
        )
        log_row: Dict[str, Any] = {
            "epoch": int(epoch),
            "train_loss_mean_microbatch": float(train_loss_mean),
            "mu_ratio": float(current_mu_ratio),
            "T_scale": float(current_T_scale),
            "teacher_forcing_epoch": float(teacher_forcing_ratio),
            "metrics_trustworthy": metrics_trustworthy,
        }
        log_row.update(epoch_l_musi_log)
        if val_log is not None:
            log_row.update(val_log)
        _biochem_append_jsonl(log_row)

        should_save_ckpt = ((epoch + 1) % ckpt_every == 0) or (epoch == epochs - 1)
        if should_save_ckpt:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "ema_model_state_dict": ema_model.state_dict() if ema_model is not None else None,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss_weighter_state_dict": loss_weighter.state_dict(),
                "best_composite": best_composite,
                "mu_score_ema": mu_score_ema,
                "teacher_best_mu_score": teacher_best_mu_score,
                "pseudo_w": pseudo_w,
                "train_cfg": asdict(train_cfg),
                "ema_decay": ema_decay,
            }
            torch.save(checkpoint, latest_ckpt_save)
            print(f"💾 Saved Biochem checkpoint -> {latest_ckpt_save.name} (every {ckpt_every} epoch(s))")

    _emit_biochem_run_end(interrupted=False)


def _parse_args():
    p = argparse.ArgumentParser(description="Biochem GNODE corrector training.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Resume from biochem_latest_checkpoint.pth (sets BIOCHEM_RESUME=1).",
    )
    mode.add_argument(
        "--new",
        action="store_true",
        help="Start a new run (sets BIOCHEM_RESUME=0 and BIOCHEM_INIT_FROM_BEST=0).",
    )
    p.add_argument(
        "--skip-pretrain",
        action="store_true",
        help="Skip AE + ODE-RXN (sets BIOCHEM_SKIP_PRETRAIN=1). Use with BIOCHEM_INIT_FROM_BEST or a good checkpoint.",
    )
    p.add_argument(
        "--skip-ae",
        action="store_true",
        help="Skip Phase 3a autoencoder only (sets BIOCHEM_SKIP_AE_PRETRAIN=1).",
    )
    p.add_argument(
        "--skip-ode-rxn",
        action="store_true",
        help="Skip Phase 3a.5 ODE reaction mimic only (sets BIOCHEM_SKIP_ODE_RXN_PRETRAIN=1).",
    )
    return p.parse_args()


def _resolve_train_mode_from_env_or_prompt() -> str:
    """``BIOCHEM_TRAIN_MODE=new|resume`` skips the interactive prompt (sweep scripts)."""
    forced = (os.environ.get("BIOCHEM_TRAIN_MODE") or "").strip().lower()
    if forced in ("new", "start", "fresh", "2"):
        return "new"
    if forced in ("resume", "continue", "1"):
        return "resume"
    return _prompt_train_mode()


def _prompt_train_mode() -> str:
    """Prompt: 1=resume, 2=new from scratch (always runs AE + ODE-RXN unless env skip flags)."""
    while True:
        raw = input(
            "Training mode [1=resume / 2=start new] [1]: "
        ).strip()
        if raw in ("", "1"):
            return "resume"
        if raw == "2":
            return "new"
        print("  Enter 1 or 2.")


if __name__ == "__main__":
    args = _parse_args()
    if args.resume:
        train_mode = "resume"
    elif args.new:
        train_mode = "new"
    else:
        train_mode = _resolve_train_mode_from_env_or_prompt()

    resume_enabled = train_mode == "resume"
    os.environ["BIOCHEM_RESUME"] = "1" if resume_enabled else "0"
    if not resume_enabled:
        os.environ["BIOCHEM_INIT_FROM_BEST"] = "0"
    if "BIOCHEM_REUSE_LAST_PRETRAIN" not in os.environ:
        os.environ["BIOCHEM_REUSE_LAST_PRETRAIN"] = "0"

    if getattr(args, "skip_pretrain", False):
        os.environ["BIOCHEM_SKIP_PRETRAIN"] = "1"
    else:
        if getattr(args, "skip_ae", False):
            os.environ["BIOCHEM_SKIP_AE_PRETRAIN"] = "1"
        if getattr(args, "skip_ode_rxn", False):
            os.environ["BIOCHEM_SKIP_ODE_RXN_PRETRAIN"] = "1"

    if resume_enabled:
        banner = "🔄 Resuming Biochem from latest checkpoint."
    else:
        banner = "🆕 Starting a new Biochem run."
    print(banner)
    try:
        train_biochem_corrector()
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user (KeyboardInterrupt).")
        raise
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n💥 CUDA out of memory during Biochem training: {e}")
        raise
    except Exception as e:
        print(f"\n💥 Unhandled exception during Biochem training: {type(e).__name__}: {e}")
        raise