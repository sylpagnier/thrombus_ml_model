"""
Tier 1 **architecture / loss explorer** — env-driven knobs for systematic experiments.

Use this when iterating toward a tight anchor Rel L2 (e.g. <5% on level-1 vessels). After each run,
compare ``reports/experiments/tier1_<name>_*.json`` and the training diary.

**Explorer execution policy**

- The active ``build_sweep_candidates()`` defines the sweep: **V3** uses **40 AdamW → 20 L-BFGS**
  (``TIER1_USE_LBFGS=1``) to polish NS residuals; the pre-sweep **sanity probe** stays Adam-only
  (1 epoch) for speed.
- Sweeps **always start fresh**: no ``tier1_best_physics.pth`` warm-start and no resume from
  ``tier1_latest_checkpoint.pth`` (per-candidate checkpoint dirs only record *this* run).
- For ad-hoc fast screening, set ``TIER1_USE_LBFGS=0`` in env or swap in a shorter sweep builder.

**Kinematic supervision weighting** (anchor nodes only, COMSOL labels):

- ``uniform`` — baseline (same as before).
- ``sdf_wall`` — upweight nodes with **small |SDF|** (near wall / lumen boundary) where field
  gradients are typically steeper: ``w = 1 + β * exp(-|SDF|/τ)``.
- ``sdf_grad`` — upweight nodes with large **|∇SDF| proxy** (mean edge |ΔSDF|): emphasizes
  geometric constriction/expansion and curvature of the lumen.
- ``shear_true`` — upweight nodes with large **ground-truth shear rate** from label gradients:
  ``w = 1 + alpha * (gamma_dot_true / mean(gamma_dot_true))``.

**Training length**

- ``TIER1_EPOCHS`` — total epochs (default ``60``); use ``~25`` for exploratory sweeps.
- ``TIER1_WARM_UP_EPOCHS``, ``TIER1_ADAM_EPOCHS`` — optional; default warm-up scales with ``adam_epochs`` (usually = ``TIER1_EPOCHS``).

**Architecture**

- ``TIER1_LATENT_DIM``, ``TIER1_DEQ_MAX_ITERS``, ``TIER1_NUM_FOURIER_FREQS`` — GINO-DEQ width/depth.
- ``TIER1_KINEMATICS_MODE`` — ``stream`` or ``direct_uvp``.
- ``TIER1_NS_DERIVATIVE_MODE`` — ``wls`` or ``autograd`` (for PDE derivatives).
- ``TIER1_USE_HARD_BCS`` — ``1``/``true`` to enforce no-slip at walls via SDF × residual (``direct_uvp`` only).
- ``TIER1_GLOBAL_POOL_MODE`` — ``mean`` (legacy global mean pool) or ``attention`` (Perceiver-style bottleneck).
- ``TIER1_NUM_GLOBAL_TOKENS`` — attention bottleneck width when ``GLOBAL_POOL_MODE=attention``.
- ``TIER1_USE_SIREN`` — ``1``/``true`` to use the SIREN INR decoder for ``direct_uvp`` kinematics.
- ``TIER1_USE_EQUIVARIANT`` — reserved flag for future vector-aware convolutions (currently no-op in the model).
- ``TIER1_USE_WIDTH_PRIORS`` — ``1``/``true`` to feed sphere-traced width + WLS flow-direction derivatives (requires graphs with 18 node channels from ``mesh_to_graph``).

We intentionally keep both options for kinematics and PDE derivatives because vessel meshes can
favor different numerical behavior; use sweep artifacts to determine the best pair for your data.

**Data**

- ``TIER1_GEOMETRY_LEVEL`` — if set to ``0`` or ``1``, only load graphs whose ``vessel_*.json``
  has a matching ``level`` field (requires ``data/raw/tier1`` JSON next to meshes).

**Best-practice ideas to try in separate runs**

1. **Weighting**: ``shear_true`` vs ``sdf_wall`` vs ``sdf_grad`` vs ``uniform``.
2. **Capacity**: latent 64 → 96/128; DEQ iters 15 → 20–30 (watch VRAM).
3. **Loss schedule**: existing ``TIER1_DATA_SCALE_*`` / ``TIER1_BC_SCALE_*`` envs (stage A vs B).
4. **Sampling**: ``TIER1_TARGET_ANCHOR_FRACTION``, hard mining (already in trainer).
5. **Robust loss**: future — Huber on kinematic channel (not wired yet).

All settings are logged to the experiment JSON at run end.
"""

from __future__ import annotations

import atexit
import json
import os
import signal
import traceback
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.paths import reports_dir


@dataclass
class T1ExplorerConfig:
    experiment_name: str = "default"
    dataset_tier: str = "tier1"
    kine_weight_mode: str = "uniform"  # uniform | sdf_wall | sdf_grad | shear_true
    sdf_wall_beta: float = 2.0
    sdf_wall_tau: float = 0.12
    sdf_grad_beta: float = 1.0
    shear_true_alpha: float = 1.0
    latent_dim: int = 128
    deq_max_iters: int = 25
    num_fourier_freqs: int = 16
    geometry_level: Optional[int] = None  # filter json "level"
    kinematics_mode: str = "direct_uvp"  # stream | direct_uvp
    ns_derivative_mode: str = "wls"  # wls | autograd
    activation_fn: str = "silu"  # relu | silu | gelu
    fourier_base: float = 1.5
    loss_weight_mode: str = "dynamic"  # dynamic | fixed | grad_norm
    anderson_beta: float = 0.8
    lambda_cont: float = 1.0
    re_curriculum: bool = False
    p_grad_supervision: float = 1.0
    advect_detach: bool = False
    pressure_bc_mode: str = "mean"  # mean | pointwise | mean_var
    momentum_loss_mode: str = "huber"  # huber | mse
    # --- V2 architecture flags (feature-gated; defaults preserve legacy behavior) ---
    use_hard_bcs: bool = False
    global_pool_mode: str = "mean"  # mean | attention
    num_global_tokens: int = 16
    use_equivariant_conv: bool = False  # reserved for future conv path
    use_siren_decoder: bool = False
    use_width_priors: bool = False  # append width channels in encoder (graphs must include them)

    @staticmethod
    def from_env() -> "T1ExplorerConfig":
        def _f(name: str, default: float) -> float:
            raw = os.environ.get(name, "").strip()
            return default if not raw else float(raw)

        gl = os.environ.get("TIER1_GEOMETRY_LEVEL", "").strip()
        geometry_level: Optional[int] = None
        if gl in ("0", "1"):
            geometry_level = int(gl)

        mode = os.environ.get("TIER1_KINE_WEIGHT_MODE", "uniform").strip().lower()
        if mode not in ("uniform", "sdf_wall", "sdf_grad", "shear_true"):
            mode = "uniform"
        kinematics_mode = os.environ.get("TIER1_KINEMATICS_MODE", "direct_uvp").strip().lower()
        if kinematics_mode not in ("stream", "direct_uvp"):
            kinematics_mode = "direct_uvp"
        ns_derivative_mode = os.environ.get("TIER1_NS_DERIVATIVE_MODE", "wls").strip().lower()
        if ns_derivative_mode not in ("wls", "autograd"):
            ns_derivative_mode = "wls"
        activation = os.environ.get("TIER1_ACTIVATION_FN", "silu").strip().lower()
        if activation not in ("relu", "silu", "gelu"):
            activation = "silu"
        loss_weight_mode = os.environ.get("TIER1_LOSS_WEIGHT_MODE", "dynamic").strip().lower()
        if loss_weight_mode not in ("dynamic", "fixed", "grad_norm"):
            loss_weight_mode = "dynamic"
        re_curriculum = os.environ.get("TIER1_RE_CURRICULUM", "0").strip().lower() in ("1", "true", "yes", "on")
        pressure_bc_mode = os.environ.get("TIER1_PRESSURE_BC_MODE", "mean").strip().lower()
        if pressure_bc_mode not in ("mean", "pointwise", "mean_var"):
            pressure_bc_mode = "mean"
        momentum_loss_mode = os.environ.get("TIER1_MOMENTUM_LOSS_MODE", "huber").strip().lower()
        if momentum_loss_mode not in ("huber", "mse"):
            momentum_loss_mode = "huber"
        global_pool_mode = os.environ.get("TIER1_GLOBAL_POOL_MODE", "mean").strip().lower()
        if global_pool_mode not in ("mean", "attention"):
            global_pool_mode = "mean"
        use_hard_bcs = os.environ.get("TIER1_USE_HARD_BCS", "0").strip().lower() in ("1", "true", "yes", "on")
        use_equivariant_conv = os.environ.get("TIER1_USE_EQUIVARIANT", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        use_siren_decoder = os.environ.get("TIER1_USE_SIREN", "0").strip().lower() in ("1", "true", "yes", "on")
        use_width_priors = os.environ.get("TIER1_USE_WIDTH_PRIORS", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        raw_tok = os.environ.get("TIER1_NUM_GLOBAL_TOKENS", "16").strip()
        try:
            num_global_tokens = max(1, int(raw_tok))
        except ValueError:
            num_global_tokens = 16

        return T1ExplorerConfig(
            experiment_name=os.environ.get("TIER1_EXPERIMENT_NAME", "default").strip() or "default",
            dataset_tier=os.environ.get("TIER1_DATASET_TIER", "tier1").strip() or "tier1",
            kine_weight_mode=mode,
            sdf_wall_beta=_f("TIER1_SDF_WALL_BETA", 2.0),
            sdf_wall_tau=_f("TIER1_SDF_WALL_TAU", 0.12),
            sdf_grad_beta=_f("TIER1_SDF_GRAD_BETA", 1.0),
            shear_true_alpha=_f("TIER1_SHEAR_TRUE_ALPHA", 1.0),
            latent_dim=int(os.environ.get("TIER1_LATENT_DIM", "128")),
            deq_max_iters=int(os.environ.get("TIER1_DEQ_MAX_ITERS", "25")),
            num_fourier_freqs=int(os.environ.get("TIER1_NUM_FOURIER_FREQS", "16")),
            geometry_level=geometry_level,
            kinematics_mode=kinematics_mode,
            ns_derivative_mode=ns_derivative_mode,
            activation_fn=activation,
            fourier_base=float(os.environ.get("TIER1_FOURIER_BASE", "1.5")),
            loss_weight_mode=loss_weight_mode,
            anderson_beta=float(os.environ.get("TIER1_ANDERSON_BETA", "0.8")),
            lambda_cont=float(os.environ.get("TIER1_LAMBDA_CONT", "1.0")),
            re_curriculum=re_curriculum,
            p_grad_supervision=float(os.environ.get("TIER1_P_GRAD_SUPERVISION", "1.0")),
            advect_detach=os.environ.get("TIER1_ADVECT_DETACH", "0").strip().lower() in ("1", "true", "yes", "on"),
            pressure_bc_mode=pressure_bc_mode,
            momentum_loss_mode=momentum_loss_mode,
            use_hard_bcs=use_hard_bcs,
            global_pool_mode=global_pool_mode,
            num_global_tokens=num_global_tokens,
            use_equivariant_conv=use_equivariant_conv,
            use_siren_decoder=use_siren_decoder,
            use_width_priors=use_width_priors,
        )

    def to_serializable(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class T1SweepCandidate:
    name: str
    explorer: T1ExplorerConfig
    env_overrides: Dict[str, str]
    epochs: int = 15
    warm_up_epochs: int = 3
    adam_epochs: int = 15


def _safe_env_set(overrides: Dict[str, str]) -> Dict[str, Optional[str]]:
    prev: Dict[str, Optional[str]] = {}
    for k, v in overrides.items():
        prev[k] = os.environ.get(k)
        os.environ[k] = str(v)
    return prev


def _safe_env_restore(prev: Dict[str, Optional[str]]) -> None:
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def write_t1_experiment_artifact(
    explorer: T1ExplorerConfig,
    *,
    best_rel_l2: float,
    best_phys_score: float,
    best_loss: float,
    early_stopped: bool,
    n_graphs: int,
    n_train: int,
    n_val: int,
    graph_dir: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write ``reports/experiments/tier1_<name>_<ts>.json`` for post-run comparison."""
    rep = reports_dir() / "experiments"
    rep.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in explorer.experiment_name)[
        :80
    ]
    path = rep / f"tier1_{safe_name}_{ts}.json"
    payload: Dict[str, Any] = {
        "tier": "tier1",
        "ts_utc": ts,
        "explorer": explorer.to_serializable(),
        "metrics": {
            "best_rel_l2": best_rel_l2,
            "best_phys_score": best_phys_score,
            "best_loss": best_loss,
            "early_stopped": early_stopped,
        },
        "data": {
            "n_graphs": n_graphs,
            "n_train": n_train,
            "n_val": n_val,
            "graph_dir": graph_dir,
        },
        "env_tier1": {k: v for k, v in sorted(os.environ.items()) if k.startswith("TIER1_")},
    }
    if extra:
        payload["extra"] = extra
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"📒 Experiment artifact: {path}")
    print("🧾 History reminder: append this run's key metrics to your Tier1 training history log.")
    return path


def write_t1_sweep_report(payload: Dict[str, Any], sweep_name: str) -> Path:
    rep = reports_dir() / "experiments"
    rep.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in sweep_name)[:80]
    path = rep / f"tier1_sweep_{safe_name}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"📘 Sweep report: {path}")
    print("🧾 History reminder: append this sweep result to your persistent Tier1 training history log.")
    return path


def _safe_name(name: str, max_len: int = 80) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:max_len]


def _write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def build_sweep_candidates() -> List[T1SweepCandidate]:
    """
    V3 architecture sweep: high capacity (``latent_dim=256``) + hard BCs where indicated,
    targeting ``<5%`` relative L2 with **40 AdamW → 20 L-BFGS** per candidate.

    Four-way comparison: legacy baseline vs attention bottleneck vs width priors vs SIREN decoder.
    """
    base_overrides = {
        "TIER1_DISABLE_FIGURES": "1",
        "TIER1_CKPT_EVERY": "5",  # Frequent checkpoints for crash safety
        "TIER1_EARLY_STOP_PATIENCE": "15",
        "TIER1_LOSS_WEIGHT_MODE": "dynamic",
        "TIER1_USE_LBFGS": "1",
        # Never load pretrained weights — fair screening from random init only.
        "TIER1_INIT_FROM_BEST": "0",

        # --- SWEEP ISOLATION ---
        # RESUME is OFF; per-candidate TIER1_CKPT_DIR (set by run_sweep) only stores
        # checkpoints from the current candidate run (no cross-candidate bleed).
        "TIER1_RESUME": "0",
        "TIER1_MICRO_BATCH_SIZE": "1",  # Cut memory footprint in half to survive 0.4 mesh
        "TIER1_ACCUMULATION_STEPS": "8",  # 1 * 8 = 8 (same effective batch size)

        "TIER1_KINEMATICS_MODE": "direct_uvp",
        "TIER1_ACTIVATION_FN": "silu",

        # --- Strict Pressure & Boundary Enforcement ---
        "TIER1_KINE_P_WEIGHT": "5.0",
        "TIER1_P_GRAD_SUPERVISION": "1.0",
        "TIER1_PRESSURE_BC_MODE": "pointwise",

        # --- Time horizon: AdamW then L-BFGS (see T1SweepCandidate adam_epochs / epochs) ---
        "TIER1_EPOCHS": "60",
        "TIER1_ADAM_EPOCHS": "40",
        "TIER1_DATA_STAGE_EPOCHS": "10",
        "TIER1_LAMBDA_CONT": "1.0",
        "TIER1_LAMBDA_CONT_START": "0.1",
        "TIER1_LAMBDA_CONT_WARMUP_EPOCHS": "10",
    }

    def _env_for(cfg: T1ExplorerConfig) -> Dict[str, str]:
        return {
            "TIER1_EXPERIMENT_NAME": cfg.experiment_name,
            "TIER1_DATASET_TIER": cfg.dataset_tier,
            "TIER1_KINE_WEIGHT_MODE": cfg.kine_weight_mode,
            "TIER1_SDF_WALL_BETA": str(cfg.sdf_wall_beta),
            "TIER1_SDF_WALL_TAU": str(cfg.sdf_wall_tau),
            "TIER1_SDF_GRAD_BETA": str(cfg.sdf_grad_beta),
            "TIER1_SHEAR_TRUE_ALPHA": str(cfg.shear_true_alpha),
            "TIER1_LATENT_DIM": str(cfg.latent_dim),
            "TIER1_DEQ_MAX_ITERS": str(cfg.deq_max_iters),
            "TIER1_NUM_FOURIER_FREQS": str(cfg.num_fourier_freqs),
            "TIER1_KINEMATICS_MODE": cfg.kinematics_mode,
            "TIER1_NS_DERIVATIVE_MODE": cfg.ns_derivative_mode,
            "TIER1_ACTIVATION_FN": cfg.activation_fn,
            "TIER1_FOURIER_BASE": str(cfg.fourier_base),
            "TIER1_LOSS_WEIGHT_MODE": cfg.loss_weight_mode,
            "TIER1_ANDERSON_BETA": str(cfg.anderson_beta),
            "TIER1_LAMBDA_CONT": str(cfg.lambda_cont),
            "TIER1_RE_CURRICULUM": ("1" if cfg.re_curriculum else "0"),
            "TIER1_P_GRAD_SUPERVISION": str(cfg.p_grad_supervision),
            "TIER1_ADVECT_DETACH": ("1" if cfg.advect_detach else "0"),
            "TIER1_PRESSURE_BC_MODE": cfg.pressure_bc_mode,
            "TIER1_MOMENTUM_LOSS_MODE": cfg.momentum_loss_mode,
            "TIER1_USE_HARD_BCS": ("1" if cfg.use_hard_bcs else "0"),
            "TIER1_GLOBAL_POOL_MODE": cfg.global_pool_mode,
            "TIER1_NUM_GLOBAL_TOKENS": str(cfg.num_global_tokens),
            "TIER1_USE_EQUIVARIANT": ("1" if cfg.use_equivariant_conv else "0"),
            "TIER1_USE_SIREN": ("1" if cfg.use_siren_decoder else "0"),
            "TIER1_USE_WIDTH_PRIORS": ("1" if cfg.use_width_priors else "0"),
        }

    common_kwargs = dict(
        dataset_tier="tier1",
        loss_weight_mode="dynamic",
        latent_dim=256,
        deq_max_iters=25,
    )

    sweep_configs: List[T1ExplorerConfig] = [
        T1ExplorerConfig(
            experiment_name="V3_Baseline_Legacy",
            use_hard_bcs=False,
            global_pool_mode="mean",
            use_width_priors=False,
            use_siren_decoder=False,
            **common_kwargs,
        ),
        T1ExplorerConfig(
            experiment_name="V3_Attention_MultiGrid",
            use_hard_bcs=True,
            global_pool_mode="attention",
            use_width_priors=False,
            use_siren_decoder=False,
            **common_kwargs,
        ),
        T1ExplorerConfig(
            experiment_name="V3_Geometric_Priors",
            use_hard_bcs=True,
            global_pool_mode="mean",
            use_width_priors=True,
            use_siren_decoder=False,
            **common_kwargs,
        ),
        T1ExplorerConfig(
            experiment_name="V3_SIREN_Implicit",
            use_hard_bcs=True,
            global_pool_mode="mean",
            use_width_priors=False,
            use_siren_decoder=True,
            **common_kwargs,
        ),
    ]

    candidates: List[T1SweepCandidate] = []
    for cfg in sweep_configs:
        env = {
            **_env_for(cfg),
            **base_overrides,
            "TIER1_TARGET_ANCHOR_FRACTION": "0.7",
        }
        candidates.append(
            T1SweepCandidate(
                name=cfg.experiment_name,
                explorer=cfg,
                env_overrides=env,
                epochs=60,
                warm_up_epochs=5,
                adam_epochs=40,
            )
        )
    return candidates


def _build_same_epoch_comparison(
    completed: List[Dict[str, Any]],
    sweep_dir: Path,
) -> Dict[str, Any]:
    """Build a same-epoch comparison table across all completed candidates.

    Strategy: for each candidate we know ``best_rel_l2``, ``best_phys_score``,
    ``best_loss``, and the actual epoch count trained.  We also parse the
    training diary JSONL (if available) to extract the *last* validation
    snapshot, giving per-candidate metrics at their respective final epoch.

    The comparison aligns all candidates by the **minimum common epoch count**
    so the reader can judge convergence at equal training effort.
    """
    import math

    ok_runs = [r for r in completed if r.get("status") == "ok"]
    if not ok_runs:
        return {"note": "no completed candidates to compare"}

    # Gather per-candidate final-epoch validation metrics from diary files.
    per_candidate: List[Dict[str, Any]] = []
    for run in ok_runs:
        name = run.get("name", "?")
        entry: Dict[str, Any] = {
            "name": name,
            "best_rel_l2": run.get("best_rel_l2"),
            "best_phys_score": run.get("best_phys_score"),
            "best_loss": run.get("best_loss"),
            "early_stopped": run.get("early_stopped", False),
            "n_graphs": run.get("n_graphs"),
            "n_train": run.get("n_train"),
            "n_val": run.get("n_val"),
            "duration_min": run.get("duration_min"),
            "dataset_tier": run.get("dataset_tier")
                or run.get("explorer", {}).get("dataset_tier"),
            "graph_dir": run.get("graph_dir"),
            "ckpt_dir": run.get("ckpt_dir"),
        }

        # Try to parse last validation event from the candidate's training diary.
        diary_metrics = _parse_last_validation_from_diary(run, sweep_dir)
        if diary_metrics:
            entry["last_val"] = diary_metrics

        per_candidate.append(entry)

    # Determine lowest common epoch across candidates (for fair comparison).
    trained_epochs = []
    for c in per_candidate:
        lv = c.get("last_val")
        if lv and isinstance(lv.get("epoch"), (int, float)) and not math.isinf(lv["epoch"]):
            trained_epochs.append(int(lv["epoch"]))
    min_common_epoch = min(trained_epochs) if trained_epochs else None

    return {
        "candidates": per_candidate,
        "min_common_epoch": min_common_epoch,
        "note": (
            "Compare 'best_rel_l2' across candidates for overall winner. "
            "'last_val' shows the final validation snapshot per candidate. "
            "'min_common_epoch' is the lowest epoch any candidate reached — "
            "use diary JSONL files to extract metrics at that epoch for a "
            "strict same-epoch comparison."
        ),
    }


def _parse_last_validation_from_diary(
    run: Dict[str, Any],
    sweep_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Extract the last ``validation`` event from a candidate's training diary JSONL."""
    # The diary path is not stored in the run dict, but we can find it by
    # scanning the reports directory for the diary started closest to the run.
    # Simpler: the trainer stores it in the run result under an undocumented key
    # or we can search by candidate name timestamp.  Since we don't have a
    # guaranteed pointer, we try a robust fallback chain.
    rep = reports_dir()
    candidate_name = run.get("name", "")
    # Heuristic: find diary files created during this sweep run.
    diary_files = sorted(rep.glob("training_diary_tier1_*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not diary_files:
        return None

    # Walk diary files newest-first to find one whose run_start mentions this
    # candidate (via experiment_name).
    for diary_path in reversed(diary_files):
        try:
            last_val = None
            matched = False
            with open(diary_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        evt = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    etype = evt.get("event")
                    if etype == "run_start":
                        t1e = evt.get("t1_explorer") or {}
                        if t1e.get("experiment_name") == candidate_name:
                            matched = True
                    if etype == "validation" and matched:
                        last_val = {
                            "epoch": evt.get("epoch"),
                            "rel_l2": evt.get("scores", {}).get("rel_l2"),
                            "rel_l2_near_wall": evt.get("scores", {}).get("rel_l2_near_wall"),
                            "rel_l2_high_sdf_grad": evt.get("scores", {}).get("rel_l2_high_sdf_grad"),
                            "continuity": evt.get("scores", {}).get("continuity"),
                            "wall_slip": evt.get("scores", {}).get("wall_slip"),
                            "shear_mse": evt.get("scores", {}).get("shear_mse"),
                        }
            if matched and last_val is not None:
                return last_val
        except OSError:
            continue
    return None


def run_sweep(sweep_name: str = "default") -> Path:
    from src.training.train_t1_predictor import train_t1_predictor

    # Keep allocator expansion active for wide latent sweeps unless caller explicitly overrides it.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    started_at = time.time()
    sweep_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_sweep_name = _safe_name(sweep_name)
    sweep_dir = reports_dir() / "experiments" / "sweeps" / f"tier1_{safe_sweep_name}_{sweep_ts}"
    entries_dir = sweep_dir / "entries"
    summary_path = sweep_dir / "sweep_summary.json"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    entries_dir.mkdir(parents=True, exist_ok=True)
    candidates = build_sweep_candidates()
    interrupted = False
    out_path: Optional[Path] = None
    completed: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    state = {"active_candidate": None}
    sanity: Dict[str, Any] = {"enabled": True, "status": "pending"}

    def _emit_once() -> None:
        nonlocal out_path
        rows = sorted(
            [r for r in completed if r.get("status") == "ok"],
            key=lambda x: (x.get("best_rel_l2", float("inf")), x.get("best_phys_score", float("inf"))),
        )

        # Build a same-epoch comparison table from the training diary JSONL files.
        # Each candidate writes a diary; we parse the last validation event from each
        # and align by epoch count to enable fair apples-to-apples comparison.
        same_epoch_comparison = _build_same_epoch_comparison(completed, sweep_dir)

        payload = {
            "tier": "tier1",
            "sweep_name": sweep_name,
            "ts_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "run_folder": str(sweep_dir),
            "sweep_defaults": {"epochs": 60, "warm_up_epochs": 5, "adam_epochs": 40},
            "interrupted": interrupted,
            "active_candidate_when_stopped": state["active_candidate"],
            "elapsed_minutes": (time.time() - started_at) / 60.0,
            "n_candidates_total": len(candidates),
            "n_completed": len(completed),
            "finished": (not interrupted) and (state["active_candidate"] is None),
            "sanity_check": sanity,
            "completed": completed,
            "failures": failures,
            "leaderboard": [
                {
                    "rank": i + 1,
                    "name": r.get("name"),
                    "best_rel_l2": r.get("best_rel_l2"),
                    "best_phys_score": r.get("best_phys_score"),
                    "best_loss": r.get("best_loss"),
                    "early_stopped": r.get("early_stopped"),
                }
                for i, r in enumerate(rows)
            ],
            "same_epoch_comparison": same_epoch_comparison,
        }
        out_path = _write_json(summary_path, payload)

    def _write_entry_debug(
        *,
        phase: str,
        idx: Optional[int],
        name: str,
        explorer: T1ExplorerConfig,
        env_overrides: Dict[str, str],
        duration_min: float,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        tb: Optional[str] = None,
    ) -> Path:
        prefix = f"{idx:02d}_" if idx is not None else ""
        safe_name = _safe_name(name, max_len=120)
        entry_path = entries_dir / f"{prefix}{safe_name}.json"
        payload: Dict[str, Any] = {
            "tier": "tier1",
            "sweep_name": sweep_name,
            "phase": phase,
            "candidate_name": name,
            "status": status,
            "duration_min": duration_min,
            "ts_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "explorer": explorer.to_serializable(),
            "env_overrides": env_overrides,
        }
        if result is not None:
            payload["result"] = result
        if error is not None:
            payload["error"] = error
        if tb is not None:
            payload["traceback"] = tb
        return _write_json(entry_path, payload)

    def _handle_interrupt(signum, _frame):
        nonlocal interrupted
        interrupted = True
        print(f"\n⚠️ Received signal {signum}; finalizing one consolidated sweep report...")
        _emit_once()
        raise KeyboardInterrupt()

    prev_sigint = signal.getsignal(signal.SIGINT)
    prev_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _handle_interrupt)
    signal.signal(signal.SIGTERM, _handle_interrupt)
    atexit.register(_emit_once)

    # Fail-fast sanity check: run a tiny 1-epoch AdamW probe before full sweep.
    sanity_cand = candidates[0]
    state["active_candidate"] = f"sanity::{sanity_cand.name}"
    sanity_overrides = dict(sanity_cand.env_overrides)
    sanity_overrides["TIER1_SKIP_EXPERIMENT_ARTIFACT"] = "1"
    sanity_overrides["TIER1_DISABLE_FIGURES"] = "1"
    sanity_overrides["TIER1_DISABLE_STAGE_A_ARTIFACTS"] = "1"
    sanity_overrides["PHASE1_TRAINING_DIARY"] = "0"
    # Exploration mode sanity: keep this Adam-only for speed.
    sanity_overrides["TIER1_EPOCHS"] = "1"
    sanity_overrides["TIER1_ADAM_EPOCHS"] = "1"
    sanity_overrides["TIER1_USE_LBFGS"] = "0"
    # Hard-pin: explorer must never inherit weights or resume (shell env cannot override).
    sanity_overrides["TIER1_INIT_FROM_BEST"] = "0"
    sanity_overrides["TIER1_RESUME"] = "0"
    sanity_ckpt = sweep_dir / "checkpoints" / "sanity_probe"
    sanity_ckpt.mkdir(parents=True, exist_ok=True)
    sanity_overrides["TIER1_CKPT_DIR"] = str(sanity_ckpt)
    print(
        "🔎 Sanity effective settings: "
        f"TIER1_CKPT_EVERY={sanity_overrides.get('TIER1_CKPT_EVERY', 'unset')}, "
        f"TIER1_PRESSURE_BC_MODE={sanity_overrides.get('TIER1_PRESSURE_BC_MODE', 'unset')}, "
        f"TIER1_EPOCHS={sanity_overrides.get('TIER1_EPOCHS', 'unset')}, "
        f"TIER1_ADAM_EPOCHS={sanity_overrides.get('TIER1_ADAM_EPOCHS', 'unset')}, "
        f"TIER1_USE_LBFGS={sanity_overrides.get('TIER1_USE_LBFGS', 'unset')}, "
        f"TIER1_INIT_FROM_BEST={sanity_overrides.get('TIER1_INIT_FROM_BEST', 'unset')}, "
        f"TIER1_RESUME={sanity_overrides.get('TIER1_RESUME', 'unset')}"
    )
    prev_env = _safe_env_set(sanity_overrides)
    t0 = time.time()
    try:
        sanity_result = train_t1_predictor(
            epochs=1,
            warm_up_epochs=0,
            adam_epochs=1,
            explorer=None,  # Force the trainer to read from os.environ
        )
        sanity = {
            "enabled": True,
            "status": "passed",
            "candidate": sanity_cand.name,
            "duration_min": (time.time() - t0) / 60.0,
            "result": sanity_result if sanity_result is not None else {"status": "unknown"},
        }
    except Exception as exc:
        tb = traceback.format_exc()
        sanity = {
            "enabled": True,
            "status": "failed",
            "candidate": sanity_cand.name,
            "duration_min": (time.time() - t0) / 60.0,
            "error": repr(exc),
            "traceback": tb,
        }
        failures.append(
            {
                "name": f"sanity::{sanity_cand.name}",
                "error": repr(exc),
                "duration_min": (time.time() - t0) / 60.0,
            }
        )
        entry_path = _write_entry_debug(
            phase="sanity",
            idx=None,
            name=f"sanity::{sanity_cand.name}",
            explorer=sanity_cand.explorer,
            env_overrides=sanity_overrides,
            duration_min=(time.time() - t0) / 60.0,
            status="failed",
            error=repr(exc),
            tb=tb,
        )
        print("❌ Sanity check failed; aborting sweep before full candidates.")
        print(f"🧾 Failure details written to: {entry_path}")
        print(tb)
        _safe_env_restore(prev_env)
        _emit_once()
        signal.signal(signal.SIGINT, prev_sigint)
        signal.signal(signal.SIGTERM, prev_sigterm)
        return out_path
    finally:
        _safe_env_restore(prev_env)
    if sanity.get("status") == "passed":
        _write_entry_debug(
            phase="sanity",
            idx=None,
            name=f"sanity::{sanity_cand.name}",
            explorer=sanity_cand.explorer,
            env_overrides=sanity_overrides,
            duration_min=(time.time() - t0) / 60.0,
            status="passed",
            result=sanity.get("result") if isinstance(sanity.get("result"), dict) else None,
        )
    # Per-candidate checkpoint directories live under the sweep folder so
    # candidates never read/overwrite each other's optimizer state or epoch counter.
    ckpt_root = sweep_dir / "checkpoints"

    for idx, cand in enumerate(candidates, start=1):
        state["active_candidate"] = cand.name
        print(f"\n=== [{idx}/{len(candidates)}] Tier1 sweep candidate: {cand.name} ===")
        overrides = dict(cand.env_overrides)
        overrides["TIER1_SKIP_EXPERIMENT_ARTIFACT"] = "1"
        overrides["TIER1_DISABLE_FIGURES"] = "1"
        overrides["TIER1_DISABLE_STAGE_A_ARTIFACTS"] = "1"

        # Checkpoint isolation: each candidate saves/loads from its own directory.
        cand_ckpt_dir = ckpt_root / _safe_name(cand.name)
        cand_ckpt_dir.mkdir(parents=True, exist_ok=True)
        overrides["TIER1_CKPT_DIR"] = str(cand_ckpt_dir)
        # Hard-pin: never warm-start or resume (shell env cannot override).
        overrides["TIER1_INIT_FROM_BEST"] = "0"
        overrides["TIER1_RESUME"] = "0"

        print(
            "🔎 Candidate effective settings: "
            f"TIER1_CKPT_EVERY={overrides.get('TIER1_CKPT_EVERY', 'unset')}, "
            f"TIER1_PRESSURE_BC_MODE={overrides.get('TIER1_PRESSURE_BC_MODE', 'unset')}, "
            f"TIER1_INIT_FROM_BEST={overrides.get('TIER1_INIT_FROM_BEST', 'unset')}, "
            f"TIER1_RESUME={overrides.get('TIER1_RESUME', 'unset')}, "
            f"TIER1_CKPT_DIR={cand_ckpt_dir}"
        )
        prev_env = _safe_env_set(overrides)
        t0 = time.time()
        try:
            result = train_t1_predictor(
                epochs=cand.epochs,
                warm_up_epochs=cand.warm_up_epochs,
                adam_epochs=cand.adam_epochs,
                explorer=None,  # Force the trainer to read from os.environ
            )
            if result is None:
                result = {"status": "unknown"}
            row = {
                "name": cand.name,
                "explorer": cand.explorer.to_serializable(),
                "env_overrides": overrides,
                "duration_min": (time.time() - t0) / 60.0,
                "ckpt_dir": str(cand_ckpt_dir),
                **result,
            }
            completed.append(row)
            _write_entry_debug(
                phase="candidate",
                idx=idx,
                name=cand.name,
                explorer=cand.explorer,
                env_overrides=overrides,
                duration_min=(time.time() - t0) / 60.0,
                status="ok",
                result=result if isinstance(result, dict) else {"status": "unknown"},
            )
        except KeyboardInterrupt:
            interrupted = True
            failures.append({"name": cand.name, "error": "KeyboardInterrupt", "duration_min": (time.time() - t0) / 60.0})
            _write_entry_debug(
                phase="candidate",
                idx=idx,
                name=cand.name,
                explorer=cand.explorer,
                env_overrides=overrides,
                duration_min=(time.time() - t0) / 60.0,
                status="interrupted",
                error="KeyboardInterrupt",
                tb=traceback.format_exc(),
            )
            break
        except Exception as exc:
            tb = traceback.format_exc()
            failures.append({"name": cand.name, "error": repr(exc), "duration_min": (time.time() - t0) / 60.0})
            _write_entry_debug(
                phase="candidate",
                idx=idx,
                name=cand.name,
                explorer=cand.explorer,
                env_overrides=overrides,
                duration_min=(time.time() - t0) / 60.0,
                status="failed",
                error=repr(exc),
                tb=tb,
            )
        finally:
            _safe_env_restore(prev_env)

    state["active_candidate"] = None
    _emit_once()
    signal.signal(signal.SIGINT, prev_sigint)
    signal.signal(signal.SIGTERM, prev_sigterm)
    print(f"📁 Sweep run folder: {sweep_dir}")
    print(f"📘 Sweep summary: {summary_path}")
    return out_path


def filter_graph_paths_by_geometry_level(
    graph_paths: List[Path], raw_tier_dir: Path, level: int
) -> List[Path]:
    """Keep only stems whose ``vessel_*.json`` lists ``\"level\"`` == ``level``."""
    kept: List[Path] = []
    for p in graph_paths:
        stem = p.stem
        js = raw_tier_dir / f"{stem}.json"
        if not js.is_file():
            continue
        try:
            with open(js, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if int(meta.get("level", -1)) == level:
                kept.append(p)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return kept


if __name__ == "__main__":
    # One-click PyCharm run: execute the Tier 1 sweep directly.
    run_sweep(sweep_name="tier1")
