"""
Tier 1 **architecture / loss explorer** — env-driven knobs for systematic experiments.

Use this when iterating toward a tight anchor Rel L2 (e.g. <5% on level-1 vessels). After each run,
compare ``reports/experiments/tier1_<name>_*.json`` and the training diary.

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
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.paths import reports_dir


@dataclass
class T1ExplorerConfig:
    experiment_name: str = "default"
    kine_weight_mode: str = "uniform"  # uniform | sdf_wall | sdf_grad | shear_true
    sdf_wall_beta: float = 2.0
    sdf_wall_tau: float = 0.12
    sdf_grad_beta: float = 1.0
    shear_true_alpha: float = 1.0
    latent_dim: int = 64
    deq_max_iters: int = 15
    num_fourier_freqs: int = 8
    geometry_level: Optional[int] = None  # filter json "level"
    kinematics_mode: str = "direct_uvp"  # stream | direct_uvp
    ns_derivative_mode: str = "wls"  # wls | autograd
    activation_fn: str = "silu"  # relu | silu | gelu
    fourier_base: float = 2.0
    loss_weight_mode: str = "dynamic"  # dynamic | fixed | grad_norm
    anderson_beta: float = 0.8
    lambda_cont: float = 1.0
    re_curriculum: bool = False
    p_grad_supervision: float = 0.0
    advect_detach: bool = False
    pressure_bc_mode: str = "mean"  # mean | pointwise | mean_var
    momentum_loss_mode: str = "huber"  # huber | mse

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

        return T1ExplorerConfig(
            experiment_name=os.environ.get("TIER1_EXPERIMENT_NAME", "default").strip() or "default",
            kine_weight_mode=mode,
            sdf_wall_beta=_f("TIER1_SDF_WALL_BETA", 2.0),
            sdf_wall_tau=_f("TIER1_SDF_WALL_TAU", 0.12),
            sdf_grad_beta=_f("TIER1_SDF_GRAD_BETA", 1.0),
            shear_true_alpha=_f("TIER1_SHEAR_TRUE_ALPHA", 1.0),
            latent_dim=int(os.environ.get("TIER1_LATENT_DIM", "64")),
            deq_max_iters=int(os.environ.get("TIER1_DEQ_MAX_ITERS", "15")),
            num_fourier_freqs=int(os.environ.get("TIER1_NUM_FOURIER_FREQS", "8")),
            geometry_level=geometry_level,
            kinematics_mode=kinematics_mode,
            ns_derivative_mode=ns_derivative_mode,
            activation_fn=activation,
            fourier_base=float(os.environ.get("TIER1_FOURIER_BASE", "2.0")),
            loss_weight_mode=loss_weight_mode,
            anderson_beta=float(os.environ.get("TIER1_ANDERSON_BETA", "0.8")),
            lambda_cont=float(os.environ.get("TIER1_LAMBDA_CONT", "1.0")),
            re_curriculum=re_curriculum,
            p_grad_supervision=float(os.environ.get("TIER1_P_GRAD_SUPERVISION", "0.0")),
            advect_detach=os.environ.get("TIER1_ADVECT_DETACH", "0").strip().lower() in ("1", "true", "yes", "on"),
            pressure_bc_mode=pressure_bc_mode,
            momentum_loss_mode=momentum_loss_mode,
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
    return path


def build_sweep_candidates() -> List[T1SweepCandidate]:
    """
    Surgical Tier 1 5-candidate sweep for a ~5 hour budget.

    Fixed baseline for all candidates:
    - direct_uvp kinematics
    - SiLU activation
    - kinematic pressure weight = 1.0
    - L-BFGS enabled
    - 12 total epochs, 10 AdamW epochs (final 2 epochs in L-BFGS phase)
    """
    base_overrides = {
        "TIER1_DISABLE_FIGURES": "1",
        "TIER1_CKPT_EVERY": "10",
        "TIER1_EARLY_STOP_PATIENCE": "5",
        "TIER1_LOSS_WEIGHT_MODE": "fixed",
        "TIER1_USE_LBFGS": "1",
        "TIER1_KINEMATICS_MODE": "direct_uvp",
        "TIER1_ACTIVATION_FN": "silu",
        "TIER1_KINE_P_WEIGHT": "1.0",
        "TIER1_EPOCHS": "12",
        "TIER1_ADAM_EPOCHS": "10",
    }
    def _env_for(cfg: T1ExplorerConfig) -> Dict[str, str]:
        return {
            "TIER1_EXPERIMENT_NAME": cfg.experiment_name,
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
        }
    candidate_specs = [
        ("C_A_Standard", 1, "sdf_wall", 2.0, 0.5),
        ("C_B_HighFreq", 1, "sdf_wall", 1.5, 0.5),
        ("C_C_SDFGrad", 1, "sdf_grad", 2.0, 0.5),
        ("C_D_AnchorHeavy", 1, "sdf_wall", 2.0, 0.7),
        ("C_E_CoupledNS", 0, "sdf_wall", 2.0, 0.5),
    ]
    candidates: List[T1SweepCandidate] = []
    for name, advect_detach, weight_mode, fourier_base, anchor_frac in candidate_specs:
        cfg = T1ExplorerConfig(
            experiment_name=name,
            kinematics_mode="direct_uvp",
            ns_derivative_mode="wls",
            activation_fn="silu",
            loss_weight_mode="fixed",
            fourier_base=float(fourier_base),
            advect_detach=bool(advect_detach),
            kine_weight_mode=weight_mode,
        )
        env = {
            **base_overrides,
            **_env_for(cfg),
            "TIER1_TARGET_ANCHOR_FRACTION": str(anchor_frac),
        }
        candidates.append(
            T1SweepCandidate(
                name=cfg.experiment_name,
                explorer=cfg,
                env_overrides=env,
                epochs=12,
                warm_up_epochs=3,
                adam_epochs=10,
            )
        )
    return candidates


def run_sweep(sweep_name: str = "default") -> Path:
    from src.training.train_t1_predictor import train_t1_predictor

    started_at = time.time()
    candidates = build_sweep_candidates()
    interrupted = False
    out_path: Optional[Path] = None
    completed: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    state = {"active_candidate": None}
    sanity: Dict[str, Any] = {"enabled": True, "status": "pending"}

    def _emit_once() -> None:
        nonlocal out_path
        if out_path is not None:
            return
        rows = sorted(
            [r for r in completed if r.get("status") == "ok"],
            key=lambda x: (x.get("best_rel_l2", float("inf")), x.get("best_phys_score", float("inf"))),
        )
        payload = {
            "tier": "tier1",
            "sweep_name": sweep_name,
            "ts_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "sweep_defaults": {"epochs": 15, "warm_up_epochs": 3, "adam_epochs": 15},
            "interrupted": interrupted,
            "active_candidate_when_stopped": state["active_candidate"],
            "elapsed_minutes": (time.time() - started_at) / 60.0,
            "n_candidates_total": len(candidates),
            "n_completed": len(completed),
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
        }
        out_path = write_t1_sweep_report(payload, sweep_name=sweep_name)

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

    # Fail-fast sanity check: run a tiny 1-epoch probe before launching full sweep.
    sanity_cand = candidates[0]
    state["active_candidate"] = f"sanity::{sanity_cand.name}"
    sanity_overrides = dict(sanity_cand.env_overrides)
    sanity_overrides["TIER1_SKIP_EXPERIMENT_ARTIFACT"] = "1"
    sanity_overrides["TIER1_DISABLE_FIGURES"] = "1"
    sanity_overrides["PHASE1_TRAINING_DIARY"] = "0"
    prev_env = _safe_env_set(sanity_overrides)
    t0 = time.time()
    try:
        sanity_result = train_t1_predictor(
            epochs=1,
            warm_up_epochs=0,
            adam_epochs=1,
            explorer=sanity_cand.explorer,
        )
        sanity = {
            "enabled": True,
            "status": "passed",
            "candidate": sanity_cand.name,
            "duration_min": (time.time() - t0) / 60.0,
            "result": sanity_result if sanity_result is not None else {"status": "unknown"},
        }
    except Exception as exc:
        sanity = {
            "enabled": True,
            "status": "failed",
            "candidate": sanity_cand.name,
            "duration_min": (time.time() - t0) / 60.0,
            "error": repr(exc),
        }
        failures.append(
            {
                "name": f"sanity::{sanity_cand.name}",
                "error": repr(exc),
                "duration_min": (time.time() - t0) / 60.0,
            }
        )
        _safe_env_restore(prev_env)
        _emit_once()
        signal.signal(signal.SIGINT, prev_sigint)
        signal.signal(signal.SIGTERM, prev_sigterm)
        return out_path
    finally:
        _safe_env_restore(prev_env)

    for idx, cand in enumerate(candidates, start=1):
        state["active_candidate"] = cand.name
        print(f"\n=== [{idx}/{len(candidates)}] Tier1 sweep candidate: {cand.name} ===")
        overrides = dict(cand.env_overrides)
        overrides["TIER1_SKIP_EXPERIMENT_ARTIFACT"] = "1"
        prev_env = _safe_env_set(overrides)
        t0 = time.time()
        try:
            result = train_t1_predictor(
                epochs=cand.epochs,
                warm_up_epochs=cand.warm_up_epochs,
                adam_epochs=cand.adam_epochs,
                explorer=cand.explorer,
            )
            if result is None:
                result = {"status": "unknown"}
            row = {
                "name": cand.name,
                "explorer": cand.explorer.to_serializable(),
                "env_overrides": overrides,
                "duration_min": (time.time() - t0) / 60.0,
                **result,
            }
            completed.append(row)
        except KeyboardInterrupt:
            interrupted = True
            failures.append({"name": cand.name, "error": "KeyboardInterrupt", "duration_min": (time.time() - t0) / 60.0})
            break
        except Exception as exc:
            failures.append({"name": cand.name, "error": repr(exc), "duration_min": (time.time() - t0) / 60.0})
        finally:
            _safe_env_restore(prev_env)

    _emit_once()
    signal.signal(signal.SIGINT, prev_sigint)
    signal.signal(signal.SIGTERM, prev_sigterm)
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
