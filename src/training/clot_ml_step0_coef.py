"""Step 0 ML ladder: learn rule coefficients (pred kine, LOAO on anchors)."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_localized_spatial import LocalizedSpatialConfig
from src.core_physics.clot_temporal_growth_rules import (
    TemporalGrowthRuleConfig,
    _localized_prog_template,
    deploy_score_from_eval_row,
    eval_temporal_rule_on_anchor,
    reset_temporal_kinematics_cache,
)
from src.utils.paths import get_project_root

# inc40 hand-tuned reference (pred-kine mean deploy ~0.538 on 6 anchors).
INC40_REFERENCE = {
    "neg_dx": 0.25,
    "sep": 0.0,
    "stasis": 0.0,
    "lgrad": 0.0,
    "onset": 0.40,
    "start_frac": 0.05,
    "end_frac": 0.22,
    "top_frac": 0.20,
    "power": 1.5,
    "boost": 1.0,
}

PARAM_NAMES: tuple[str, ...] = tuple(INC40_REFERENCE.keys())

BOUNDS: dict[str, tuple[float, float]] = {
    "neg_dx": (0.10, 0.70),
    "sep": (0.0, 0.45),
    "stasis": (0.0, 0.45),
    "lgrad": (0.0, 0.45),
    "onset": (0.25, 0.50),
    "start_frac": (0.03, 0.10),
    "end_frac": (0.18, 0.32),
    "top_frac": (0.15, 0.28),
    "power": (1.2, 2.0),
    "boost": (1.0, 1.75),
}


@dataclass(frozen=True)
class Step0RuleCoefs:
    neg_dx: float = 0.25
    sep: float = 0.0
    stasis: float = 0.0
    lgrad: float = 0.0
    onset: float = 0.40
    start_frac: float = 0.05
    end_frac: float = 0.22
    top_frac: float = 0.20
    power: float = 1.5
    boost: float = 1.0

    @classmethod
    def inc40_baseline(cls) -> Step0RuleCoefs:
        return cls(**INC40_REFERENCE)

    @classmethod
    def from_vector(cls, x: Sequence[float]) -> Step0RuleCoefs:
        vals = {name: float(x[i]) for i, name in enumerate(PARAM_NAMES)}
        return cls(**vals)

    def to_vector(self) -> list[float]:
        d = asdict(self)
        return [float(d[name]) for name in PARAM_NAMES]

    @classmethod
    def bounds_vectors(cls) -> tuple[list[float], list[float]]:
        lo = [BOUNDS[n][0] for n in PARAM_NAMES]
        hi = [BOUNDS[n][1] for n in PARAM_NAMES]
        return lo, hi

    def to_rule_config(self, *, name: str = "ml_step0_coef") -> TemporalGrowthRuleConfig:
        base = _localized_prog_template()
        loc = base.localized
        if loc is None:
            loc = LocalizedSpatialConfig(mode="wall_half")
        loc = replace(
            loc,
            mode="wall_half",
            segment_top_frac=float(self.top_frac),
            skip_wall_arc_frac=0.0,
            neg_dx_risk_weight=float(self.neg_dx),
            sep_stream_risk_weight=float(self.sep),
            stasis_risk_weight=float(self.stasis),
            low_grad_risk_weight=float(self.lgrad),
            species_gt_top_q=0.0,
            species_risk_weight=0.0,
            normalize_risk_per_half=True,
            wall_halves=("lower", "upper"),
        )
        return replace(
            base,
            name=name,
            kind="progressive_topk",
            localized=loc,
            start_frac=float(self.start_frac),
            end_frac=float(self.end_frac),
            power=float(self.power),
            global_onset_frac=float(self.onset),
            promotion_boost=float(self.boost),
        )

    def to_env(self) -> dict[str, str]:
        """Env vars for timeline viz / promote (pred-kine deploy)."""
        return {
            "CLOT_TEMPORAL_RULE_KIND": "progressive_topk",
            "CLOT_TEMPORAL_RULE_NAME": "ml_step0_coef",
            "CLOT_TEMPORAL_START_FRAC": f"{self.start_frac:.4f}",
            "CLOT_TEMPORAL_END_FRAC": f"{self.end_frac:.4f}",
            "CLOT_TEMPORAL_POWER": f"{self.power:.4f}",
            "CLOT_TEMPORAL_GLOBAL_ONSET": f"{self.onset:.4f}",
            "CLOT_TEMPORAL_PROMOTION_BOOST": f"{self.boost:.4f}",
            "CLOT_TEMPORAL_ONSET_SPREAD": "0.55",
            "CLOT_TEMPORAL_MIN_ONSET": "0.08",
            "CLOT_LOCALIZED_MODE": "wall_half",
            "CLOT_LOCALIZED_TOP_FRAC": f"{self.top_frac:.4f}",
            "CLOT_LOCALIZED_SKIP_ARC": "0.00",
            "CLOT_LOCALIZED_NEG_DX_WEIGHT": f"{self.neg_dx:.4f}",
            "CLOT_SHEAR_W_NEG_DX": f"{self.neg_dx:.4f}",
            "CLOT_SHEAR_W_SEP": f"{self.sep:.4f}",
            "CLOT_SHEAR_W_STASIS": f"{self.stasis:.4f}",
            "CLOT_SHEAR_W_LGRAD": f"{self.lgrad:.4f}",
            "CLOT_TEMPORAL_VEL_SOURCE": "kinematics",
        }

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Step0RuleCoefs:
        return cls(**{k: float(d[k]) for k in PARAM_NAMES})


def load_step0_coef_json(path: Path | str) -> Step0RuleCoefs:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    coef = payload.get("coef") or payload
    return Step0RuleCoefs.from_dict(coef)


def discover_anchor_paths(anchor_dir: Path) -> list[Path]:
    return sorted(anchor_dir.glob("patient*.pt"))


def eval_coef_on_anchor(
    coef: Step0RuleCoefs,
    *,
    graph_path: Path,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    pair_stride: int = 1,
) -> dict[str, Any]:
    reset_temporal_kinematics_cache()
    data = torch.load(graph_path, map_location=device, weights_only=False)
    stem = graph_path.stem
    cfg = coef.to_rule_config()
    row = eval_temporal_rule_on_anchor(
        data,
        cfg,
        stem=stem,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        pair_stride=max(1, int(pair_stride)),
    )
    row["deploy_score"] = deploy_score_from_eval_row(row)
    return row


def eval_coef_on_anchors(
    coef: Step0RuleCoefs,
    *,
    anchor_paths: Sequence[Path],
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    pair_stride: int = 1,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in anchor_paths:
        rows.append(
            eval_coef_on_anchor(
                coef,
                graph_path=path,
                device=device,
                phys_cfg=phys_cfg,
                bio_cfg=bio_cfg,
                pair_stride=pair_stride,
            )
        )
    return rows


def _objective_from_rows(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 1e6
    deploys = [float(r.get("deploy_score", float("nan"))) for r in rows]
    deploys = [d for d in deploys if d == d]
    if not deploys:
        return 1e6
    mean_deploy = sum(deploys) / len(deploys)
    penalty = 0.0
    for r in rows:
        early = float(r.get("early_mean_pred_frac", 0.0) or 0.0)
        pred = float(r.get("tfinal_band_pred_frac", 0.0) or 0.0)
        if early > 0.08:
            penalty += 0.15 * (early - 0.08)
        if pred > 0.65:
            penalty += 0.10 * (pred - 0.65)
    return -(mean_deploy - penalty)


_eval_cache: dict[tuple[str, tuple[float, ...]], float] = {}


def clear_step0_eval_cache() -> None:
    _eval_cache.clear()


def objective_vector(
    x: Sequence[float],
    *,
    anchor_paths: Sequence[Path],
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    cache_key: str = "",
    pair_stride: int = 1,
) -> float:
    key = (cache_key, int(pair_stride), tuple(round(float(v), 5) for v in x))
    if key in _eval_cache:
        return _eval_cache[key]
    coef = Step0RuleCoefs.from_vector(x)
    rows = eval_coef_on_anchors(
        coef,
        anchor_paths=anchor_paths,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        pair_stride=pair_stride,
    )
    val = _objective_from_rows(rows)
    _eval_cache[key] = val
    return val


def optimize_coef_on_anchors(
    anchor_paths: Sequence[Path],
    *,
    device: torch.device,
    seed: int = 0,
    maxiter: int = 24,
    popsize: int = 12,
    x0: Step0RuleCoefs | None = None,
    method: str = "search",
    search_pair_stride: int = 4,
) -> tuple[Step0RuleCoefs, float, list[dict[str, Any]]]:
    import numpy as np

    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    lo, hi = Step0RuleCoefs.bounds_vectors()
    lo_a = np.asarray(lo, dtype=np.float64)
    hi_a = np.asarray(hi, dtype=np.float64)
    init = np.clip(
        np.asarray((x0 or Step0RuleCoefs.inc40_baseline()).to_vector(), dtype=np.float64),
        lo_a,
        hi_a,
    )
    cache_key = "|".join(p.stem for p in anchor_paths)

    stride = max(1, int(search_pair_stride))

    def _obj(x: Sequence[float]) -> float:
        return objective_vector(
            x,
            anchor_paths=anchor_paths,
            device=device,
            phys_cfg=phys,
            bio_cfg=bio,
            cache_key=cache_key,
            pair_stride=stride,
        )

    if method == "de":
        from scipy.optimize import differential_evolution

        n_dim = init.shape[0]
        n_pop = max(5, int(popsize) * n_dim)
        rng = np.random.default_rng(seed)
        init_pop = rng.uniform(lo_a, hi_a, size=(n_pop, n_dim))
        init_pop[0] = init
        result = differential_evolution(
            _obj,
            bounds=list(zip(lo, hi)),
            seed=seed,
            maxiter=maxiter,
            popsize=popsize,
            tol=0.01,
            polish=True,
            init=init_pop,
            updating="immediate",
            workers=1,
        )
        best_x = result.x
        best_loss = float(result.fun)
    else:
        # Random search + coordinate polish (default; ~maxiter evals, interactive-friendly).
        rng = np.random.default_rng(seed)
        n_samples = max(8, int(maxiter))
        candidates = [init]
        candidates.append(rng.uniform(lo_a, hi_a, size=init.shape[0]))
        for _ in range(n_samples - 2):
            candidates.append(rng.uniform(lo_a, hi_a, size=init.shape[0]))
        best_x = init
        best_loss = _obj(init)
        for cand in candidates[1:]:
            loss = _obj(cand)
            if loss < best_loss:
                best_x, best_loss = cand, loss
        # 1D coordinate steps from best.
        step = 0.25 * (hi_a - lo_a)
        for _ in range(2):
            improved = False
            for j in range(init.shape[0]):
                for sign in (-1.0, 1.0):
                    trial = best_x.copy()
                    trial[j] = float(np.clip(trial[j] + sign * step[j], lo_a[j], hi_a[j]))
                    loss = _obj(trial)
                    if loss < best_loss:
                        best_x, best_loss = trial, loss
                        improved = True
            step *= 0.5
            if not improved:
                break

    best = Step0RuleCoefs.from_vector(best_x)
    rows = eval_coef_on_anchors(
        best,
        anchor_paths=anchor_paths,
        device=device,
        phys_cfg=phys,
        bio_cfg=bio,
        pair_stride=1,
    )
    return best, float(-best_loss), rows


def run_loao_step0(
    anchor_dir: Path,
    *,
    device: torch.device,
    seed: int = 0,
    maxiter: int = 24,
    popsize: int = 12,
    method: str = "search",
    search_pair_stride: int = 4,
) -> dict[str, Any]:
    paths = discover_anchor_paths(anchor_dir)
    if len(paths) < 2:
        raise ValueError(f"need >=2 anchors in {anchor_dir}")

    baseline = Step0RuleCoefs.inc40_baseline()
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    baseline_rows = eval_coef_on_anchors(
        baseline, anchor_paths=paths, device=device, phys_cfg=phys, bio_cfg=bio
    )
    baseline_mean = sum(r["deploy_score"] for r in baseline_rows) / len(baseline_rows)

    folds: list[dict[str, Any]] = []
    holdout_deploys: list[float] = []
    for hold in paths:
        train = [p for p in paths if p != hold]
        best, _, train_rows = optimize_coef_on_anchors(
            train,
            device=device,
            seed=seed,
            maxiter=maxiter,
            popsize=popsize,
            method=method,
            search_pair_stride=search_pair_stride,
        )
        hold_row = eval_coef_on_anchor(
            best,
            graph_path=hold,
            device=device,
            phys_cfg=phys,
            bio_cfg=bio,
            pair_stride=1,
        )
        holdout_deploys.append(float(hold_row["deploy_score"]))
        folds.append(
            {
                "holdout": hold.stem,
                "coef": best.to_dict(),
                "holdout_deploy": hold_row["deploy_score"],
                "holdout_tfinal_shape": hold_row.get("tfinal_clot_shape"),
                "holdout_band_f1": hold_row.get("tfinal_band_f1"),
                "train_mean_deploy": sum(r["deploy_score"] for r in train_rows) / len(train_rows),
            }
        )

    loao_mean = sum(holdout_deploys) / len(holdout_deploys)
    # Final coef: optimize on all anchors for deploy + viz.
    final_coef, _, final_rows = optimize_coef_on_anchors(
        paths,
        device=device,
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        method=method,
        search_pair_stride=search_pair_stride,
    )
    final_mean = sum(r["deploy_score"] for r in final_rows) / len(final_rows)

    return {
        "step": 0,
        "anchor_dir": str(anchor_dir),
        "vel_source": "kinematics",
        "baseline_inc40_mean_deploy": baseline_mean,
        "loao_mean_deploy": loao_mean,
        "full_train_mean_deploy": final_mean,
        "coef": final_coef.to_dict(),
        "vector": final_coef.to_vector(),
        "per_anchor_final": final_rows,
        "loao_folds": folds,
        "pass_loao_vs_baseline": loao_mean >= baseline_mean + 0.01,
    }


def default_out_dir() -> Path:
    return get_project_root() / "outputs/biochem/clot_ml_ladder/step0_coef"
