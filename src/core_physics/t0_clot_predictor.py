"""T0 clot predictors from GT flow + species (no GT spf.mu input).

Predictor families:
  - ``mu_growth``: COMSOL Carreau + mu1(Mat)+mu2(fi), clot from dmu vs t0 anchor
  - ``species_hard``: COMSOL step thresholds on Mat / fi
  - ``species_soft``: sigmoid Mat / fi (phi_clot_from_mat_fi)
"""

from __future__ import annotations

import contextlib
import math
import os
from dataclasses import dataclass, field
from typing import Iterator

import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_phi_simple import (
    mat_si_for_gelation_from_log1p,
    species_log1p_nd_to_si,
)
from src.core_physics.t0_mu_physics import (
    clot_phi_binary_from_mu_growth,
    gt_clot_phi_at_time,
    gt_mu_anchor_cap_si,
    predict_mu_si_at_time,
    rollout_t0_clot_phi,
    t0_physics_env,
)
from src.training.train_clot_phi_simple import _clot_metrics
from src.utils.rheology import phi_clot_from_mat_fi

_GAMMA_SCALE_KEYS = (
    "CLOT_PHI_PHYSICS_GAMMA_SCALE",
    "CLOT_PHI_PHYSICS_POISEUILLE_SCALE",
)


@contextlib.contextmanager
def t0_gt_baseline_env(
    *,
    gamma_mode: str,
    gamma_scale: float = 1.0,
    poiseuille_scale: float | None = None,
    hard_step: bool = True,
) -> Iterator[dict[str, str]]:
    """GT oracle env: no COMSOL spf.sr sidecar, no deploy caps."""
    saved = {k: os.environ.get(k) for k in _GAMMA_SCALE_KEYS}
    saved["CLOT_PHI_PHYSICS_COMSOL_SR_ANCHOR"] = os.environ.get("CLOT_PHI_PHYSICS_COMSOL_SR_ANCHOR")
    with t0_physics_env("", gamma_mode=gamma_mode, hard_step=hard_step) as physics:
        os.environ.pop("CLOT_PHI_PHYSICS_COMSOL_SR_ANCHOR", None)
        os.environ["CLOT_PHI_PHYSICS_GAMMA_MODE"] = gamma_mode
        os.environ["CLOT_PHI_PHYSICS_GAMMA_SCALE"] = f"{float(gamma_scale):g}"
        if poiseuille_scale is not None:
            os.environ["CLOT_PHI_PHYSICS_POISEUILLE_SCALE"] = f"{float(poiseuille_scale):g}"
        cfg = dict(physics)
        cfg["gamma_mode"] = gamma_mode
        cfg["gamma_scale"] = f"{float(gamma_scale):g}"
        if poiseuille_scale is not None:
            cfg["poiseuille_scale"] = f"{float(poiseuille_scale):g}"
        try:
            yield cfg
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def predict_clot_phi_species_hard(
    data,
    time_index: int,
    bio_cfg: BiochemConfig,
    device: torch.device,
) -> torch.Tensor:
    """COMSOL step thresholds: Mat >= crit or fi >= crit."""
    t = int(time_index)
    y = data.y[t].to(device)
    mat_si = mat_si_for_gelation_from_log1p(y[:, 15], bio_cfg)
    fi_si = species_log1p_nd_to_si(y[:, 4:16], bio_cfg)[:, 8]
    phi_mat = (mat_si.reshape(-1) >= float(bio_cfg.viscosity_mat_crit)).to(dtype=torch.float32)
    phi_fi = (fi_si.reshape(-1) >= float(bio_cfg.viscosity_fi_crit)).to(dtype=torch.float32)
    return torch.maximum(phi_mat, phi_fi)


def predict_clot_phi_species_soft(
    data,
    time_index: int,
    bio_cfg: BiochemConfig,
    device: torch.device,
) -> torch.Tensor:
    """Soft Mat/FI triggers (differentiable proxy)."""
    t = int(time_index)
    y = data.y[t].to(device)
    mat_si = mat_si_for_gelation_from_log1p(y[:, 15], bio_cfg)
    fi_si = species_log1p_nd_to_si(y[:, 4:16], bio_cfg)[:, 8]
    return phi_clot_from_mat_fi(
        mat_si,
        fi_si,
        mat_crit=float(bio_cfg.viscosity_mat_crit),
        fi_crit=float(bio_cfg.viscosity_fi_crit),
        temp_mat=max(float(bio_cfg.viscosity_gnode_temp_mat) * float(bio_cfg.soft_step_T_scale), 1e-8),
        temp_fi=max(float(bio_cfg.viscosity_gnode_temp_fi) * float(bio_cfg.soft_step_T_scale), 1e-8),
        combine="max",
    ).reshape(-1).to(dtype=torch.float32)


def _timeline_f1(
    data,
    phys: PhysicsConfig,
    device: torch.device,
    times: list[int],
    phi_fn,
) -> dict[str, float]:
    f1s: list[float] = []
    precs: list[float] = []
    recs: list[float] = []
    f1s_growth: list[float] = []
    mask = torch.ones(int(data.num_nodes), dtype=torch.bool, device=device)
    for t in times:
        phi_gt = gt_clot_phi_at_time(data, t, phys, device)
        phi_pred = phi_fn(t)
        m = _clot_metrics(phi_pred.reshape(-1), phi_gt.reshape(-1), mask)
        f1s.append(float(m["clot_f1"]))
        precs.append(float(m["clot_prec"]))
        recs.append(float(m["clot_rec"]))
        if float(m["gt_pos_frac"]) > 1e-4:
            f1s_growth.append(float(m["clot_f1"]))
    return {
        "mean_f1": sum(f1s) / max(len(f1s), 1),
        "mean_f1_growth_times": sum(f1s_growth) / max(len(f1s_growth), 1),
        "mean_prec": sum(precs) / max(len(precs), 1),
        "mean_rec": sum(recs) / max(len(recs), 1),
        "f1_last": f1s[-1] if f1s else float("nan"),
        "prec_last": precs[-1] if precs else float("nan"),
        "rec_last": recs[-1] if recs else float("nan"),
    }


@dataclass
class PredictorSweepRow:
    predictor: str
    gamma_mode: str
    gamma_scale: float
    poiseuille_scale: float | None
    mean_f1: float
    mean_f1_growth_times: float
    f1_last: float
    mean_prec: float
    mean_rec: float
    bulk_mu_ratio_t0: float | None = None

    def to_dict(self) -> dict:
        return {
            "predictor": self.predictor,
            "gamma_mode": self.gamma_mode,
            "gamma_scale": self.gamma_scale,
            "poiseuille_scale": self.poiseuille_scale,
            "mean_f1": self.mean_f1,
            "mean_f1_growth_times": self.mean_f1_growth_times,
            "f1_last": self.f1_last,
            "mean_prec": self.mean_prec,
            "mean_rec": self.mean_rec,
            "bulk_mu_ratio_t0": self.bulk_mu_ratio_t0,
        }


@dataclass
class PredictorSweepReport:
    anchor: str
    times: list[int]
    rows: list[PredictorSweepRow] = field(default_factory=list)
    best: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "anchor": self.anchor,
            "times": self.times,
            "rows": [r.to_dict() for r in self.rows],
            "best": self.best,
        }


def sweep_t0_clot_predictors(
    data,
    *,
    anchor: str,
    times: list[int],
    phys: PhysicsConfig | None = None,
    bio: BiochemConfig | None = None,
    device: torch.device | None = None,
) -> PredictorSweepReport:
    """Sweep gamma proxies (no spf.sr sidecar) and species-direct clot predictors."""
    phys = phys or PhysicsConfig(phase="biochem")
    bio = bio or BiochemConfig(phase="biochem")
    dev = device or torch.device("cpu")
    n_steps = int(data.y.shape[0])
    times = sorted({max(0, min(int(t), n_steps - 1)) for t in times})

    gamma_modes = ("kinematic", "poiseuille", "max", "graph", "max_kinematic")
    poi_scales = (0.75, 0.85, 1.0)
    gamma_scales = (0.85, 1.0, 1.15)

    rows: list[PredictorSweepRow] = []

    # Species-only (no gamma)
    for name, fn in (
        ("species_hard", lambda t: predict_clot_phi_species_hard(data, t, bio, dev)),
        ("species_soft", lambda t: predict_clot_phi_species_soft(data, t, bio, dev)),
    ):
        stats = _timeline_f1(data, phys, dev, times, fn)
        rows.append(
            PredictorSweepRow(
                predictor=name,
                gamma_mode="n/a",
                gamma_scale=1.0,
                poiseuille_scale=None,
                mean_f1=stats["mean_f1"],
                mean_f1_growth_times=stats["mean_f1_growth_times"],
                f1_last=stats["f1_last"],
                mean_prec=stats["mean_prec"],
                mean_rec=stats["mean_rec"],
            )
        )

    # Mu-growth with gamma sweep
    for gmode in gamma_modes:
        for gscale in gamma_scales:
            poi_list: tuple[float | None, ...] = poi_scales if gmode in ("max", "poiseuille") else (None,)
            for poi in poi_list:
                with t0_gt_baseline_env(
                    gamma_mode=gmode,
                    gamma_scale=gscale,
                    poiseuille_scale=poi,
                ):
                    def _mu_phi(t: int, gm=gmode) -> torch.Tensor:
                        step = predict_mu_si_at_time(
                            data, t, phys, bio, dev, gamma_mode=gm
                        )
                        anchor_mu = gt_mu_anchor_cap_si(data, phys, dev)
                        return clot_phi_binary_from_mu_growth(
                            step.mu_pred_si, anchor_mu, phys
                        )

                    stats = _timeline_f1(data, phys, dev, times, _mu_phi)
                    bulk_ratio = None
                    if 0 in times:
                        step0 = predict_mu_si_at_time(
                            data, 0, phys, bio, dev, gamma_mode=gmode
                        )
                        bulk = step0.mu_gt_si < 0.012
                        if bool(bulk.any().item()):
                            bulk_ratio = float(
                                (
                                    step0.mu_gt_si[bulk]
                                    / step0.mu_pred_si[bulk].clamp(min=1e-8)
                                ).median().item()
                            )
                    rows.append(
                        PredictorSweepRow(
                            predictor="mu_growth",
                            gamma_mode=gmode,
                            gamma_scale=gscale,
                            poiseuille_scale=poi,
                            mean_f1=stats["mean_f1"],
                            mean_f1_growth_times=stats["mean_f1_growth_times"],
                            f1_last=stats["f1_last"],
                            mean_prec=stats["mean_prec"],
                            mean_rec=stats["mean_rec"],
                            bulk_mu_ratio_t0=bulk_ratio,
                        )
                    )

    # Nucleation-projected mu growth (best bulk gamma = kinematic scale 1.0 on patient007)
    for gmode in ("kinematic", "max"):
        with t0_gt_baseline_env(gamma_mode=gmode, gamma_scale=1.0, poiseuille_scale=0.85):
            traj = rollout_t0_clot_phi(
                data,
                phys,
                bio,
                dev,
                gamma_mode=gmode,
                nucleation=True,
                nucleation_hops=1,
            )

            def _nuc_phi(t: int, tr=traj) -> torch.Tensor:
                return tr[int(t)]["phi"]

            stats = _timeline_f1(data, phys, dev, times, _nuc_phi)
            rows.append(
                PredictorSweepRow(
                    predictor="mu_growth_nucleation",
                    gamma_mode=gmode,
                    gamma_scale=1.0,
                    poiseuille_scale=0.85,
                    mean_f1=stats["mean_f1"],
                    mean_f1_growth_times=stats["mean_f1_growth_times"],
                    f1_last=stats["f1_last"],
                    mean_prec=stats["mean_prec"],
                    mean_rec=stats["mean_rec"],
                )
            )

    best: dict[str, dict] = {}
    for pred in ("species_hard", "species_soft", "mu_growth", "mu_growth_nucleation"):
        subset = [r for r in rows if r.predictor == pred]
        if not subset:
            continue
        winner = max(subset, key=lambda r: (r.mean_f1_growth_times, r.f1_last))
        best[pred] = winner.to_dict()

    overall = max(rows, key=lambda r: (r.mean_f1_growth_times, r.f1_last))
    best["overall"] = overall.to_dict()
    return PredictorSweepReport(anchor=anchor, times=times, rows=rows, best=best)
