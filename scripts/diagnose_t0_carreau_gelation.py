"""T0 oracle: compare GT mu_eff vs factorized Carreau x gelation (GT u,v + GT species).

Answers: does mu = M * mu_Carreau(gamma_dot) match COMSOL when M = mu1(Mat)+mu2(FI)?
Uses GT flow [u,v] only (not GT mu). Species from GT y[:,4:16].

Usage (repo root)::

    python scripts/diagnose_t0_carreau_gelation.py
    python scripts/diagnose_t0_carreau_gelation.py --anchor patient007 --times 0,17,35,53
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND  # noqa: E402
from src.core_physics.clot_growth_masks import gt_growth_commit_mask_at_time  # noqa: E402
from src.core_physics.clot_phi_simple import (  # noqa: E402
    cap_mu_eff_si,
    carreau_mu_si_from_uv,
    clot_phi_thresh_si,
    comsol_carreau_mu_si_from_uv,
    mu1_comsol_from_mat_si,
    mu1_gelation_from_mat_si,
    mu2_comsol_from_fi_si,
    mu2_gelation_from_fi_si,
    physics_mu_eff_si,
    species_log1p_nd_to_si,
)
from src.training.clot_ml_step0_coef import discover_anchor_paths  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    if a.numel() < 2:
        return float("nan")
    ac = a - a.mean()
    bc = b - b.mean()
    den = ac.pow(2).sum().sqrt() * bc.pow(2).sum().sqrt()
    if float(den.item()) < 1e-12:
        return float("nan")
    return float((ac * bc).sum().item() / den.item())


def _rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    d = (a.reshape(-1).float() - b.reshape(-1).float()).pow(2).mean().sqrt()
    return float(d.item())


def _log_mae(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        (torch.log(a.reshape(-1).clamp(min=1e-8)) - torch.log(b.reshape(-1).clamp(min=1e-8)))
        .abs()
        .mean()
        .item()
    )


def mu_mult_hard_comsol(
    fi_si: torch.Tensor,
    mat_si: torch.Tensor,
    bio: BiochemConfig,
    *,
    ratio_max: float = 80.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hard COMSOL steps: mu1 in {1,r}, mu2 in {0,r}, M = mu1+mu2."""
    mat_c = float(bio.viscosity_mat_crit)
    fi_c = float(bio.viscosity_fi_crit)
    r = float(ratio_max)
    mat = mat_si.reshape(-1)
    fi = fi_si.reshape(-1)
    mu1 = torch.where(mat >= mat_c, torch.full_like(mat, r), torch.ones_like(mat))
    mu2 = torch.where(fi >= fi_c, torch.full_like(fi, r), torch.zeros_like(fi))
    m = mu1 + mu2
    return mu1, mu2, m


def mu_models_at_step(
    data,
    t: int,
    *,
    device: torch.device,
    phys: PhysicsConfig,
    bio: BiochemConfig,
    ratio_max: float,
    mu_blood_si: float,
) -> dict[str, torch.Tensor]:
    y = data.y[int(t)].to(device=device, dtype=torch.float32)
    u = y[:, 0]
    v = y[:, 1]
    mu_gt = cap_mu_eff_si(phys.viscosity_nd_to_si(y[:, STATE_CHANNEL_MU_EFF_ND]))
    sp_log = y[:, 4:16]
    sp_si = species_log1p_nd_to_si(sp_log, bio)
    fi_si = sp_si[:, 8]
    mat_si = sp_si[:, 11]

    mu_c = carreau_mu_si_from_uv(data, u, v, phys)
    mu1_h, mu2_h, m_hard = mu_mult_hard_comsol(fi_si, mat_si, bio, ratio_max=ratio_max)

    # User factorized: M * baseline Carreau (M=1 -> standard Carreau at GT shear).
    mu_factorized = m_hard * mu_c
    mu_comsol_export = mu_blood_si * m_hard

    mu1_soft_c = mu1_comsol_from_mat_si(mat_si, bio, ratio_max)
    mu2_soft_c = mu2_comsol_from_fi_si(fi_si, bio, ratio_max)
    m_soft = mu1_soft_c + mu2_soft_c
    mu_factorized_soft = m_soft * mu_c
    mu_comsol_export_soft = mu_blood_si * m_soft

    mu1_g = mu1_gelation_from_mat_si(mat_si, bio, ratio_max)
    mu2_g = mu2_gelation_from_fi_si(fi_si, bio, ratio_max)
    gel_g = mu1_g + mu2_g
    mu_carreau_additive = mu_c * (1.0 + gel_g)

    mu_comsol_carreau_max = comsol_carreau_mu_si_from_uv(
        data, u, v, m_hard, phys, device=device, gamma_mode="max"
    )
    mu_comsol_carreau_graph = comsol_carreau_mu_si_from_uv(
        data, u, v, m_hard, phys, device=device, gamma_mode="graph"
    )
    mu_comsol_carreau_soft = comsol_carreau_mu_si_from_uv(
        data, u, v, mu1_soft_c + mu2_soft_c, phys, device=device, gamma_mode="max"
    )

    os.environ["CLOT_PHI_PHYSICS_MU_BASE"] = "comsol_carreau"
    os.environ["CLOT_PHI_PHYSICS_GAMMA_MODE"] = "max"
    mu_code_comsol_carreau = physics_mu_eff_si(
        mu_c, sp_log, bio, device=device, data=data, u_nd=u, v_nd=v, phys_cfg=phys, time_index=t
    )
    os.environ["CLOT_PHI_PHYSICS_MU_BASE"] = "carreau"
    mu_code_carreau = physics_mu_eff_si(
        mu_c, sp_log, bio, device=device, data=data, u_nd=u, v_nd=v, phys_cfg=phys, time_index=t
    )

    return {
        "mu_gt": mu_gt.reshape(-1),
        "mu_carreau": mu_c.reshape(-1),
        "M_hard": m_hard.reshape(-1),
        "mu_factorized_hard": mu_factorized.reshape(-1),
        "mu_comsol_export_hard": mu_comsol_export.reshape(-1),
        "mu_factorized_soft": mu_factorized_soft.reshape(-1),
        "mu_comsol_export_soft": mu_comsol_export_soft.reshape(-1),
        "mu_carreau_additive": mu_carreau_additive.reshape(-1),
        "mu_comsol_carreau_max": mu_comsol_carreau_max.reshape(-1),
        "mu_comsol_carreau_graph": mu_comsol_carreau_graph.reshape(-1),
        "mu_comsol_carreau_soft": mu_comsol_carreau_soft.reshape(-1),
        "mu_code_comsol_carreau": mu_code_comsol_carreau.reshape(-1),
        "mu_code_carreau": mu_code_carreau.reshape(-1),
        "fi_si": fi_si.reshape(-1),
        "mat_si": mat_si.reshape(-1),
    }


def _mask_stats(
    name: str,
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    m = mask.reshape(-1).bool()
    if not bool(m.any().item()):
        return {"name": name, "n": 0}
    p = pred[m]
    g = gt[m]
    return {
        "name": name,
        "n": int(m.sum().item()),
        "pearson_r": _pearson(p, g),
        "rmse": _rmse(p, g),
        "log_mae": _log_mae(p, g),
        "pred_mean": float(p.mean().item()),
        "gt_mean": float(g.mean().item()),
        "pred_p95": float(torch.quantile(p, 0.95).item()),
        "gt_p95": float(torch.quantile(g, 0.95).item()),
    }


def diagnose_anchor(
    graph_path: Path,
    *,
    device: torch.device,
    phys: PhysicsConfig,
    bio: BiochemConfig,
    times: list[int] | None,
    ratio_max: float,
    mu_blood_si: float,
) -> dict:
    data = torch.load(graph_path, map_location=device, weights_only=False)
    n_steps = int(data.y.shape[0])
    t_list = times if times is not None else [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1]
    t_list = sorted({max(0, min(int(t), n_steps - 1)) for t in t_list})

    model_keys = (
        "mu_comsol_carreau_max",
        "mu_comsol_carreau_soft",
        "mu_comsol_carreau_graph",
        "mu_code_comsol_carreau",
        "mu_factorized_hard",
        "mu_comsol_export_hard",
        "mu_factorized_soft",
        "mu_comsol_export_soft",
        "mu_carreau_additive",
        "mu_carreau",
        "mu_code_carreau",
    )

    per_t: list[dict] = []
    thr = clot_phi_thresh_si(phys)

    for t in t_list:
        bundle = mu_models_at_step(
            data, t, device=device, phys=phys, bio=bio, ratio_max=ratio_max, mu_blood_si=mu_blood_si
        )
        mu_gt = bundle["mu_gt"]
        growth_clot = gt_growth_commit_mask_at_time(data, t, phys, device)
        abs_clot = mu_gt >= thr
        bulk = mu_gt < thr

        row: dict = {"t": int(t), "models": {}, "M_hard_frac_gt1": float((bundle["M_hard"] > 1.01).float().mean().item())}
        for key in model_keys:
            stats = {
                "all": _mask_stats(key, bundle[key], mu_gt, torch.ones_like(mu_gt, dtype=torch.bool)),
                "growth_clot": _mask_stats(key, bundle[key], mu_gt, growth_clot),
                "abs_clot": _mask_stats(key, bundle[key], mu_gt, abs_clot),
                "bulk": _mask_stats(key, bundle[key], mu_gt, bulk),
            }
            row["models"][key] = stats
        per_t.append(row)

    # Best model by mean growth-clot pearson across times
    scores: dict[str, list[float]] = {k: [] for k in model_keys}
    for row in per_t:
        for key in model_keys:
            r = row["models"][key]["growth_clot"].get("pearson_r", float("nan"))
            if r == r:
                scores[key].append(r)

    ranking = sorted(
        ((k, sum(v) / len(v) if v else float("nan")) for k, v in scores.items()),
        key=lambda x: (-(x[1] if x[1] == x[1] else -1e9)),
    )

    return {
        "anchor": graph_path.stem,
        "n_steps": n_steps,
        "ratio_max": ratio_max,
        "mu_blood_si": mu_blood_si,
        "phys_mu_0_si": float(phys.mu_0),
        "phys_mu_inf_si": float(phys.mu_inf),
        "fi_crit": float(bio.viscosity_fi_crit),
        "mat_crit": float(bio.viscosity_mat_crit),
        "ranking_mean_r_growth_clot": [{"model": k, "mean_r": r} for k, r in ranking],
        "per_t": per_t,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="T0 Carreau x gelation oracle diagnostic")
    ap.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--anchor", default="", help="Single anchor stem (default: all)")
    ap.add_argument("--times", default="", help="Comma-separated time indices")
    ap.add_argument("--ratio-max", type=float, default=4.0)
    ap.add_argument("--mu-blood-si", type=float, default=0.0035)
    ap.add_argument("--out", default="outputs/biochem/clot_trigger/t0_carreau_gelation_diag.json")
    args = ap.parse_args()

    os.environ["CLOT_PHI_PHYSICS_MU_BASE"] = "carreau"
    os.environ["CLOT_PHI_PHYSICS_MU_RATIO_MAX"] = str(args.ratio_max)
    os.environ["CLOT_PHI_PHYSICS_HARD_STEP"] = "0"
    os.environ["CLOT_PHI_PHYSICS_GELATION_GATE"] = "0"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    root = get_project_root()

    anchor_dir = Path(args.anchor_dir)
    if not anchor_dir.is_absolute():
        anchor_dir = root / anchor_dir
    paths = discover_anchor_paths(anchor_dir)
    if args.anchor.strip():
        paths = [p for p in paths if p.stem == args.anchor.strip()]
    if not paths:
        print(f"[ERR] no graphs in {anchor_dir}", file=sys.stderr)
        return 2

    times = None
    if args.times.strip():
        times = [int(x.strip()) for x in args.times.split(",") if x.strip()]

    rows = [
        diagnose_anchor(
            p,
            device=device,
            phys=phys,
            bio=bio,
            times=times,
            ratio_max=float(args.ratio_max),
            mu_blood_si=float(args.mu_blood_si),
        )
        for p in paths
    ]

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"anchors": rows}, indent=2), encoding="utf-8")

    print(f"[i] Carreau gelation diagnostic (GT u,v + GT species; ratio_max={args.ratio_max})", flush=True)
    print(f"[i] mu_blood={args.mu_blood_si} phys mu_0={phys.mu_0} mu_inf={phys.mu_inf}", flush=True)
    for row in rows:
        rank = row["ranking_mean_r_growth_clot"]
        best = rank[0] if rank else {"model": "?", "mean_r": float("nan")}
        print(f"[OK] {row['anchor']}: best={best['model']} mean_r(growth_clot)={best['mean_r']:.3f}", flush=True)
        for leg in rank[:4]:
            print(f"     {leg['model']:24s} r={leg['mean_r']:.3f}", flush=True)

    print(f"[save] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
