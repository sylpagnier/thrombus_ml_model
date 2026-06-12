"""Diagnose GT mu vs Carreau: export expr, mu_b scale, shear-rate / gradient scaling.

Usage (repo root)::

    python scripts/diagnose_mu_export_shear.py
    python scripts/diagnose_mu_export_shear.py --anchor patient007 --time 35
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.clot_phi_simple import (  # noqa: E402
    carreau_mu_si_from_uv,
    comsol_carreau_mu_si_from_uv,
    gamma_dot_nd_kinematic_from_uv,
    resolve_gamma_dot_nd_for_carreau,
    species_log1p_nd_to_si,
)
from src.core_physics.kinematics_clot_prior import shear_rate_si  # noqa: E402
from src.core_physics.clot_phi_mu_inject import gamma_dot_nd_from_uv  # noqa: E402
from src.training.clot_ml_step0_coef import discover_anchor_paths  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402
from src.utils.rheology import carreau_yasuda_viscosity, compute_shear_rate  # noqa: E402


def _stats(x: torch.Tensor) -> dict[str, float]:
    x = x.reshape(-1).float()
    return {
        "median": float(x.median().item()),
        "mean": float(x.mean().item()),
        "p10": float(x.quantile(0.10).item()),
        "p90": float(x.quantile(0.90).item()),
        "max": float(x.max().item()),
    }


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    ac = a - a.mean()
    bc = b - b.mean()
    den = ac.pow(2).sum().sqrt() * bc.pow(2).sum().sqrt()
    if float(den.item()) < 1e-12:
        return float("nan")
    return float((ac * bc).sum().item() / den.item())


def _hard_m(mat: torch.Tensor, fi: torch.Tensor, bio: BiochemConfig, ratio_max: float = 80.0) -> torch.Tensor:
    mc = float(bio.viscosity_mat_crit)
    fc = float(bio.viscosity_fi_crit)
    mu1 = torch.where(mat >= mc, torch.tensor(ratio_max), torch.tensor(1.0))
    mu2 = torch.where(fi >= fc, torch.tensor(ratio_max), torch.tensor(0.0))
    return mu1 + mu2


def _carreau_si_from_gamma_nd(
    gamma_nd: torch.Tensor,
    data,
    phys: PhysicsConfig,
) -> torch.Tensor:
    """Carreau using canonical ND gamma (same convention as mesh prior)."""
    u_ref = float(data.u_ref.view(-1)[0].item())
    d_bar = float(data.d_bar.view(-1)[0].item())
    lam_nd = phys.lam * u_ref / d_bar
    mu_inf_nd = phys.mu_inf / phys.mu_viscosity_nd_scale
    mu_0_nd = phys.mu_0 / phys.mu_viscosity_nd_scale
    mu_nd = carreau_yasuda_viscosity(
        gamma_nd.reshape(-1, 1),
        torch.full_like(gamma_nd.reshape(-1, 1), mu_inf_nd),
        torch.full_like(gamma_nd.reshape(-1, 1), mu_0_nd),
        torch.full_like(gamma_nd.reshape(-1, 1), lam_nd),
        float(phys.n),
        float(phys.a),
    )
    return phys.viscosity_nd_to_si(mu_nd).reshape(-1)


def _poiseuille_gamma_nd_analytic(data, u_nd: torch.Tensor, v_nd: torch.Tensor) -> torch.Tensor:
    """Analytic ND shear from parabolic profile using graph sdf/width (diagnostic only)."""
    from src.core_physics.clot_phi_simple import sdf_nd_from_data  # noqa: E402
    from src.data_gen.lib.graph_velocity_priors import width_nd_to_radius_nd  # noqa: E402

    n = int(data.num_nodes)
    sdf = sdf_nd_from_data(data, u_nd.device, n).float()
    if hasattr(data, "width_nd") and data.width_nd is not None:
        width = data.width_nd.view(-1).float()
    else:
        width = 2.0 * sdf.clamp(min=1e-6)
    r_nd = width_nd_to_radius_nd(width.reshape(-1, 1)).reshape(-1).clamp(min=1e-6)
    speed_nd = torch.sqrt(u_nd.reshape(-1) ** 2 + v_nd.reshape(-1) ** 2).clamp(min=1e-8)
  # u_max ~ speed at centerline proxy: scale speed by inverse parabolic factor at sdf
    r_lane = (r_nd - sdf).clamp(min=0.0)
    parab = (1.0 - (r_lane / r_nd.clamp(min=1e-6)) ** 2).clamp(min=1e-6)
    u_max_nd = speed_nd / parab
    return torch.abs(-2.0 * u_max_nd * r_lane / (r_nd ** 2 + 1e-12))


def _read_export_header(path: Path) -> dict[str, object]:
    out: dict[str, object] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return out
    lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for _ in range(30):
            line = f.readline()
            if not line:
                break
            lines.append(line.rstrip())
    out["head_lines"] = lines[:15]
    joined = "\n".join(lines).lower()
    for token in ("spf.mu", "mu_b*(mu1", "mu_b * (mu1", "mu_effective", "mu_eff"):
        if token in joined:
            out.setdefault("hints", []).append(token)
    return out


def diagnose_anchor(
    anchor: str,
    times: list[int],
    *,
    root: Path,
) -> dict[str, object]:
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    graph_path = root / "data" / "processed" / "graphs_biochem_anchors" / f"{anchor}.pt"
    if not graph_path.is_file():
        paths = discover_anchor_paths(root=root)
        graph_path = next((p for p in paths if anchor in p.stem), None)
        if graph_path is None:
            raise FileNotFoundError(f"No graph for {anchor}")
    data = torch.load(graph_path, map_location="cpu", weights_only=False)

    export_txt = root / "data" / "processed" / "cfd_results_biochem" / f"{anchor}.txt"
    export_meta = _read_export_header(export_txt)

    props = {
        "u_ref": data.u_ref.view(-1)[:1],
        "d_bar": data.d_bar.view(-1)[:1],
    }
    u_ref = float(data.u_ref.view(-1)[0].item())
    d_bar = float(data.d_bar.view(-1)[0].item())

    wall = data.mask_wall.view(-1).bool() if hasattr(data, "mask_wall") else torch.zeros(int(data.num_nodes), dtype=torch.bool)
    interior = ~wall

    rows: list[dict[str, object]] = []
    for t in times:
        if t >= int(data.y.shape[0]):
            continue
        y = data.y[int(t)]
        u_nd = y[:, 0]
        v_nd = y[:, 1]
        mu_gt = phys.viscosity_nd_to_si(y[:, 3])
        sp = species_log1p_nd_to_si(y[:, 4:16], bio)
        mat = sp[:, 11]
        fi = sp[:, 8]
        m_hard = _hard_m(mat, fi, bio)
        bulk = m_hard == 1

        gamma_graph_nd = gamma_dot_nd_from_uv(data, u_nd, v_nd)
        gamma_si = shear_rate_si(data, u_nd, v_nd, props)
        gamma_poi_nd = _poiseuille_gamma_nd_analytic(data, u_nd, v_nd)

        mu_car_code = carreau_mu_si_from_uv(data, u_nd, v_nd, phys).reshape(-1)
        mu_car_ggraph = _carreau_si_from_gamma_nd(gamma_graph_nd, data, phys)
        mu_car_gpoi = _carreau_si_from_gamma_nd(gamma_poi_nd, data, phys)
        gamma_kin_nd = gamma_dot_nd_kinematic_from_uv(data, u_nd, v_nd, device=mu_gt.device)
        mu_car_kin = _carreau_si_from_gamma_nd(gamma_kin_nd, data, phys)
        m_one = torch.ones_like(m_hard)
        mu_comsol_max = comsol_carreau_mu_si_from_uv(
            data, u_nd, v_nd, m_one, phys, device=mu_gt.device, gamma_mode="max"
        ).reshape(-1)

        mu_b_candidates = {
            "0.0035_Pa_s_other_ai": 0.0035,
            "0.0084_Pa_s_implied": 0.0084,
            "0.035_Pa_s_code_default": 0.035,
        }
        gel_preds: dict[str, dict[str, float]] = {}
        for name, mu_b in mu_b_candidates.items():
            pred = mu_b * m_hard
            gel_preds[name] = {
                "all_ratio_gt_over_pred_median": float((mu_gt / pred.clamp(min=1e-8)).median().item()),
                "bulk_M1_ratio_median": float((mu_gt[bulk] / pred[bulk].clamp(min=1e-8)).median().item())
                if bulk.any()
                else float("nan"),
            }

        row: dict[str, object] = {
            "time": int(t),
            "mu_gt_si": _stats(mu_gt),
            "mu_carreau_code_path": _stats(mu_car_code),
            "mu_carreau_from_graph_gamma": _stats(mu_car_ggraph),
            "mu_carreau_from_poiseuille_gamma": _stats(mu_car_gpoi),
            "mu_carreau_from_kinematic_gamma": _stats(mu_car_kin),
            "mu_comsol_carreau_max_M1": _stats(mu_comsol_max),
            "gamma_graph_nd": _stats(gamma_graph_nd),
            "gamma_kinematic_nd": _stats(gamma_kin_nd),
            "gamma_resolved_nd": _stats(
                resolve_gamma_dot_nd_for_carreau(data, u_nd, v_nd, device=mu_gt.device)
            ),
            "gamma_si_from_graph": _stats(gamma_si),
            "gamma_poiseuille_nd": _stats(gamma_poi_nd),
            "u_ref_m_s": u_ref,
            "d_bar_m": d_bar,
            "frac_mu_car_at_mu0": float((mu_car_code > phys.mu_0 - 1e-4).float().mean().item()),
            "pearson_gt_vs_car_code": _pearson(mu_gt, mu_car_code),
            "pearson_gt_vs_car_graph_gamma": _pearson(mu_gt, mu_car_ggraph),
            "pearson_gt_vs_car_poiseuille": _pearson(mu_gt, mu_car_gpoi),
            "pearson_gt_vs_car_kinematic": _pearson(mu_gt, mu_car_kin),
            "pearson_gt_vs_comsol_carreau_max": _pearson(mu_gt, mu_comsol_max),
            "bulk_M1_comsol_carreau_ratio_median": float(
                (mu_gt[bulk] / mu_comsol_max[bulk].clamp(min=1e-8)).median().item()
            )
            if bulk.any()
            else float("nan"),
            "ratio_gt_over_car_median": float((mu_gt / mu_car_code.clamp(min=1e-8)).median().item()),
            "bulk_M1_fraction": float(bulk.float().mean().item()),
            "bulk_M1_mu_gt_median": float(mu_gt[bulk].median().item()) if bulk.any() else float("nan"),
            "gelation_mu_b_tests": gel_preds,
            "wall_gamma_si_median": float(gamma_si[wall].median().item()) if wall.any() else float("nan"),
            "interior_gamma_si_median": float(gamma_si[interior].median().item()) if interior.any() else float("nan"),
        }
        rows.append(row)

    return {
        "anchor": anchor,
        "graph": str(graph_path),
        "export_meta": export_meta,
        "physics_config_si": {
            "mu_inf": phys.mu_inf,
            "mu_0": phys.mu_0,
            "lam": phys.lam,
            "cgs_mu_to_pa_s": phys.cgs_mu_to_pa_s,
        },
        "times": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose mu export vs Carreau shear scaling")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--times", default="0,17,35,53")
    ap.add_argument("--out", default="outputs/biochem/diagnostics/mu_export_shear_diag.json")
    args = ap.parse_args()

    root = get_project_root()
    times = [int(x.strip()) for x in str(args.times).split(",") if x.strip()]
    report = diagnose_anchor(args.anchor, times, root=root)

    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[OK] wrote {out_path}")
    t0 = report["times"][0] if report["times"] else {}
    print(f"[i] {report['anchor']} export exists: {report['export_meta'].get('exists')}")
    if report["export_meta"].get("hints"):
        print(f"[i] export header hints: {report['export_meta']['hints']}")
    if t0:
        print(
            f"[i] t={t0['time']}: mu_gt med={t0['mu_gt_si']['median']:.5f} "
            f"mu_car med={t0['mu_carreau_code_path']['median']:.5f} "
            f"gamma_si med={t0['gamma_si_from_graph']['median']:.4f} wall={t0['wall_gamma_si_median']:.2f}"
        )
        for k, v in t0["gelation_mu_b_tests"].items():
            print(f"    {k}: bulk M=1 gt/pred={v['bulk_M1_ratio_median']:.3f}")


if __name__ == "__main__":
    main()
