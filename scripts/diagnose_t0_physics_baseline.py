"""T0 baseline physics diagnostic: mu, gamma, gelation, phi vs COMSOL GT.

Usage::

    python scripts/diagnose_t0_physics_baseline.py --anchor patient007
    python scripts/diagnose_t0_physics_baseline.py --anchor patient007 --times 0,35,53
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
    build_clot_phi_step,
    cap_mu_eff_si,
    clot_phi_physics_gamma_mode,
    clot_phi_physics_mu_base_mode,
    gamma_dot_nd_graph_from_uv,
    gamma_dot_nd_kinematic_from_uv,
    gamma_dot_nd_poiseuille_from_uv,
    physics_mu_eff_si,
    resolve_gamma_dot_nd_for_carreau,
    species_log1p_nd_to_si,
    mu1_comsol_from_mat_si,
    mu2_comsol_from_fi_si,
)
from src.core_physics.clot_trigger_rollout import rollout_clot_trigger_physics  # noqa: E402
from src.core_physics.neighbor_band_trigger import apply_physics_trigger_baseline_env  # noqa: E402
from src.core_physics.clot_growth_masks import gt_growth_commit_mask_at_time  # noqa: E402
from src.training.clot_trigger_stack import apply_clot_trigger_honest_env  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402


def _stats(x: torch.Tensor) -> dict[str, float]:
    x = x.reshape(-1).float()
    return {
        "median": float(x.median().item()),
        "mean": float(x.mean().item()),
        "p10": float(x.quantile(0.10).item()),
        "p90": float(x.quantile(0.90).item()),
        "max": float(x.max().item()),
    }


def _carreau_from_gamma_si(
    gamma_si: torch.Tensor,
    phys: PhysicsConfig,
    data,
    *,
    gel_factor: torch.Tensor,
) -> torch.Tensor:
    from src.core_physics.clot_phi_simple import (
        carreau_mu_si_from_gamma_nd,
        clot_phi_physics_mu_blood_si,
    )

    u_ref = float(data.u_ref.view(-1)[0].item())
    d_bar = float(data.d_bar.view(-1)[0].item())
    gamma_nd = gamma_si.reshape(-1) * (d_bar / max(u_ref, 1e-8))
    gf = gel_factor.reshape(-1).float().clamp(min=1e-8)
    mu_0 = float(phys.mu_0) * gf
    mu_inf = clot_phi_physics_mu_blood_si(phys) * gf
    return carreau_mu_si_from_gamma_nd(gamma_nd, mu_0, mu_inf, phys, data=data)


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    ac = a - a.mean()
    bc = b - b.mean()
    den = ac.pow(2).sum().sqrt() * bc.pow(2).sum().sqrt()
    if float(den.item()) < 1e-12:
        return float("nan")
    return float((ac * bc).sum().item() / den.item())


def _load_gamma_sidecar_si(
    anchor: str,
    root: Path,
    n_nodes: int,
    time_index: int = 0,
) -> torch.Tensor | None:
    """Optional COMSOL ``spf.sr`` sidecar matched to graph node order (units 1/s)."""
    for rel in (
        f"data/processed/cfd_results_biochem_diag/{anchor}_sr.pt",
        f"data/processed/cfd_results_biochem_diag/{anchor}_gammat.pt",
        f"outputs/biochem/diagnostics/{anchor}_gammat_si.pt",
    ):
        p = root / rel
        if not p.is_file():
            continue
        obj = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(obj, dict) and "gamma_si" in obj:
            g = obj["gamma_si"].float()
        elif torch.is_tensor(obj):
            g = obj.float()
        else:
            continue
        if g.dim() == 2 and int(g.shape[1]) == int(n_nodes):
            ti = max(0, min(int(time_index), int(g.shape[0]) - 1))
            return g[ti].reshape(-1)
        if g.dim() == 1 and int(g.numel()) == int(n_nodes):
            return g.reshape(-1)
    return None


def diagnose_anchor(
    anchor: str,
    times: list[int],
    *,
    root: Path,
    ratio_max: float,
) -> dict[str, object]:
    apply_clot_trigger_honest_env()
    apply_physics_trigger_baseline_env()

    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    device = torch.device("cpu")
    graph_path = root / "data" / "processed" / "graphs_biochem_anchors" / f"{anchor}.pt"
    if not graph_path.is_file():
        raise FileNotFoundError(f"Missing graph: {graph_path}")

    data = torch.load(graph_path, map_location=device, weights_only=False)
    n_nodes = int(data.num_nodes)
    wall = data.mask_wall.view(-1).bool() if hasattr(data, "mask_wall") else torch.zeros(n_nodes, dtype=torch.bool)
    interior = ~wall
    gamma_sidecar_all = None
    sidecar_path = root / f"data/processed/cfd_results_biochem_diag/{anchor}_sr.pt"
    if sidecar_path.is_file():
        obj = torch.load(sidecar_path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict) and torch.is_tensor(obj.get("gamma_si")):
            gamma_sidecar_all = obj["gamma_si"].float()

    traj = rollout_clot_trigger_physics(data, phys_cfg=phys, bio_cfg=bio, device=device, time_stride=1)
    step0 = build_clot_phi_step(data, 0, phys, bio, device)
    mu_anchor = cap_mu_eff_si(
        physics_mu_eff_si(
            step0.mu_c_si,
            step0.species_log_gt,
            bio,
            device=device,
            data=data,
            u_nd=step0.u_flow_nd,
            v_nd=step0.v_flow_nd,
            phys_cfg=phys,
            time_index=0,
        )
    ).reshape(-1)

    rows: list[dict[str, object]] = []
    for t in times:
        if t >= int(data.y.shape[0]):
            continue
        y = data.y[int(t)]
        u_nd, v_nd = y[:, 0], y[:, 1]
        mu_gt = phys.viscosity_nd_to_si(y[:, 3])
        sp = species_log1p_nd_to_si(y[:, 4:16], bio)
        mat, fi = sp[:, 11], sp[:, 8]
        gel_hard = mu1_comsol_from_mat_si(mat, bio, ratio_max) + mu2_comsol_from_fi_si(fi, bio, ratio_max)
        gel_soft = gel_hard  # same hard step unless env toggled

        step = build_clot_phi_step(data, int(t), phys, bio, device)
        mu_phys = cap_mu_eff_si(
            physics_mu_eff_si(
                step.mu_c_si,
                step.species_log_gt,
                bio,
                device=device,
                data=data,
                u_nd=step.u_flow_nd,
                v_nd=step.v_flow_nd,
                phys_cfg=phys,
                time_index=int(t),
            )
        ).reshape(-1)

        g_graph = gamma_dot_nd_graph_from_uv(data, u_nd, v_nd)
        g_kin = gamma_dot_nd_kinematic_from_uv(data, u_nd, v_nd, device=device)
        g_poi = gamma_dot_nd_poiseuille_from_uv(data, u_nd, v_nd, device=device)
        g_res = resolve_gamma_dot_nd_for_carreau(data, u_nd, v_nd, device=device)
        u_ref = float(data.u_ref.view(-1)[0].item())
        d_bar = float(data.d_bar.view(-1)[0].item())
        g_graph_si = g_graph * (u_ref / max(d_bar, 1e-8))

        bulk = mu_gt < 0.012
        growth = gt_growth_commit_mask_at_time(data, int(t), phys, device)
        phi_gt = step.phi_gt.reshape(-1)
        phi_deploy = traj[int(t)]["phi"].reshape(-1)
        dmu_phys = (mu_phys - mu_anchor).clamp(min=0.0)
        dmu_gt = (mu_gt - phys.viscosity_nd_to_si(data.y[0, :, 3])).clamp(min=0.0)

        row: dict[str, object] = {
            "time": int(t),
            "mu_gt_si": _stats(mu_gt),
            "mu_phys_si": _stats(mu_phys),
            "bulk_gt_over_phys_median": float((mu_gt[bulk] / mu_phys[bulk].clamp(min=1e-8)).median().item())
            if bulk.any()
            else float("nan"),
            "pearson_all_mu": _pearson(mu_phys, mu_gt),
            "pearson_wall_mu": _pearson(mu_phys[wall], mu_gt[wall]) if wall.any() else float("nan"),
            "pearson_growth_mu": _pearson(mu_phys[growth], mu_gt[growth]) if growth.sum() > 10 else float("nan"),
            "gamma_nd_graph": _stats(g_graph),
            "gamma_nd_kinematic": _stats(g_kin),
            "gamma_nd_poiseuille": _stats(g_poi),
            "gamma_nd_resolved": _stats(g_res),
            "gamma_si_graph_bulk_med": float(g_graph_si[interior].median().item()) if interior.any() else float("nan"),
            "gel_hard": _stats(gel_hard),
            "dmu_phys": _stats(dmu_phys),
            "dmu_gt": _stats(dmu_gt),
            "phi_gt_pos_frac": float((phi_gt > 0.5).float().mean().item()),
            "phi_deploy_pos_frac": float((phi_deploy > 0.5).float().mean().item()),
            "phi_deploy_on_growth_recall": float(
                ((phi_deploy[growth] > 0.5).float().sum() / growth.float().sum().clamp(min=1.0)).item()
            )
            if growth.any()
            else float("nan"),
        }
        gamma_gt_si = None
        if gamma_sidecar_all is not None:
            ti = max(0, min(int(t), int(gamma_sidecar_all.shape[0]) - 1))
            gamma_gt_si = gamma_sidecar_all[ti].reshape(-1)
        if gamma_gt_si is not None:
            g_res_si = g_res * (u_ref / max(d_bar, 1e-8))
            g_kin_si = g_kin * (u_ref / max(d_bar, 1e-8))
            g_graph_si = g_graph * (u_ref / max(d_bar, 1e-8))
            row["gamma_comsol_si"] = _stats(gamma_gt_si)
            row["pearson_gamma_resolved_vs_comsol"] = _pearson(g_res_si, gamma_gt_si)
            row["pearson_gamma_kinematic_vs_comsol"] = _pearson(g_kin_si, gamma_gt_si)
            row["pearson_gamma_graph_vs_comsol"] = _pearson(g_graph_si, gamma_gt_si)
            row["gamma_scale_kin_median"] = float(
                (gamma_gt_si[interior] / g_kin_si[interior].clamp(min=1e-8)).median().item()
            ) if interior.any() else float("nan")
            row["gamma_scale_resolved_median"] = float(
                (gamma_gt_si / g_res_si.clamp(min=1e-8)).median().item()
            )
            mu_from_comsol_sr = _carreau_from_gamma_si(
                gamma_gt_si, phys, data, gel_factor=torch.ones_like(gamma_gt_si)
            )
            row["mu_carreau_from_comsol_sr"] = _stats(mu_from_comsol_sr)
            row["bulk_gt_over_mu_carreau_sr_median"] = float(
                (mu_gt[bulk] / mu_from_comsol_sr[bulk].clamp(min=1e-8)).median().item()
            ) if bulk.any() else float("nan")
        rows.append(row)

    bulk_ok = all(
        0.85 <= float(r.get("bulk_gt_over_phys_median", float("nan"))) <= 1.15
        for r in rows
        if r.get("time") == 0
    )
    gamma_sidecar = gamma_sidecar_all is not None
    return {
        "anchor": anchor,
        "graph": str(graph_path),
        "physics_env": {
            "mu_base": clot_phi_physics_mu_base_mode(),
            "gamma_mode": clot_phi_physics_gamma_mode(),
            "ratio_max": float(ratio_max),
        },
        "gamma_comsol_sidecar": gamma_sidecar,
        "checks": {
            "bulk_mu_t0_in_band": bulk_ok,
            "gamma_sidecar_present": gamma_sidecar,
        },
        "times": rows,
        "notes": (
            "bulk_gt_over_phys_median near 1.0 at t=0 => COMSOL spf.mu baseline OK; "
            "optional sidecar data/processed/cfd_results_biochem_diag/{anchor}_gammat.pt "
            "with dict {gamma_si: [N]} validates shear proxy."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="T0 baseline physics diagnostic")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--times", default="0,17,35,53")
    ap.add_argument("--ratio-max", type=float, default=4.0)
    ap.add_argument("--out", default="outputs/biochem/clot_trigger/t0_physics_baseline_diag.json")
    args = ap.parse_args()

    root = get_project_root()
    times = [int(x.strip()) for x in str(args.times).split(",") if x.strip()]
    report = diagnose_anchor(args.anchor, times, root=root, ratio_max=float(args.ratio_max))

    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[OK] {args.anchor} -> {out_path}")
    t0 = next((r for r in report["times"] if r["time"] == 0), report["times"][0] if report["times"] else {})
    if t0:
        print(
            f"[i] t=0 bulk GT/pred={t0['bulk_gt_over_phys_median']:.3f} "
            f"gamma_kin_nd_med={t0['gamma_nd_kinematic']['median']:.3f} "
            f"gamma_graph_nd_med={t0['gamma_nd_graph']['median']:.5f}"
        )
    print(f"[i] checks: {report['checks']}")
    if not report["gamma_comsol_sidecar"]:
        print(
            "[i] No COMSOL gamma sidecar (optional). See docs/COMSOL_MU_RHEOLOGY_CHECKLIST.md "
            "section 'Optional gamma validation export'."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
