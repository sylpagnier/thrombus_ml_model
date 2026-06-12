"""T0 gelation diagnostics using COMSOL debug export (mu1, mu2, Mat, fi, spf.sr).

Factorizes where analytical mu/clot fails:
  A) Carreau formula + COMSOL mu1/mu2/sr  -> spf.mu  (formula check)
  B) Python mu1/mu2 from graph species    -> COMSOL mu1/mu2 (species mapping)
  C) Python mu from graph species+flow    -> spf.mu       (full T0)
  D) Nucleation mask ablation on clot phi

Usage::

    python scripts/build_comsol_debug_sidecar.py --anchor patient007
    python scripts/diagnose_t0_gelation_comsol.py --anchor patient007
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND  # noqa: E402
from src.core_physics.clot_growth_masks import gt_growth_commit_mask_at_time  # noqa: E402
from src.core_physics.t0_mu_physics import (  # noqa: E402
    clot_phi_binary_from_mu_growth,
    debug_sidecar_path,
    gt_clot_phi_at_time,
    gt_mu_anchor_cap_si,
    load_debug_sidecar,
    predict_mu_si_at_time,
    predict_mu_si_from_comsol_export_legs,
    predict_mu_si_from_graph_species_legs,
    rollout_t0_clot_phi,
    t0_physics_env,
)
from src.training.train_clot_phi_simple import _clot_metrics  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402


def _pearson(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if mask is not None:
        m = mask.reshape(-1).bool()
        if not bool(m.any().item()):
            return float("nan")
        a, b = a.reshape(-1)[m], b.reshape(-1)[m]
    a = a.float()
    b = b.float()
    ac = a - a.mean()
    bc = b - b.mean()
    den = ac.pow(2).sum().sqrt() * bc.pow(2).sum().sqrt()
    if float(den.item()) < 1e-12:
        return float("nan")
    return float((ac * bc).sum().item() / den.item())


def _median_ratio(gt: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if mask is not None:
        m = mask.reshape(-1).bool()
        if not bool(m.any().item()):
            return float("nan")
        gt, pred = gt[m], pred[m]
    return float((gt / pred.clamp(min=1e-8)).median().item())


def _f1(phi_pred: torch.Tensor, phi_gt: torch.Tensor) -> dict[str, float]:
    mask = torch.ones(int(phi_pred.numel()), dtype=torch.bool, device=phi_pred.device)
    return _clot_metrics(phi_pred.reshape(-1), phi_gt.reshape(-1), mask)


def diagnose_anchor(anchor: str, *, times: list[int] | None = None) -> dict:
    root = get_project_root()
    graph_path = root / "data/processed/graphs_biochem_anchors" / f"{anchor}.pt"
    debug = load_debug_sidecar(anchor, root=root)
    if debug is None:
        raise FileNotFoundError(
            f"Missing {debug_sidecar_path(anchor, root=root)}; run build_comsol_debug_sidecar.py"
        )

    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    device = torch.device("cpu")
    data = torch.load(graph_path, map_location=device, weights_only=False)
    n_steps = int(data.y.shape[0])
    if times is None:
        times = list(range(0, n_steps, max(1, n_steps // 10))) + [n_steps - 1]
    times = sorted({max(0, min(int(t), n_steps - 1)) for t in times})

    # Mat/FI graph vs COMSOL (fix broken reference in diagnose_time)
    from src.core_physics.clot_phi_simple import mat_si_for_gelation_from_log1p, species_log1p_nd_to_si

    per_time: list[dict] = []
    with t0_physics_env(anchor, gamma_mode="comsol_sr") as physics:
        gamma_mode = physics["gamma_mode"]
        for t in times:
            y = data.y[t].to(device)
            mu_gt = phys.viscosity_nd_to_si(y[:, STATE_CHANNEL_MU_EFF_ND]).reshape(-1)
            anchor_mu = gt_mu_anchor_cap_si(data, phys, device)
            growth = gt_growth_commit_mask_at_time(data, t, phys, device)
            bulk = anchor_mu < 0.012

            mu_comsol_legs = predict_mu_si_from_comsol_export_legs(data, t, phys, debug, device)
            mu_graph_full, py_mu1, py_mu2 = predict_mu_si_from_graph_species_legs(
                data, t, phys, bio, debug, device, mat_source="graph", fi_source="graph"
            )
            mu_comsol_mat_fi, _, _ = predict_mu_si_from_graph_species_legs(
                data, t, phys, bio, debug, device, mat_source="comsol", fi_source="comsol"
            )
            step = predict_mu_si_at_time(data, t, phys, bio, device, gamma_mode=gamma_mode)

            comsol_mu1 = debug["mu1"][t].reshape(-1).to(device)
            comsol_mu2 = debug["mu2"][t].reshape(-1).to(device)
            comsol_mat = debug["mat_si"][t].reshape(-1).to(device)
            comsol_fi = debug["fi_si"][t].reshape(-1).to(device)
            mat_graph = mat_si_for_gelation_from_log1p(y[:, 15], bio)
            sp_graph = species_log1p_nd_to_si(y[:, 4:16], bio)
            fi_graph = sp_graph[:, 8]

            phi_gt = gt_clot_phi_at_time(data, t, phys, device)
            phi_raw = clot_phi_binary_from_mu_growth(step.mu_pred_si, anchor_mu, phys)
            fp = (phi_raw > 0.5) & (phi_gt < 0.5)
            tp = (phi_raw > 0.5) & (phi_gt > 0.5)

            per_time.append(
                {
                    "time_index": t,
                    "time_s": float(debug["times_s"][t].item()),
                    "n_growth": int(growth.sum().item()),
                    "A_formula_comsol_legs": {
                        "pearson_all": _pearson(mu_gt, mu_comsol_legs),
                        "ratio_bulk": _median_ratio(mu_gt, mu_comsol_legs, bulk),
                        "ratio_growth": _median_ratio(mu_gt, mu_comsol_legs, growth),
                        "rel_l2_all": float(
                            (
                                (mu_gt - mu_comsol_legs).pow(2).sum().sqrt()
                                / mu_gt.pow(2).sum().sqrt().clamp(min=1e-12)
                            ).item()
                        ),
                    },
                    "B_species_to_legs": {
                        "pearson_mu1": _pearson(py_mu1, comsol_mu1),
                        "pearson_mu2": _pearson(py_mu2, comsol_mu2),
                        "pearson_mat": _pearson(mat_graph, comsol_mat),
                        "pearson_fi": _pearson(fi_graph, comsol_fi),
                        "frac_nodes_mu1_diff": float((py_mu1 != comsol_mu1).float().mean().item()),
                        "frac_nodes_mu2_diff": float((py_mu2 != comsol_mu2).float().mean().item()),
                    },
                    "C_graph_species_t0": {
                        "pearson_all": _pearson(mu_gt, step.mu_pred_si),
                        "pearson_growth": _pearson(step.mu_pred_si, mu_gt, growth),
                        "ratio_growth": _median_ratio(mu_gt, step.mu_pred_si, growth),
                        "f1_phi_raw": _f1(phi_raw, phi_gt),
                    },
                    "D_comsol_mat_fi_legs": {
                        "pearson_all": _pearson(mu_gt, mu_comsol_mat_fi),
                        "ratio_growth": _median_ratio(mu_gt, mu_comsol_mat_fi, growth),
                        "f1_phi": _f1(
                            clot_phi_binary_from_mu_growth(mu_comsol_mat_fi, anchor_mu, phys), phi_gt
                        ),
                    },
                    "E_false_positives": {
                        "n_fp": int(fp.sum().item()),
                        "n_tp": int(tp.sum().item()),
                        "fp_comsol_mu1_median": float(comsol_mu1[fp].median().item()) if fp.any() else float("nan"),
                        "fp_py_mu1_median": float(py_mu1[fp].median().item()) if fp.any() else float("nan"),
                        "fp_comsol_mu1_gt1_frac": float((comsol_mu1[fp] > 1.01).float().mean().item())
                        if fp.any()
                        else float("nan"),
                        "fp_py_mu1_gt1_frac": float((py_mu1[fp] > 1.01).float().mean().item()) if fp.any() else float("nan"),
                        "fp_comsol_mat_median": float(comsol_mat[fp].median().item()) if fp.any() else float("nan"),
                        "fp_comsol_fi_median": float(comsol_fi[fp].median().item()) if fp.any() else float("nan"),
                    },
                }
            )

        # Nucleation ablation at final time
        t_last = times[-1]
        phi_gt_last = gt_clot_phi_at_time(data, t_last, phys, device)
        ablation: dict[str, dict] = {}
        for label, nuc in (
            ("raw", False),
            ("nucleation_wall_hop1", True),
        ):
            traj = rollout_t0_clot_phi(
                data,
                phys,
                bio,
                device,
                gamma_mode=gamma_mode,
                nucleation=nuc,
                nucleation_hops=1,
                use_dgamma_wall_seed=False,
            )
            phi = traj[t_last]["phi"]
            ablation[label] = _f1(phi, phi_gt_last)
            ablation[f"{label}_pred_pos"] = float((phi > 0.5).float().mean().item())

        # Timeline F1 with nucleation
        traj_nuc = rollout_t0_clot_phi(
            data, phys, bio, device, gamma_mode=gamma_mode, nucleation=True, nucleation_hops=1
        )
        traj_raw = rollout_t0_clot_phi(
            data, phys, bio, device, gamma_mode=gamma_mode, nucleation=False
        )
        timeline = []
        for t in times:
            pg = gt_clot_phi_at_time(data, t, phys, device)
            timeline.append(
                {
                    "time_index": t,
                    "f1_raw": _f1(traj_raw[t]["phi"], pg),
                    "f1_nucleation": _f1(traj_nuc[t]["phi"], pg),
                }
            )

    # Summarize conclusions
    last = per_time[-1]
    a_ok = last["A_formula_comsol_legs"]["rel_l2_all"] < 0.05
    b_bad = last["B_species_to_legs"]["frac_nodes_mu1_diff"] > 0.05
    fp_comsol_mu1 = last["E_false_positives"].get("fp_comsol_mu1_gt1_frac", float("nan"))

    conclusions = []
    if a_ok:
        conclusions.append("Carreau+COMSOL mu1/mu2/sr reproduces spf.mu (formula OK).")
    else:
        conclusions.append("Carreau formula still mismatches even with COMSOL legs (check mph expr).")
    if b_bad:
        conclusions.append("Python mu1/mu2 from graph species disagree with COMSOL export legs.")
    if math.isfinite(fp_comsol_mu1) and fp_comsol_mu1 < 0.5:
        conclusions.append(
            "Most false positives have COMSOL mu1=1: Python mu1 step fires but COMSOL has not gelled."
        )
    elif math.isfinite(fp_comsol_mu1) and fp_comsol_mu1 > 0.5:
        conclusions.append("False positives often have COMSOL mu1>1: species/leg mapping may still be wrong.")

    return {
        "anchor": anchor,
        "physics": physics,
        "match_frac": float(debug.get("match_frac", float("nan"))),
        "times": per_time,
        "nucleation_ablation_t_last": ablation,
        "timeline_f1": timeline,
        "conclusions": conclusions,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="T0 gelation COMSOL debug diagnostics")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--out", default="outputs/biochem/clot_trigger/t0_gelation_comsol_diag.json")
    args = ap.parse_args()

    report = diagnose_anchor(args.anchor)
    out = get_project_root() / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] {args.anchor} -> {out}")
    for line in report["conclusions"]:
        print(f"[i] {line}")
    last = report["times"][-1]
    print(
        f"[i] t={last['time_index']} A rel_l2={last['A_formula_comsol_legs']['rel_l2_all']:.4f} "
        f"B mu1_diff_frac={last['B_species_to_legs']['frac_nodes_mu1_diff']:.3f} "
        f"C f1_raw={last['C_graph_species_t0']['f1_phi_raw']['clot_f1']:.3f} "
        f"D f1_comsol_mat_fi={last['D_comsol_mat_fi_legs']['f1_phi']['clot_f1']:.3f}"
    )
    ab = report["nucleation_ablation_t_last"]
    print(
        f"[i] nucleation ablation t_last: raw F1={ab['raw']['clot_f1']:.3f} "
        f"nuc F1={ab['nucleation_wall_hop1']['clot_f1']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
