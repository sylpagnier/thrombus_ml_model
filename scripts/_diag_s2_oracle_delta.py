"""Oracle: patch s0 species on FN/FP nodes toward GT; measure clot F1."""
from __future__ import annotations

import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_nucleation_mask import resolve_nucleation_eligibility
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time, predict_clot_phi_at_time
from src.core_physics.t0_rung4_ladder import (
    FI_SLICE_IDX,
    MAT_SLICE_IDX,
    _build_s0_deploy_species,
    resting_species_log_nd,
)
from src.core_physics.t0_r4_s2_species import _s0_gate_from_species
from src.core_physics.t0_rung_config import RUNG2_GAMMA_MODE, t0_rung2_env
from src.training.train_clot_phi_simple import _clot_metrics
from src.utils.paths import get_project_root


@torch.no_grad()
def rollout_patched(data, phys, bio, device, *, mode: str) -> torch.Tensor:
    out = data.y.clone().to(device=device)
    commits_prev = None
    rest = resting_species_log_nd(data, device)
    for t in range(int(data.y.shape[0])):
        elig = resolve_nucleation_eligibility(
            data, t, device, phys, bio, commits_prev=commits_prev, growth_seed="pred",
        ).reshape(-1).bool()
        s0 = _build_s0_deploy_species(data, t, device, bio, elig=elig, commits_prev=commits_prev)
        sp = s0.clone()
        sp_gt = data.y[t, :, 4:16].to(device=device, dtype=torch.float32)
        phi_gt = gt_clot_phi_at_time(data, t, phys, device).reshape(-1)
        gt_clot = phi_gt >= 0.5
        gate = _s0_gate_from_species(s0, data, device, bio, elig)
        s0_hot = gate > 0.25
        fn = gt_clot & elig & ~s0_hot
        fp = s0_hot & elig & ~gt_clot
        if mode == "fn_gt":
            sp[fn] = sp_gt[fn]
        elif mode == "fp_rest":
            sp[fp, FI_SLICE_IDX] = rest[fp, FI_SLICE_IDX]
            sp[fp, MAT_SLICE_IDX] = rest[fp, MAT_SLICE_IDX]
        elif mode == "fn_fp":
            sp[fn] = sp_gt[fn]
            sp[fp, FI_SLICE_IDX] = rest[fp, FI_SLICE_IDX]
            sp[fp, MAT_SLICE_IDX] = rest[fp, MAT_SLICE_IDX]
        elif mode == "gt_clot_band":
            sp[gt_clot & elig] = sp_gt[gt_clot & elig]
        elif mode == "all_gt_elig":
            sp[elig] = sp_gt[elig]
        out[t, :, 4:16] = sp
        with t0_rung2_env():
            phi_raw, _ = predict_clot_phi_at_time(
                data, t, phys, bio, device,
                gamma_mode=RUNG2_GAMMA_MODE, flow_source="gt", pred_species_series=out,
            )
        commits_prev = (phi_raw.reshape(-1) >= 0.5).bool()
    return out


def f1_at_last(data, pred, phys, bio, device) -> float:
    t = int(data.y.shape[0]) - 1
    phi_gt = gt_clot_phi_at_time(data, t, phys, device)
    with t0_rung2_env():
        phi_p, _ = predict_clot_phi_at_time(
            data, t, phys, bio, device,
            gamma_mode=RUNG2_GAMMA_MODE, flow_source="gt", pred_species_series=pred,
        )
    m = _clot_metrics(
        phi_p.reshape(-1), phi_gt.reshape(-1),
        torch.ones(data.num_nodes, device=device, dtype=torch.bool),
    )
    return float(m["clot_f1"])


def main() -> int:
    device = torch.device("cuda")
    p = get_project_root() / "data/processed/graphs_biochem_anchors/patient007.pt"
    data = torch.load(p, map_location=device, weights_only=False)
    phys, bio = PhysicsConfig(phase="biochem"), BiochemConfig(phase="biochem")
    for mode in ("fn_gt", "fp_rest", "fn_fp", "gt_clot_band", "all_gt_elig"):
        pred = rollout_patched(data, phys, bio, device, mode=mode)
        f1 = f1_at_last(data, pred, phys, bio, device)
        print(f"{mode:16s} F1={f1:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
