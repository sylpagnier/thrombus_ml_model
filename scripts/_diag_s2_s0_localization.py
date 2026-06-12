"""Quick s0 vs GT species localization diagnostic."""
from __future__ import annotations

import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_nucleation_mask import resolve_nucleation_eligibility
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time, predict_clot_phi_at_time
from src.core_physics.t0_rung4_ladder import (
    FI_SLICE_IDX,
    MAT_SLICE_IDX,
    _build_s0_deploy_species,
    rollout_rung4_species_series,
)
from src.core_physics.t0_rung_config import RUNG2_GAMMA_MODE, t0_rung2_env
from src.training.train_clot_phi_simple import _clot_metrics
from src.utils.paths import get_project_root


def main() -> int:
    device = torch.device("cuda")
    p = get_project_root() / "data/processed/graphs_biochem_anchors/patient007.pt"
    data = torch.load(p, map_location=device, weights_only=False)
    phys, bio = PhysicsConfig(phase="biochem"), BiochemConfig(phase="biochem")
    t_last = int(data.y.shape[0]) - 1
    commits_prev = None
    for t in range(t_last):
        with t0_rung2_env():
            phi_t, _ = predict_clot_phi_at_time(
                data, t, phys, bio, device,
                gamma_mode=RUNG2_GAMMA_MODE, flow_source="gt", pred_species_series=data.y,
            )
        commits_prev = (phi_t.reshape(-1) >= 0.5).bool()
    elig = resolve_nucleation_eligibility(
        data, t_last, device, phys, bio, commits_prev=commits_prev, growth_seed="pred",
    ).reshape(-1).bool()
    phi_gt = gt_clot_phi_at_time(data, t_last, phys, device).reshape(-1)
    s0 = _build_s0_deploy_species(data, t_last, device, bio, elig=elig, commits_prev=commits_prev)
    sp_gt = data.y[t_last, :, 4:16]
    gt_clot = phi_gt >= 0.5
    fi_thr = float(sp_gt[:, FI_SLICE_IDX].quantile(0.92).item())
    mat_thr = float(sp_gt[:, MAT_SLICE_IDX].quantile(0.92).item())
    s0_hot = (s0[:, FI_SLICE_IDX] > fi_thr) | (s0[:, MAT_SLICE_IDX] > mat_thr)
    fn = gt_clot & elig & ~s0_hot
    fp = s0_hot & elig & ~gt_clot
    both = gt_clot & s0_hot & elig
    print(f"t_last={t_last} elig={int(elig.sum())} gt_clot={int(gt_clot.sum())}")
    print(f"FN (gt clot, s0 cold): {int(fn.sum())}")
    print(f"FP (s0 hot, not gt clot): {int(fp.sum())}")
    print(f"both hot: {int(both.sum())}")
    for name, m in [("FN", fn), ("FP", fp), ("both", both), ("elig", elig)]:
        if bool(m.any().item()):
            fi_err = (s0[m, FI_SLICE_IDX] - sp_gt[m, FI_SLICE_IDX]).abs().mean().item()
            mat_err = (s0[m, MAT_SLICE_IDX] - sp_gt[m, MAT_SLICE_IDX]).abs().mean().item()
            print(f"  {name} fi_mae={fi_err:.6f} mat_mae={mat_err:.6f}")

    pred = rollout_rung4_species_series(data, phys, bio, device, step="s0")
    for t in [0, 20, 40, t_last]:
        phi_gt_t = gt_clot_phi_at_time(data, t, phys, device)
        with t0_rung2_env():
            phi_p, _ = predict_clot_phi_at_time(
                data, t, phys, bio, device,
                gamma_mode=RUNG2_GAMMA_MODE, flow_source="gt", pred_species_series=pred,
            )
        m = _clot_metrics(
            phi_p.reshape(-1), phi_gt_t.reshape(-1),
            torch.ones(data.num_nodes, device=device, dtype=torch.bool),
        )
        print(f"t={t} s0 F1={m['clot_f1']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
