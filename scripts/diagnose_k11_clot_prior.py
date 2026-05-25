"""Compare K11 clot triggers / kin prior vs COMSOL high-μ on an anchor graph.

Usage:
  python scripts/diagnose_k11_clot_prior.py --anchor patient007
  python scripts/diagnose_k11_clot_prior.py --anchor patient007 --checkpoint outputs/biochem/biochem_teacher_last.pth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.architecture.gnode_biochem import (  # noqa: E402
    biochem_truth_node_mask,
    k11_bio_trigger_score,
    k11_clot_apply_mask,
    k11_clot_region_mask,
    k11_clot_trigger_score,
    k11_mech_trigger_score,
)
from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND  # noqa: E402
from src.core_physics.clot_kinematics_fields import (  # noqa: E402
    compute_clot_kinematics_fields,
    score_clot_risk_from_fields,
)
from src.training.train_biochem_corrector import _k11_clot_gt_label  # noqa: E402


def _load_anchor(stem: str, device: torch.device):
    path = REPO / "data" / "processed" / "graphs_biochem_anchors" / f"{stem}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.load(path, map_location=device, weights_only=False)


def _time_index(data, t_query: float | None) -> int:
    if not hasattr(data, "t") or data.t is None:
        return -1
    t = data.t.detach().cpu().numpy().astype(np.float64).reshape(-1)
    if t_query is None:
        return int(np.argmax(t))
    return int(np.argmin(np.abs(t - float(t_query))))


def _graph_props(data, device, dtype):
    if isinstance(data.u_ref, torch.Tensor) and data.u_ref.numel() == data.num_nodes:
        u_ref = data.u_ref.to(device=device, dtype=dtype).reshape(-1)[:1]
        d_bar = data.d_bar.to(device=device, dtype=dtype).reshape(-1)[:1]
    else:
        u_ref = torch.as_tensor(data.u_ref, device=device, dtype=dtype).reshape(1)
        d_bar = torch.as_tensor(data.d_bar, device=device, dtype=dtype).reshape(1)
    return {"u_ref": u_ref, "d_bar": d_bar}


def _overlap_report(name: str, mask_a: np.ndarray, mask_b: np.ndarray, n: int) -> None:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = int((a & b).sum())
    na = int(a.sum())
    nb = int(b.sum())
    prec = inter / max(nb, 1)
    rec = inter / max(na, 1)
    jacc = inter / max(int((a | b).sum()), 1)
    print(
        f"  {name}: |A|={na} |B|={nb} |∩|={inter} "
        f"precision(B|A)={prec:.3f} recall(A|B)={rec:.3f} jaccard={jacc:.4f} "
        f"(frac domain={na/n:.4f})"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--anchor", default="patient007")
    p.add_argument("--t-final", type=float, default=7950.0, help="COMSOL time [s] for GT slice")
    p.add_argument("--checkpoint", default="", help="Optional teacher ckpt for rollout p_clot")
    p.add_argument("--thresh-prior", type=float, default=0.25)
    p.add_argument("--thresh-clot", type=float, default=-1.0, help="<0: use GT label rule")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = _load_anchor(args.anchor, device)
    bio_cfg = BiochemConfig(phase="biochem")
    phys = PhysicsConfig()

    if not hasattr(data, "y") or data.y is None:
        raise RuntimeError("Anchor graph has no data.y trajectory")

    ti = _time_index(data, args.t_final)
    y = data.y[ti].to(device)
    truth = (
        biochem_truth_node_mask(data, int(data.num_nodes), device)
        .view(-1)
        .bool()
        .cpu()
        .numpy()
    )
    wall = (
        data.mask_wall.view(-1).bool().cpu().numpy()
        if hasattr(data, "mask_wall") and data.mask_wall is not None
        else np.zeros(data.num_nodes, dtype=bool)
    )
    sdf = data.x[:, 2]
    mu_g_si = (
        phys.viscosity_nd_to_si(y[:, STATE_CHANNEL_MU_EFF_ND])
        .reshape(-1)
        .detach()
        .cpu()
        .numpy()
    )
    y_clot_k11 = _k11_clot_gt_label(
        torch.from_numpy(mu_g_si), phys
    ).cpu().numpy().astype(bool)
    # Training high-μ tail uses p90 on truth nodes (matches val log lines).
    mu_cut = float(np.quantile(mu_g_si[truth], 0.90))
    y_clot_hi = mu_g_si >= mu_cut
    y_clot = y_clot_hi  # localized COMSOL clots for overlap stats
    n = int(data.num_nodes)
    ma = truth

    species_log = y[:, 4:16]
    u_nd = y[:, 0]
    v_nd = y[:, 1]
    props = _graph_props(data, device, torch.float32)

    class _Stub:
        fi_crit = 0.6
        mat_crit = 2e7

        def species_log_nd_to_si(self, species_log_t):
            from src.architecture.gnode_biochem import GNODE_Phase3

            # scales match training model; load real model if checkpoint given
            if getattr(self, "_model", None) is not None:
                return self._model.species_log_nd_to_si(species_log_t)
            return torch.expm1(species_log_t.clamp(-10, 8))

    stub = _Stub()
    model = None
    if args.checkpoint:
        from src.architecture.gnode_biochem import GNODE_Phase3
        from src.evaluation.visualize_pipeline import (
            _checkpoint_state_dict,
            _inject_biochem_kinematic_lora,
            _load_torch_checkpoint,
            resolve_gnode_phase3_ctor_kwargs,
        )

        ckpt_path = Path(args.checkpoint)
        biochem_meta, biochem_state = _checkpoint_state_dict(_load_torch_checkpoint(ckpt_path))
        ctor = resolve_gnode_phase3_ctor_kwargs(
            biochem_meta,
            biochem_state,
            bio_encoder_prior_dim_default=2,
            latent_dim_default=256,
        )
        model = GNODE_Phase3(
            phys_cfg=phys,
            in_channels=int(ctor["in_channels"]),
            spatial_channels=int(ctor["spatial_channels"]),
            latent_dim=int(ctor["latent_dim"]),
            max_inner_iters=int(ctor["max_inner_iters"]),
            bio_encoder_prior_dim=int(ctor["bio_encoder_prior_dim"]),
            mu_ratio_max=bio_cfg.mu_ratio_max,
            mat_crit=bio_cfg.viscosity_mat_crit,
            fi_crit=bio_cfg.viscosity_fi_crit,
            temp_mat=bio_cfg.viscosity_gnode_temp_mat,
            temp_fi=bio_cfg.viscosity_gnode_temp_fi,
            num_fourier_freqs=int(ctor["num_fourier_freqs"]),
            use_siren_decoder=bool(ctor["use_siren_decoder"]),
            gnode_layers=int(ctor["gnode_layers"]),
            use_hard_bcs=bool(ctor["use_hard_bcs"]),
        ).to(device)
        _inject_biochem_kinematic_lora(model)
        _, state = _checkpoint_state_dict(_load_torch_checkpoint(ckpt_path))
        model.load_state_dict(state, strict=False)
        model.eval()
        stub._model = model
        stub.fi_crit = float(model.fi_crit)
        stub.mat_crit = float(model.mat_crit)

    with torch.no_grad():
        bio = k11_bio_trigger_score(stub, species_log).reshape(-1).cpu().numpy()
        mech = k11_mech_trigger_score(data, u_nd, v_nd, bio_cfg, props).reshape(-1).cpu().numpy()
        comb = k11_clot_trigger_score(stub, species_log, u_nd, v_nd, data, bio_cfg, props).reshape(-1).cpu().numpy()
        kin_fields = compute_clot_kinematics_fields(data, u_nd, v_nd, bio_cfg, props)
        prior_comsol, _, _ = score_clot_risk_from_fields(kin_fields, bio_cfg)
        dgamma_dx = kin_fields.dgamma_dx_phys.reshape(-1).detach().cpu().numpy()
        flux_dx = kin_fields.flux_path_dx.reshape(-1).detach().cpu().numpy()
        prior_comsol_np = prior_comsol.reshape(-1).cpu().numpy()
        adj = (
            k11_clot_apply_mask(sdf, data.mask_wall)
            .reshape(-1)
            .cpu()
            .numpy()
        )
        band = (
            k11_clot_region_mask(sdf, data.mask_wall)
            .reshape(-1)
            .cpu()
            .numpy()
        )

    t_comsol = float(data.t[ti].item()) if hasattr(data, "t") else float("nan")
    print(f"Anchor {args.anchor} | COMSOL index {ti} | t≈{t_comsol:.1f}s | truth nodes {ma.sum()}/{n}")

    print("\n=== GT clot prevalence ===")
    print(
        f"  K11 label rule (μ≥0.055 OR μ≥1.2×μ_inf): "
        f"{int(y_clot_k11[ma].sum())} / {int(ma.sum())} "
        f"(often degenerate — ratio×μ_inf≈0.004 marks almost all nodes)"
    )
    print(f"  high-μ tail (p90 cutoff={mu_cut:.4f} Pa·s): {int(y_clot[ma].sum())} / {int(ma.sum())}")
    print(f"  clot ∩ wall:      {int((y_clot & wall)[ma].sum())}")
    print(f"  clot ∩ adjacent:  {int((y_clot & (adj > 0.05))[ma].sum())}")
    print(f"  μ_gt p90 on truth: {np.quantile(mu_g_si[ma], 0.9):.4f} Pa·s")

    thr_p = float(args.thresh_prior)
    prior_hi = comb >= thr_p
    bio_hi = bio >= thr_p
    mech_hi = mech >= thr_p

    print(f"\n=== COMSOL kinematic fields (GT u,v at t≈{t_comsol:.1f}s) ===")
    if ma.any():
        clot_ma = y_clot & ma
        non_ma = ma & ~clot_ma
        print(
            f"  dγ/dx [1/(m·s)]: clot mean={dgamma_dx[clot_ma].mean():.1f} "
            f"non-clot mean={dgamma_dx[non_ma].mean():.1f} "
            f"(expect clot more negative)"
        )
        print(
            f"  flux_path_dx: clot mean={flux_dx[clot_ma].mean():.3f} "
            f"non-clot mean={flux_dx[non_ma].mean():.3f}"
        )
        print(
            f"  comsol_hybrid prior: clot mean={prior_comsol_np[clot_ma].mean():.4f} "
            f"non-clot mean={prior_comsol_np[non_ma].mean():.4f} "
            f"frac>={thr_p} clot={float((prior_comsol_np[clot_ma] >= thr_p).mean()):.3f}"
        )

    print(f"\n=== Soft triggers on truth nodes (threshold {thr_p}) ===")
    for label, arr in (
        ("bio (FI/Mat)", bio),
        ("mech (kin prior)", mech),
        ("combined OR", comb),
        ("comsol_hybrid prior", prior_comsol_np),
        ("adjacent apply", adj),
        ("off-wall band only", band),
    ):
        if ma.any():
            frac_hi = float((arr[ma] >= thr_p).mean())
            print(
                f"  {label}: mean={arr[ma].mean():.4f} "
                f"p90={np.quantile(arr[ma], 0.9):.4f} frac>={thr_p}={frac_hi:.3f}"
            )
        else:
            print(f"  {label}: (no truth nodes)")

    print("\n=== Overlap: COMSOL clot vs trigger masks (truth nodes only) ===")
    clot_ma = y_clot & ma
    _overlap_report("prior_hi vs GT clot", clot_ma, prior_hi & ma, int(ma.sum()))
    _overlap_report("mech_hi vs GT clot", clot_ma, mech_hi & ma, int(ma.sum()))
    _overlap_report("bio_hi vs GT clot", clot_ma, bio_hi & ma, int(ma.sum()))
    _overlap_report("prior_hi vs GT clot (adjacent only)", clot_ma, prior_hi & (adj > 0.05) & ma, int(ma.sum()))

    print("\n=== Wall vs adjacent (why train gate_wall=0) ===")
    print(f"  apply_mask mean on wall nodes: {adj[wall].mean():.4f} (expect ~0)")
    print(f"  apply_mask mean on non-wall:    {adj[~wall].mean():.4f}")
    print(f"  apply p90 on truth non-wall:    {np.quantile(adj[ma & ~wall], 0.9) if (ma & ~wall).any() else 0:.4f}")

    if model is not None:
        print("\n=== Model open-loop rollout (optional) ===")
        roll_device = torch.device("cpu")
        model_cpu = model.to(roll_device)
        data_cpu = data.to(roll_device)
        eval_times = (
            data_cpu.t.to(roll_device)
            if hasattr(data_cpu, "t")
            else torch.linspace(0, 1, 12, device=roll_device)
        )
        pred = model_cpu(data_cpu, eval_times)
        if isinstance(pred, tuple):
            pred = pred[0]
        pred_np = pred.detach().cpu().numpy()
        t_roll = int(pred_np.shape[0]) - 1
        mu_p_si = phys.viscosity_nd_to_si(
            torch.from_numpy(pred_np[t_roll, :, STATE_CHANNEL_MU_EFF_ND])
        ).numpy()
        p_clot = getattr(model, "_last_mu_trigger_gate", None)
        if p_clot is not None:
            pc = p_clot.view(-1).cpu().numpy()
            print(f"  rollout last frame: gate_mean={pc.mean():.4f} gate_p90={np.quantile(pc, 0.9):.4f}")
            print(f"  μ_eff mean={mu_p_si.mean():.4f} p90={np.quantile(mu_p_si, 0.9):.4f}")
            print(f"  μ_eff on nodes with gate>0.5: count={(pc > 0.5).sum()} mean μ={(mu_p_si[pc > 0.5].mean() if (pc > 0.5).any() else 0):.4f}")
        trig = getattr(model, "_last_k11_trigger_mask", None)
        if trig is not None:
            tr = trig.view(-1).cpu().numpy()
            print(f"  last trigger_mask: mean={tr.mean():.4f} p90={np.quantile(tr, 0.9):.4f}")


if __name__ == "__main__":
    main()
