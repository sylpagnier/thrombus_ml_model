"""Fast signs-of-life probe for patient001 hop_ge2 hard-zero.

Goal: minutes, not hours. Detect whether the growth head can move lumen
nodes under teacher forcing vs closed-loop, on band vs global features.

Does NOT run the full crack ladder / multi-anchor probes.

Usage:
  python scripts/diagnose_001_signs_of_life.py
  python scripts/diagnose_001_signs_of_life.py --steps 40 --skip-deploy
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

from src.biochem_gnn.mat_growth_simple import apply_mat_growth_leg_env  # noqa: E402
from src.biochem_gnn.config import apply_deploy_env  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig, VesselConfig  # noqa: E402
from src.core_physics.clot_growth_masks import (  # noqa: E402
    graph_dilate_hops,
    gt_growth_commit_mask_at_time,
)
from src.core_physics.species_pushforward_continuous import (  # noqa: E402
    align_continuous_feature_dim,
    bind_band_geometry,
    build_continuous_step_features,
    clear_offwall_model_cache,
    compute_hop_distances,
    continuous_delta_threshold,
    deploy_eval_time_index,
    eval_deploy_clot_f1,
    load_continuous_bundle,
    maybe_drop_latent,
    noisy_teacher_log_state0,
    predict_continuous_step_delta,
    pushforward_log_state_step,
    splice_dynamic_flow,
    train_deploy_eval_flow_source,
)
from src.core_physics.species_pushforward_gnn import build_band_base_features  # noqa: E402
from src.core_physics.species_snapshot_gnn import (  # noqa: E402
    induced_subgraph,
    snapshot_wall_hops,
)
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.inference.corrector_coupling import resolve_kinematics_checkpoint  # noqa: E402
from src.training.biochem_species_scope import pushforward_state_bulk_indices  # noqa: E402
from src.training.train_offwall_growth import (  # noqa: E402
    _band_static_to_device,
    build_global_base_features,
    freeze_growth_backbone,
    unroll_offwall_loss_custom,
)
from src.utils import species_channels as sc  # noqa: E402
from src.utils.kinematics_inference import (  # noqa: E402
    load_kinematics_predictor,
    predict_kinematics_and_latent,
)
from src.utils.paths import get_project_root  # noqa: E402


def _mat_ch() -> int:
    try:
        from src.training.biochem_species_scope import pushforward_local_index

        return int(pushforward_local_index("mat"))
    except Exception:
        return 0


@torch.no_grad()
def _lumen_delta_stats(
    model,
    *,
    base_feats: torch.Tensor,
    edge_index: torch.Tensor,
    log_state0: torch.Tensor,
    wall_mask: torch.Tensor,
    pos: torch.Tensor,
    hop: torch.Tensor,
    species0: torch.Tensor,
    vel0: torch.Tensor | None,
    flow_series: torch.Tensor | None,
    flow_cols: tuple[int, int] | None,
    t0: int,
    n_steps: int = 4,
) -> dict[str, float]:
    """Short closed/open-loop roll; report Mat delta activity on hop>=2."""
    bind_band_geometry(
        model,
        {"pos_band": pos, "edge_index": edge_index, "wall_mask_band": wall_mask},
    )
    log_state = log_state0.clone()
    thr = float(continuous_delta_threshold())
    midx = _mat_ch()
    lumen = (hop >= 2) & (~wall_mask.reshape(-1))
    n_lumen = int(lumen.sum().item())
    abs_sum = 0.0
    n_fire = 0
    for s in range(n_steps):
        flow_ti = t0 + s
        step_feats = splice_dynamic_flow(base_feats, flow_series, flow_cols, flow_ti)
        step_feats = maybe_drop_latent(step_feats, model, False)
        model.log_state = log_state
        model.species_block = species0
        model.velocity = vel0
        feats = build_continuous_step_features(
            step_feats,
            log_state,
            training=False,
            time_index=t0 + s + 1,
            velocity=vel0,
            pos_band=pos,
            edge_index=edge_index,
        )
        feats = align_continuous_feature_dim(feats, model)
        # Use model forward path via predict_continuous_step_delta for dual-head fidelity
        delta = predict_continuous_step_delta(
            model,
            step_feats,
            edge_index,
            log_state,
            training=False,
            pos_band=pos,
            time_index=t0 + s + 1,
            flow_series=flow_series,
            flow_cols=flow_cols,
            flow_time_index=flow_ti,
            wall_mask_band=wall_mask,
            species_block=species0,
            velocity=vel0,
        )
        d = delta[:, midx].reshape(-1)
        abs_sum += float(d[lumen].abs().sum().item()) if n_lumen else 0.0
        n_fire += int((d[lumen] > thr).sum().item()) if n_lumen else 0
        log_state = pushforward_log_state_step(
            log_state, delta, straight_through=False, wall_speed=None, vel_decay_alphas=None
        )
    return {
        "n_lumen": float(n_lumen),
        "mat_abs_sum": abs_sum,
        "n_fire_gt_thr": float(n_fire),
        "mean_abs_per_lumen_step": abs_sum / max(n_lumen * n_steps, 1),
    }


def _build_late_tile(
    *,
    data,
    pack_band: dict,
    pack_global_feats: torch.Tensor,
    pack_global_flow,
    pack_global_flow_cols,
    phys,
    device,
    feat_source: str,
    hops_k: int,
    frontier_hops: int,
    unroll: int,
):
    n = int(data.num_nodes)
    edge_full = data.edge_index.to(device)
    from src.core_physics.clot_phi_simple import _wall_mask_from_data

    wall_full = _wall_mask_from_data(data, device, n)
    hop_full = compute_hop_distances(edge_full, wall_full, n)
    t_dep = int(deploy_eval_time_index(int(data.y.shape[0])))
    t0 = max(0, t_dep - unroll)
    win = list(range(t0, min(t0 + unroll + 1, int(data.y.shape[0]))))

    clot = torch.zeros(n, dtype=torch.bool, device=device)
    for ti in win:
        clot |= gt_growth_commit_mask_at_time(data, ti, phys, device)

    use_band = feat_source == "band"
    if use_band:
        band = pack_band
        band_nodes = band["node_idx"].long().to(device)
        edge_train = band["edge_index"].to(device)
        base = band["base_feats"].to(device)
        wall_train = band["wall_mask_band"].bool().to(device)
        pos = band["pos_band"].to(device)
        hop_train = hop_full[band_nodes]
        flow = band.get("flow_series")
        if flow is not None:
            flow = flow.to(device)
        flow_cols = band.get("flow_cols")
        clot_b = clot[band_nodes]
        seed = clot_b
        sub = graph_dilate_hops(seed, edge_train, hops_k)
        local_idx, edge_sub, _ = induced_subgraph(sub, edge_train)
        full_idx = band_nodes[local_idx]
        growth = graph_dilate_hops(seed, edge_train, frontier_hops)
        hop_sub = hop_train[local_idx]
        wall_sub = wall_train[local_idx]
        train_mask = growth[local_idx] & (hop_sub >= 2)
        base_sub = base[local_idx]
        pos_sub = pos[local_idx]
        flow_sub = flow[:, local_idx] if flow is not None else None
    else:
        sub = graph_dilate_hops(clot, edge_full, hops_k)
        local_idx, edge_sub, _ = induced_subgraph(sub, edge_full)
        full_idx = local_idx
        growth = graph_dilate_hops(clot, edge_full, frontier_hops)
        hop_sub = hop_full[local_idx]
        wall_sub = wall_full[local_idx]
        train_mask = growth[local_idx] & (hop_sub >= 2)
        base_sub = pack_global_feats.to(device)[local_idx]
        pos_sub = data.x[local_idx.cpu(), :2].to(device=device, dtype=base_sub.dtype)
        flow_sub = (
            pack_global_flow[:, local_idx.cpu()].to(device)
            if pack_global_flow is not None
            else None
        )
        flow_cols = pack_global_flow_cols

    bulk = pushforward_state_bulk_indices()
    full_cpu = full_idx.detach().cpu()
    series = []
    vel = []
    spb = []
    for ti in win:
        y = data.y[int(ti)].to(device=device, dtype=torch.float32)
        sp = y[:, sc.SPECIES_BLOCK]
        series.append(torch.stack([sp[:, int(c)] for c in bulk], dim=-1)[full_idx])
        vel.append(y[full_idx, 0:2].contiguous())
        spb.append(sp[full_idx].contiguous())

    return {
        "win": win,
        "base_feats": base_sub,
        "edge_index": edge_sub,
        "train_mask": train_mask,
        "pos": pos_sub,
        "wall_mask": wall_sub,
        "hop": hop_sub,
        "series": series,
        "velocity": vel,
        "species_block": spb,
        "flow_series": flow_sub,
        "flow_cols": flow_cols,
        "n_mask": int(train_mask.sum().item()),
        "n_lumen_in_tile": int(((hop_sub >= 2) & (~wall_sub)).sum().item()),
    }


def _mini_train(model, tile, *, steps: int, lr: float, lumen_w: float, device) -> list[float]:
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    losses = []
    model.train()
    for _ in range(steps):
        loss = unroll_offwall_loss_custom(
            model,
            base_feats=tile["base_feats"],
            edge_index=tile["edge_index"],
            log_series=tile["series"],
            train_mask=tile["train_mask"],
            pos_band=tile["pos"],
            time_window=tile["win"],
            flow_series=tile["flow_series"],
            flow_cols=tile["flow_cols"],
            wall_mask_band=tile["wall_mask"],
            species_block=tile["species_block"],
            velocity=tile["velocity"],
            loss_mode="loss_lumen_shape",
            device=device,
            hop_dist=tile["hop"],
            lumen_shape_weight=lumen_w,
        )
        if not loss.requires_grad:
            losses.append(float("nan"))
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))
    return losses


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default="patient001")
    ap.add_argument(
        "--wall-ckpt",
        default="outputs/biochem/biochem_gnn/locked/species_gnn_best.pth",
    )
    ap.add_argument("--mat-leg", default="WC_v7_clot_phi_mse")
    ap.add_argument("--steps", type=int, default=40, help="Mini-train steps per feat source")
    ap.add_argument("--unroll", type=int, default=8)
    ap.add_argument("--hops-k", type=int, default=5)
    ap.add_argument("--frontier-hops", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lumen-shape-weight", type=float, default=10.0)
    ap.add_argument("--skip-deploy", action="store_true")
    ap.add_argument(
        "--out",
        default="outputs/biochem/offwall_model/wc_v7_crack_001_3h/signs_of_life.json",
    )
    args = ap.parse_args()

    device = require_cuda_device()
    root = get_project_root()
    wall_ckpt = Path(args.wall_ckpt)
    if not wall_ckpt.is_absolute():
        wall_ckpt = root / wall_ckpt

    apply_mat_growth_leg_env(args.mat_leg, force=True)
    os.environ["SPECIES_LUMEN_SHAPE_FN_W"] = "25"
    os.environ["SPECIES_LUMEN_SHAPE_FP_W"] = "0.35"
    os.environ["SPECIES_CONTINUOUS_UNDERPRED_WEIGHT"] = "12.0"
    os.environ["SPECIES_TWO_MODEL_MODE"] = "0"
    os.environ.pop("SPECIES_OFFWALL_MODEL_CKPT", None)

    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    graph = root / VesselConfig(phase="biochem_anchors").graph_output_dir / f"{args.anchor}.pt"
    data = torch.load(graph, map_location="cpu", weights_only=False)

    print("=" * 72, flush=True)
    print(f"SIGNS OF LIFE — {args.anchor} (steps={args.steps})", flush=True)
    print("=" * 72, flush=True)

    kine = load_kinematics_predictor(
        str(resolve_kinematics_checkpoint()),
        device,
        phys_cfg=PhysicsConfig(phase="kinematics"),
    )
    with torch.no_grad():
        pred_uv, z = predict_kinematics_and_latent(kine, data)
    data.u0_pred = pred_uv[:, 0].detach().cpu()
    data.v0_pred = pred_uv[:, 1].detach().cpu()
    band = build_band_base_features(
        data, kine, device, wall_hops=snapshot_wall_hops(), z_kin_override=z
    )
    band_cpu = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in band.items()}
    glob_feats = build_global_base_features(data, kine, device).cpu()
    glob_flow = band_cpu.get("flow_series")  # reuse band flow width if present
    # Prefer full-graph dynamic flow if available via band builder path already on band;
    # for global tiles use None unless we rebuild — OK for signs-of-life.
    glob_flow_full = None
    glob_flow_cols = None

    report: dict = {"anchor": args.anchor, "steps": args.steps, "arms": {}}

    for feat_source in ("band", "global"):
        print(f"\n[i] === feat_source={feat_source} ===", flush=True)
        clear_offwall_model_cache()
        bundle = load_continuous_bundle(
            wall_ckpt, device=device, quiet=True, architecture="dual"
        )
        assert bundle is not None
        model = bundle.model
        n_fr, n_tr = freeze_growth_backbone(model)
        print(f"[i] freeze-backbone frozen={n_fr} trainable={n_tr}", flush=True)

        tile = _build_late_tile(
            data=data,
            pack_band=band_cpu,
            pack_global_feats=glob_feats,
            pack_global_flow=glob_flow_full,
            pack_global_flow_cols=glob_flow_cols,
            phys=phys,
            device=device,
            feat_source=feat_source,
            hops_k=int(args.hops_k),
            frontier_hops=int(args.frontier_hops),
            unroll=int(args.unroll),
        )
        print(
            f"[i] tile mask_nodes={tile['n_mask']} lumen_in_tile={tile['n_lumen_in_tile']} "
            f"win={tile['win'][0]}..{tile['win'][-1]}",
            flush=True,
        )
        if tile["n_mask"] <= 0:
            print("[ERR] empty frontier_ge2 mask — cannot train", flush=True)
            report["arms"][feat_source] = {"error": "empty_train_mask"}
            continue

        # Pre-train activity (teacher IC)
        model.eval()
        pre_tf = _lumen_delta_stats(
            model,
            base_feats=tile["base_feats"],
            edge_index=tile["edge_index"],
            log_state0=noisy_teacher_log_state0(
                tile["series"][0], tile["edge_index"], training=False
            ),
            wall_mask=tile["wall_mask"],
            pos=tile["pos"],
            hop=tile["hop"],
            species0=tile["species_block"][0],
            vel0=tile["velocity"][0],
            flow_series=tile["flow_series"],
            flow_cols=tile["flow_cols"],
            t0=int(tile["win"][0]),
        )
        resting = torch.zeros_like(tile["series"][0])
        pre_cl = _lumen_delta_stats(
            model,
            base_feats=tile["base_feats"],
            edge_index=tile["edge_index"],
            log_state0=resting,
            wall_mask=tile["wall_mask"],
            pos=tile["pos"],
            hop=tile["hop"],
            species0=tile["species_block"][0],
            vel0=tile["velocity"][0],
            flow_series=tile["flow_series"],
            flow_cols=tile["flow_cols"],
            t0=int(tile["win"][0]),
        )
        print(
            f"[i] pre  TF lumen fire={pre_tf['n_fire_gt_thr']:.0f} "
            f"mean_abs={pre_tf['mean_abs_per_lumen_step']:.3e} | "
            f"CL fire={pre_cl['n_fire_gt_thr']:.0f} mean_abs={pre_cl['mean_abs_per_lumen_step']:.3e}",
            flush=True,
        )

        losses = _mini_train(
            model,
            tile,
            steps=int(args.steps),
            lr=float(args.lr),
            lumen_w=float(args.lumen_shape_weight),
            device=device,
        )
        loss0 = next((x for x in losses if x == x), float("nan"))
        loss1 = next((x for x in reversed(losses) if x == x), float("nan"))
        print(f"[i] loss {loss0:.4f} -> {loss1:.4f} ({args.steps} steps)", flush=True)

        model.eval()
        post_tf = _lumen_delta_stats(
            model,
            base_feats=tile["base_feats"],
            edge_index=tile["edge_index"],
            log_state0=noisy_teacher_log_state0(
                tile["series"][0], tile["edge_index"], training=False
            ),
            wall_mask=tile["wall_mask"],
            pos=tile["pos"],
            hop=tile["hop"],
            species0=tile["species_block"][0],
            vel0=tile["velocity"][0],
            flow_series=tile["flow_series"],
            flow_cols=tile["flow_cols"],
            t0=int(tile["win"][0]),
        )
        post_cl = _lumen_delta_stats(
            model,
            base_feats=tile["base_feats"],
            edge_index=tile["edge_index"],
            log_state0=resting,
            wall_mask=tile["wall_mask"],
            pos=tile["pos"],
            hop=tile["hop"],
            species0=tile["species_block"][0],
            vel0=tile["velocity"][0],
            flow_series=tile["flow_series"],
            flow_cols=tile["flow_cols"],
            t0=int(tile["win"][0]),
        )
        print(
            f"[i] post TF lumen fire={post_tf['n_fire_gt_thr']:.0f} "
            f"mean_abs={post_tf['mean_abs_per_lumen_step']:.3e} | "
            f"CL fire={post_cl['n_fire_gt_thr']:.0f} mean_abs={post_cl['mean_abs_per_lumen_step']:.3e}",
            flush=True,
        )

        arm = {
            "n_mask": tile["n_mask"],
            "loss_start": loss0,
            "loss_end": loss1,
            "pre_teacher": pre_tf,
            "pre_closed_loop": pre_cl,
            "post_teacher": post_tf,
            "post_closed_loop": post_cl,
            "life_teacher": post_tf["n_fire_gt_thr"] > 0.5
            or post_tf["mean_abs_per_lumen_step"] > 1e-5,
            "life_closed_loop": post_cl["n_fire_gt_thr"] > 0.5
            or post_cl["mean_abs_per_lumen_step"] > 1e-5,
        }

        if feat_source == "band" and not args.skip_deploy:
            # Cheap single-anchor compound deploy (no 007/004/008).
            tmp = root / "outputs/biochem/offwall_model/wc_v7_crack_001_3h/_signs_growth_tmp.pth"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            from src.core_physics.species_pushforward_continuous import save_continuous_checkpoint

            save_continuous_checkpoint(
                tmp,
                model,
                {"signs_of_life": True, "train_feat_source": "band"},
            )
            clear_offwall_model_cache()
            os.environ["SPECIES_TWO_MODEL_MODE"] = "1"
            os.environ["SPECIES_OFFWALL_MODEL_CKPT"] = str(tmp)
            os.environ["SPECIES_TWO_MODEL_ROUTE"] = "wall"
            wall_b = load_continuous_bundle(wall_ckpt, device=device, quiet=True)
            assert wall_b is not None
            static = _band_static_to_device(band_cpu, device)
            data_g = data.to(device)
            flow_eval = train_deploy_eval_flow_source()
            apply_deploy_env(overrides={"T0_R4_FLOW_SOURCE": flow_eval})
            clf = eval_deploy_clot_f1(
                wall_b.model,
                data_g,
                static,
                phys,
                bio,
                device,
                time_index=deploy_eval_time_index(int(data.y.shape[0])),
                flow_source=flow_eval,
            )
            arm["deploy_001"] = {
                "clot_f1": float(clf.get("deploy_clot_f1", 0.0)),
                "hop_ge2_n_pred": float(clf.get("deploy_clot_offwall_n_pred_hop_ge2", 0.0)),
                "hop_ge2_n_gt": float(clf.get("deploy_clot_offwall_n_gt_hop_ge2", 0.0)),
            }
            print(
                f"[i] deploy001 clot_f1={arm['deploy_001']['clot_f1']:.3f} "
                f"hop_ge2={arm['deploy_001']['hop_ge2_n_pred']:.0f}/"
                f"{arm['deploy_001']['hop_ge2_n_gt']:.0f}",
                flush=True,
            )
            os.environ["SPECIES_TWO_MODEL_MODE"] = "0"
            os.environ.pop("SPECIES_OFFWALL_MODEL_CKPT", None)
            clear_offwall_model_cache()

        report["arms"][feat_source] = arm

    b = report["arms"].get("band") or {}
    g = report["arms"].get("global") or {}
    if b.get("life_teacher") and not b.get("life_closed_loop"):
        verdict = "life_on_teacher_dead_on_closed_loop"
    elif b.get("life_closed_loop") or (
        b.get("deploy_001") and b["deploy_001"].get("hop_ge2_n_pred", 0) > 0.5
    ):
        verdict = "signs_of_life_band"
    elif g.get("life_teacher") and not b.get("life_teacher"):
        verdict = "life_only_on_global_feats_path_bug"
    elif b.get("life_teacher") or g.get("life_teacher"):
        verdict = "partial_life_no_deploy"
    else:
        verdict = "dead_everywhere_wiring_or_loss"

    report["verdict"] = verdict
    print("\n" + "=" * 72, flush=True)
    print(f"[i] verdict={verdict}", flush=True)

    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[save] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
