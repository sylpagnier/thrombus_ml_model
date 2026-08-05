"""Longer band mini-train + single compound deploy on patient001."""

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

from scripts.diagnose_001_signs_of_life import (  # noqa: E402
    _build_late_tile,
    _lumen_delta_stats,
    _mini_train,
)
from src.biochem_gnn.config import apply_deploy_env  # noqa: E402
from src.biochem_gnn.mat_growth_simple import apply_mat_growth_leg_env  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.species_pushforward_continuous import (  # noqa: E402
    clear_offwall_model_cache,
    continuous_delta_out_scale,
    continuous_delta_threshold,
    continuous_mat_commit_thresh,
    deploy_eval_time_index,
    eval_deploy_clot_f1,
    load_continuous_bundle,
    noisy_teacher_log_state0,
    save_continuous_checkpoint,
    train_deploy_eval_flow_source,
)
from src.core_physics.species_pushforward_gnn import build_band_base_features  # noqa: E402
from src.core_physics.species_snapshot_gnn import snapshot_wall_hops  # noqa: E402
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.inference.corrector_coupling import resolve_kinematics_checkpoint  # noqa: E402
from src.training.train_offwall_growth import (  # noqa: E402
    _band_static_to_device,
    build_global_base_features,
    freeze_growth_backbone,
)
from src.utils.kinematics_inference import (  # noqa: E402
    load_kinematics_predictor,
    predict_kinematics_and_latent,
)
from src.utils.paths import get_project_root  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    apply_mat_growth_leg_env("WC_v7_clot_phi_mse", force=True)
    os.environ["SPECIES_LUMEN_SHAPE_FN_W"] = "25"
    os.environ["SPECIES_LUMEN_SHAPE_FP_W"] = "0.35"
    os.environ["SPECIES_CONTINUOUS_UNDERPRED_WEIGHT"] = "12.0"

    device = require_cuda_device()
    root = get_project_root()
    wall = root / "outputs/biochem/biochem_gnn/locked/species_gnn_best.pth"
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    data = torch.load(
        root / "data/processed/graphs_biochem_anchors/patient001.pt",
        map_location="cpu",
        weights_only=False,
    )
    kine = load_kinematics_predictor(
        str(resolve_kinematics_checkpoint()),
        device,
        phys_cfg=PhysicsConfig(phase="kinematics"),
    )
    with torch.no_grad():
        uv, z = predict_kinematics_and_latent(kine, data)
    data.u0_pred = uv[:, 0].cpu()
    data.v0_pred = uv[:, 1].cpu()
    band = build_band_base_features(
        data, kine, device, wall_hops=snapshot_wall_hops(), z_kin_override=z
    )
    band_cpu = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in band.items()}
    glob = build_global_base_features(data, kine, device).cpu()

    print(
        f"[i] thr={continuous_delta_threshold():.1e} "
        f"commit={continuous_mat_commit_thresh():.1e} "
        f"out_scale={continuous_delta_out_scale():.1e}",
        flush=True,
    )
    print(f"[i] LONG band mini-train steps={args.steps}", flush=True)

    clear_offwall_model_cache()
    bundle = load_continuous_bundle(wall, device=device, quiet=True, architecture="dual")
    assert bundle is not None
    model = bundle.model
    freeze_growth_backbone(model)
    tile = _build_late_tile(
        data=data,
        pack_band=band_cpu,
        pack_global_feats=glob,
        pack_global_flow=None,
        pack_global_flow_cols=None,
        phys=phys,
        device=device,
        feat_source="band",
        hops_k=5,
        frontier_hops=2,
        unroll=8,
    )
    model.eval()
    pre = _lumen_delta_stats(
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
    losses = _mini_train(
        model, tile, steps=int(args.steps), lr=float(args.lr), lumen_w=10.0, device=device
    )
    loss0 = next(x for x in losses if x == x)
    loss1 = next(x for x in reversed(losses) if x == x)
    model.eval()
    post = _lumen_delta_stats(
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
    print(
        f"[i] fire {pre['n_fire_gt_thr']:.0f}->{post['n_fire_gt_thr']:.0f} "
        f"mean_abs {pre['mean_abs_per_lumen_step']:.3e}->{post['mean_abs_per_lumen_step']:.3e} "
        f"loss {loss0:.4f}->{loss1:.4f}",
        flush=True,
    )

    out_dir = root / "outputs/biochem/offwall_model/wc_v7_crack_001_3h"
    tmp = out_dir / f"_narrow_growth_band_{args.steps}.pth"
    save_continuous_checkpoint(
        tmp, model, {"steps": int(args.steps), "train_feat_source": "band"}
    )

    clear_offwall_model_cache()
    os.environ["SPECIES_TWO_MODEL_MODE"] = "1"
    os.environ["SPECIES_OFFWALL_MODEL_CKPT"] = str(tmp)
    os.environ["SPECIES_TWO_MODEL_ROUTE"] = "wall"
    wb = load_continuous_bundle(wall, device=device, quiet=True)
    assert wb is not None
    static = _band_static_to_device(band_cpu, device)
    flow = train_deploy_eval_flow_source()
    apply_deploy_env(overrides={"T0_R4_FLOW_SOURCE": flow})
    print("[i] compound deploy 001...", flush=True)
    m = eval_deploy_clot_f1(
        wb.model,
        data.to(device),
        static,
        phys,
        bio,
        device,
        time_index=deploy_eval_time_index(int(data.y.shape[0])),
        flow_source=flow,
    )
    ge2 = float(m.get("deploy_clot_offwall_n_pred_hop_ge2", 0) or 0)
    gt = float(m.get("deploy_clot_offwall_n_gt_hop_ge2", 0) or 0)
    f1 = float(m.get("deploy_clot_f1", 0) or 0)
    print(f"[OK] steps={args.steps} clot_f1={f1:.3f} hop_ge2={ge2:.0f}/{gt:.0f}", flush=True)

    payload = {
        "steps": int(args.steps),
        "fire_pre": pre["n_fire_gt_thr"],
        "fire_post": post["n_fire_gt_thr"],
        "mean_abs_pre": pre["mean_abs_per_lumen_step"],
        "mean_abs_post": post["mean_abs_per_lumen_step"],
        "loss": [loss0, loss1],
        "clot_f1": f1,
        "hop_ge2_pred": ge2,
        "hop_ge2_gt": gt,
        "thr": continuous_delta_threshold(),
        "commit": continuous_mat_commit_thresh(),
        "out_scale": continuous_delta_out_scale(),
    }
    out_json = out_dir / f"diagnose_narrow_{args.steps}.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[save] {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
