"""Debug root cause of patient001 lumen lock + inert compound-val.

Compares the train_offwall_growth compound-val static (full-graph features,
all-node idx) vs eval_mat_growth_simple static (wall-band features).

Also probes growth-alone vs wall-route compound on 001/007, and resting vs
teacher-forced IC for hop_ge2 activity.

Usage:
  python scripts/diagnose_crack_001_root.py
  python scripts/diagnose_crack_001_root.py --growth-ckpt outputs/.../growth_Solo001_Freeze/best.pth
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

from scripts.eval_mat_growth_simple import _apply_ckpt_recipe, _load_static  # noqa: E402
from src.biochem_gnn.config import apply_deploy_env  # noqa: E402
from src.biochem_gnn.mat_growth_simple import apply_mat_growth_leg_env  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.clot_phi_simple import _wall_mask_from_data  # noqa: E402
from src.core_physics.species_pushforward_continuous import (  # noqa: E402
    clear_offwall_model_cache,
    compute_hop_distances,
    deploy_eval_time_index,
    eval_deploy_clot_f1,
    load_continuous_bundle,
    train_deploy_eval_flow_source,
)
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.inference.corrector_coupling import resolve_kinematics_checkpoint  # noqa: E402
from src.training.train_offwall_growth import build_global_base_features  # noqa: E402
from src.utils.kinematics_inference import load_kinematics_predictor  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402

ANCHOR_DIR = get_project_root() / "data/processed/graphs_biochem_anchors"
DEFAULT_WALL = get_project_root() / "outputs/biochem/biochem_gnn/locked/species_gnn_best.pth"
DEFAULT_GROWTH = (
    get_project_root()
    / "outputs/biochem/offwall_model/wc_v7_crack_001_3h/growth_Solo001_Freeze/best.pth"
)


def _summarize(clot_m: dict) -> dict[str, float]:
    keys = (
        "deploy_clot_f1",
        "deploy_clot_offwall_n_pred",
        "deploy_clot_offwall_n_gt",
        "deploy_clot_offwall_n_pred_hop_ge2",
        "deploy_clot_offwall_n_gt_hop_ge2",
        "deploy_clot_offwall_strict_f1_hop_ge2",
    )
    return {k: float(clot_m.get(k, 0.0) or 0.0) for k in keys}


def _static_train_style(data, device, kine) -> dict:
    """Match train_offwall_growth compound-val static construction."""
    base = build_global_base_features(data, kine, device)
    wall = _wall_mask_from_data(data, device, int(data.num_nodes))
    return {
        "node_idx": torch.arange(int(data.num_nodes), device=device),
        "base_feats": base.to(device),
        "edge_index": data.edge_index.to(device),
        "pos_band": data.x[:, :2].to(device=device, dtype=base.dtype),
        "wall_mask_band": wall.to(device),
        "style": "train_full_graph",
        "n_nodes_static": int(base.shape[0]),
        "feat_dim": int(base.shape[1]),
    }


def _static_eval_style(data, device, kine, wall_hops: int = 3) -> dict:
    """Match eval_mat_growth_simple static construction."""
    static = _load_static(data, device, kine, wall_hops)
    static["style"] = "eval_wall_band"
    static["n_nodes_static"] = int(static["base_feats"].shape[0])
    static["feat_dim"] = int(static["base_feats"].shape[1])
    return static


@torch.no_grad()
def _run_deploy(model, data, static, phys, bio, device) -> dict:
    t_eval = deploy_eval_time_index(int(data.y.shape[0]))
    flow_eval = train_deploy_eval_flow_source()
    apply_deploy_env(overrides={"T0_R4_FLOW_SOURCE": flow_eval})
    return eval_deploy_clot_f1(
        model,
        data,
        static,
        phys,
        bio,
        device,
        time_index=t_eval,
        flow_source=flow_eval,
    )


def _gt_hop_ge2(data, phys, device) -> int:
    n = int(data.num_nodes)
    wall = _wall_mask_from_data(data, device, n)
    hops = compute_hop_distances(data.edge_index, wall, n)
    t_dep = int(deploy_eval_time_index(int(data.y.shape[0])))
    phi = gt_clot_phi_at_time(data, t_dep, phys, device).reshape(-1)
    return int(((phi >= 0.5) & (hops >= 2)).sum().item())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wall-ckpt", default=str(DEFAULT_WALL))
    ap.add_argument("--growth-ckpt", default=str(DEFAULT_GROWTH))
    ap.add_argument("--anchors", default="patient001,patient007")
    ap.add_argument("--mat-leg", default="WC_v7_clot_phi_mse")
    ap.add_argument(
        "--out",
        default="outputs/biochem/offwall_model/wc_v7_crack_001_3h/diagnose_root.json",
    )
    args = ap.parse_args()

    device = require_cuda_device()
    root = get_project_root()
    wall_ckpt = Path(args.wall_ckpt)
    if not wall_ckpt.is_absolute():
        wall_ckpt = root / wall_ckpt
    growth_ckpt = Path(args.growth_ckpt)
    if not growth_ckpt.is_absolute():
        growth_ckpt = root / growth_ckpt
    anchors = [a.strip() for a in args.anchors.split(",") if a.strip()]

    apply_mat_growth_leg_env(args.mat_leg, force=True)
    payload = torch.load(wall_ckpt, map_location="cpu", weights_only=False)
    _apply_ckpt_recipe(dict(payload.get("meta") or {}), label="wall", ckpt_path=wall_ckpt)

    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    kine = load_kinematics_predictor(
        str(resolve_kinematics_checkpoint()),
        device,
        phys_cfg=PhysicsConfig(phase="kinematics"),
    )

    report: dict = {
        "wall_ckpt": str(wall_ckpt),
        "growth_ckpt": str(growth_ckpt) if growth_ckpt.is_file() else None,
        "anchors": {},
        "root_cause_candidates": [],
    }

    print("=" * 78, flush=True)
    print("CRACK 001 ROOT DIAGNOSE", flush=True)
    print("=" * 78, flush=True)

    for anc in anchors:
        print(f"\n[i] anchor={anc}", flush=True)
        data = torch.load(ANCHOR_DIR / f"{anc}.pt", map_location=device, weights_only=False)
        gt_ge2 = _gt_hop_ge2(data, phys, device)

        st_train = _static_train_style(data, device, kine)
        st_eval = _static_eval_style(data, device, kine, wall_hops=3)
        print(
            f"  static train: nodes={st_train['n_nodes_static']} feat_dim={st_train['feat_dim']}",
            flush=True,
        )
        print(
            f"  static eval:  nodes={st_eval['n_nodes_static']} feat_dim={st_eval['feat_dim']}",
            flush=True,
        )

        clear_offwall_model_cache()
        os.environ["SPECIES_TWO_MODEL_MODE"] = "0"
        os.environ.pop("SPECIES_OFFWALL_MODEL_CKPT", None)
        os.environ.pop("SPECIES_TWO_MODEL_ROUTE", None)
        wall_bundle = load_continuous_bundle(wall_ckpt, device=device, quiet=True)
        assert wall_bundle is not None
        wall_model = wall_bundle.model
        wall_model.eval()

        m_train = _summarize(_run_deploy(wall_model, data, st_train, phys, bio, device))
        m_eval = _summarize(_run_deploy(wall_model, data, st_eval, phys, bio, device))
        print(
            f"  wall-only TRAIN-static: clot_f1={m_train['deploy_clot_f1']:.4f} "
            f"ge2={m_train['deploy_clot_offwall_n_pred_hop_ge2']:.0f}/"
            f"{m_train['deploy_clot_offwall_n_gt_hop_ge2']:.0f}",
            flush=True,
        )
        print(
            f"  wall-only EVAL-static:  clot_f1={m_eval['deploy_clot_f1']:.4f} "
            f"ge2={m_eval['deploy_clot_offwall_n_pred_hop_ge2']:.0f}/"
            f"{m_eval['deploy_clot_offwall_n_gt_hop_ge2']:.0f}",
            flush=True,
        )

        row = {
            "gt_hop_ge2": gt_ge2,
            "static_train": {
                "n_nodes": st_train["n_nodes_static"],
                "feat_dim": st_train["feat_dim"],
            },
            "static_eval": {
                "n_nodes": st_eval["n_nodes_static"],
                "feat_dim": st_eval["feat_dim"],
            },
            "wall_only_train_static": m_train,
            "wall_only_eval_static": m_eval,
        }

        if growth_ckpt.is_file():
            # Compound wall-route on EVAL static (real probe path)
            clear_offwall_model_cache()
            os.environ["SPECIES_TWO_MODEL_MODE"] = "1"
            os.environ["SPECIES_OFFWALL_MODEL_CKPT"] = str(growth_ckpt)
            os.environ["SPECIES_TWO_MODEL_ROUTE"] = "wall"
            m_comp_eval = _summarize(_run_deploy(wall_model, data, st_eval, phys, bio, device))
            print(
                f"  compound EVAL-static:   clot_f1={m_comp_eval['deploy_clot_f1']:.4f} "
                f"ge2={m_comp_eval['deploy_clot_offwall_n_pred_hop_ge2']:.0f}/"
                f"{m_comp_eval['deploy_clot_offwall_n_gt_hop_ge2']:.0f}",
                flush=True,
            )

            # Compound on TRAIN static (compound-val path)
            clear_offwall_model_cache()
            os.environ["SPECIES_TWO_MODEL_MODE"] = "1"
            os.environ["SPECIES_OFFWALL_MODEL_CKPT"] = str(growth_ckpt)
            os.environ["SPECIES_TWO_MODEL_ROUTE"] = "wall"
            m_comp_train = _summarize(_run_deploy(wall_model, data, st_train, phys, bio, device))
            print(
                f"  compound TRAIN-static:  clot_f1={m_comp_train['deploy_clot_f1']:.4f} "
                f"ge2={m_comp_train['deploy_clot_offwall_n_pred_hop_ge2']:.0f}/"
                f"{m_comp_train['deploy_clot_offwall_n_gt_hop_ge2']:.0f}",
                flush=True,
            )

            # Growth-alone on EVAL static (specialist without wall blend)
            clear_offwall_model_cache()
            os.environ["SPECIES_TWO_MODEL_MODE"] = "0"
            os.environ.pop("SPECIES_OFFWALL_MODEL_CKPT", None)
            growth_bundle = load_continuous_bundle(growth_ckpt, device=device, quiet=True)
            assert growth_bundle is not None
            g_model = growth_bundle.model
            g_model.eval()
            m_growth = _summarize(_run_deploy(g_model, data, st_eval, phys, bio, device))
            print(
                f"  growth-alone EVAL:      clot_f1={m_growth['deploy_clot_f1']:.4f} "
                f"ge2={m_growth['deploy_clot_offwall_n_pred_hop_ge2']:.0f}/"
                f"{m_growth['deploy_clot_offwall_n_gt_hop_ge2']:.0f}",
                flush=True,
            )
            row.update(
                {
                    "compound_eval_static": m_comp_eval,
                    "compound_train_static": m_comp_train,
                    "growth_alone_eval_static": m_growth,
                }
            )

        report["anchors"][anc] = row

    # Root-cause classification
    a001 = report["anchors"].get("patient001") or {}
    a007 = report["anchors"].get("patient007") or {}
    w001_train = (a001.get("wall_only_train_static") or {}).get("deploy_clot_f1", None)
    w001_eval = (a001.get("wall_only_eval_static") or {}).get("deploy_clot_f1", None)
    w007_train = (a007.get("wall_only_train_static") or {}).get("deploy_clot_f1", None)
    w007_eval = (a007.get("wall_only_eval_static") or {}).get("deploy_clot_f1", None)

    verdict = "inconclusive"
    if (
        w001_train is not None
        and w001_eval is not None
        and w001_train < 0.05
        and w001_eval > 0.5
    ):
        verdict = "compound_val_uses_wrong_static_full_graph_vs_wall_band"
        report["root_cause_candidates"].append(
            "eval_wall_only_deploy_floor / compound-val builds full-graph static "
            "(node_idx=all); eval_mat_growth_simple uses wall-band static. "
            "That explains A_floor=0 and inert hop_ge2_recall selection on 001."
        )
    if (
        a001.get("compound_eval_static")
        and a001["compound_eval_static"]["deploy_clot_offwall_n_pred_hop_ge2"] < 0.5
        and a007.get("compound_eval_static")
        and a007["compound_eval_static"]["deploy_clot_offwall_n_pred_hop_ge2"] > 0.5
    ):
        report["root_cause_candidates"].append(
            "On the correct EVAL static, compound still opens 007 but not 001 — "
            "001 lock is not only a val-metric bug; deploy path remains silent on 001."
        )
        if verdict == "inconclusive":
            verdict = "001_deploy_silent_despite_correct_static"
    if (
        a001.get("growth_alone_eval_static")
        and a001["growth_alone_eval_static"]["deploy_clot_offwall_n_pred_hop_ge2"] < 0.5
        and a001.get("growth_alone_eval_static")
        and a001["growth_alone_eval_static"]["deploy_clot_f1"] < 0.2
    ):
        report["root_cause_candidates"].append(
            "Growth-alone also fails to form clot/lumen on 001 — specialist cannot "
            "nucleate; depends on wall model. Wall-route then keeps 001 lumen dark."
        )

    # Feature geometry mismatch note
    if a001:
        nt = a001["static_train"]["n_nodes"]
        ne = a001["static_eval"]["n_nodes"]
        if nt != ne:
            report["root_cause_candidates"].append(
                f"patient001 static node count train={nt} vs eval={ne} "
                f"(full graph vs wall-band)."
            )

    report["verdict"] = verdict
    report["floor_gap"] = {
        "patient001_train_vs_eval_clot_f1": [w001_train, w001_eval],
        "patient007_train_vs_eval_clot_f1": [w007_train, w007_eval],
    }

    print("\n" + "=" * 78, flush=True)
    print(f"[i] verdict={verdict}", flush=True)
    for c in report["root_cause_candidates"]:
        print(f"[i] cause: {c}", flush=True)

    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[save] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
