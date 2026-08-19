"""Train the Advanced ML Architecture sweep sequentially.

Usage:
    python scripts/train_advanced_ml_sweep.py --epochs 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.biochem_gnn.mat_growth_simple import (
    WALL_COHORT_V2_GENERALIZATION,
    WALL_COHORT_V2_TRAIN,
)
from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.species_pushforward_continuous import resolve_deploy_eval_time_index
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
from src.differentiable_wall_model.differentiable_ode import DifferentiableWallModel
from src.differentiable_wall_model.evaluation import evaluate_cohort
from src.differentiable_wall_model.losses import CombinedWallClotLoss
from src.differentiable_wall_model.parameters import GlobalPhysicsParameters
from src.differentiable_wall_model.advanced_models import (
    GATv2ResidualCorrector,
    SpatiotemporalResidualCorrector,
    EdgeConditionedResidualCorrector,
    StateInjectedResidualCorrector,
    MultiTaskResidualCorrector,
    MeshGraphNetCorrector,
    RGPResidualCorrector,
)

GRAPH_DIR = Path("data/processed/graphs_biochem_anchors")


def get_data(train_anchors, flow_source):
    valid_train = []
    for a in train_anchors:
        p = GRAPH_DIR / f"{a}.pt"
        if p.exists():
            d = torch.load(p, map_location="cpu", weights_only=False)
            if int(d.y.shape[0]) >= 150:
                if flow_source == "pred" and getattr(d, "u0_pred", None) is None:
                    continue
                valid_train.append(a)
    return valid_train


def preload_graphs(valid_train, phys_cfg, device=torch.device("cpu")):
    train_graphs = {}
    for a in valid_train:
        d = torch.load(GRAPH_DIR / f"{a}.pt", map_location="cpu", weights_only=False)
        t_eval = resolve_deploy_eval_time_index(int(d.y.shape[0]))
        gt_phi = gt_clot_phi_at_time(d, t_eval, phys_cfg, device=device).reshape(-1)
        wall = d.mask_wall.reshape(-1).bool()
        train_graphs[a] = {"data": d, "gt_phi": gt_phi * wall.float(), "wall": wall}
    return train_graphs


def train_rung(name, model_factory, train_graphs, valid_train, sealed_anchors, phys_cfg, args, device, use_multitask=False):
    print(f"\n{'='*50}\n Sweep Leg: {name}\n{'='*50}")
    
    current_lr = args.lr
    max_retries = 3
    
    for attempt in range(max_retries):
        if attempt > 0:
            print(f"\n[!] Retry {attempt}/{max_retries - 1} for {name} with LR = {current_lr:.6f}")
            
        model = model_factory().to(device)
        loss_fn = CombinedWallClotLoss()
        optimizer = optim.AdamW(model.parameters(), lr=current_lr)
        mse = nn.MSELoss()

        # Base eval
        base_eval_tr = evaluate_cohort(model, valid_train, GRAPH_DIR, phys_cfg, flow_source=args.flow, device=device)
        base_eval_sl = evaluate_cohort(model, sealed_anchors, GRAPH_DIR, phys_cfg, flow_source=args.flow, device=device)
        print(f"[Before] Train Deploy F1: {base_eval_tr['mean_f1']:.4f} | Sealed: {base_eval_sl['mean_f1']:.4f}")

        start_time = time.time()
        for ep in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad()
            cohort_loss_val = 0.0

            for a, pack in train_graphs.items():
                d = pack["data"]
                gt = pack["gt_phi"].to(device)
                wall = pack["wall"].to(device)
                
                # Forward pass
                out = model(d, flow_source=args.flow, device=device)
                prob = out["prob_clot"]
                
                # Primary loss
                l_dict = loss_fn(prob, gt, wall)
                loss_vessel = l_dict["loss"]
                
                # Auxiliary loss for MultiTask model
                if use_multitask and "wss_pred" in out:
                    # Assuming WSS is channel 3 in data.y (standard configuration for kinematics)
                    gt_wss = d.y[0, :, 3].to(device) * wall
                    pred_wss = out["wss_pred"] * wall
                    wss_loss = mse(pred_wss, gt_wss) * 0.1 # scaled down to not overpower primary loss
                    loss_vessel = loss_vessel + wss_loss

                loss_vessel = loss_vessel / len(train_graphs)
                loss_vessel.backward()
                
                cohort_loss_val += float(l_dict["loss"].item())

            optimizer.step()
            avg_loss = cohort_loss_val / len(train_graphs)

            if ep % 5 == 0 or ep == 1 or ep == args.epochs:
                print(f"  Epoch {ep:3d}/{args.epochs:3d} | Loss: {avg_loss:.5f}")

        elapsed = time.time() - start_time
        print(f"[OK] Trained {name} in {elapsed:.1f}s")

        final_eval_tr = evaluate_cohort(model, valid_train, GRAPH_DIR, phys_cfg, flow_source=args.flow, device=device)
        final_eval_sl = evaluate_cohort(model, sealed_anchors, GRAPH_DIR, phys_cfg, flow_source=args.flow, device=device)
        print(f"[After ] Train Deploy F1: {final_eval_tr['mean_f1']:.4f} | Sealed: {final_eval_sl['mean_f1']:.4f}")

        # Check for model collapse (F1 == 0.0)
        if final_eval_tr['mean_f1'] <= 0.001 and attempt < max_retries - 1:
            print(f"[WARN] {name} collapsed (Train F1 = {final_eval_tr['mean_f1']:.4f}). Halving LR and retrying...")
            current_lr *= 0.2
            continue

        return {
            "leg": name,
            "before_train_f1": base_eval_tr["mean_f1"],
            "before_sealed_f1": base_eval_sl["mean_f1"],
            "after_train_f1": final_eval_tr["mean_f1"],
            "after_sealed_f1": final_eval_sl["mean_f1"],
        }
        
    return {
        "leg": name,
        "before_train_f1": base_eval_tr["mean_f1"],
        "before_sealed_f1": base_eval_sl["mean_f1"],
        "after_train_f1": 0.0,
        "after_sealed_f1": 0.0,
        "failed": True
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10, help="Number of tuning epochs per leg")
    ap.add_argument("--lr", type=float, default=0.01, help="Learning rate for GNN correctors")
    ap.add_argument("--flow", choices=["pred", "gt"], default="pred", help="Flow field source")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Running on device: {device}")

    bio_cfg = BiochemConfig(phase="biochem")
    phys_cfg = PhysicsConfig(phase="biochem")
    
    valid_train = get_data(list(WALL_COHORT_V2_TRAIN), args.flow)
    sealed_anchors = list(WALL_COHORT_V2_GENERALIZATION)
    print(f"[i] Training cohort size: {len(valid_train)} full-horizon vessels")

    train_graphs = preload_graphs(valid_train, phys_cfg)
    
    results = []

    # Get optimal global params via the provider mechanism
    param_global = GlobalPhysicsParameters()
    model_dummy = DifferentiableWallModel(bio_cfg=bio_cfg, parameter_provider=param_global).to(device)
    
    # We evaluate it on the first graph to instantiate the parameters
    test_d = list(train_graphs.values())[0]["data"]
    _ = model_dummy(test_d, flow_source=args.flow, device=device)
    optimal_global_params = param_global.get_effective_parameters()

    def get_frozen_base_model():
        param_frozen = GlobalPhysicsParameters(
            init_da_scale=optimal_global_params["da_scale"],
            init_wake=optimal_global_params["wake"],
            init_lss=optimal_global_params["lss"],
            init_sgt_cgs=optimal_global_params["sgt_cgs"],
            init_tau_low=optimal_global_params["tau_low"],
            init_tau_sep=optimal_global_params["tau_sep"],
            init_relax=optimal_global_params["relax"],
            init_phi_temp=optimal_global_params["phi_temp"],
            trainable_keys=() # Freeze completely
        )
        return DifferentiableWallModel(bio_cfg=bio_cfg, parameter_provider=param_frozen).to(device)

    # --- LEG 1: GATv2 ---
    res1 = train_rung("1_GATv2", lambda: GATv2ResidualCorrector(get_frozen_base_model(), in_channels=18), train_graphs, valid_train, sealed_anchors, phys_cfg, args, device)
    results.append(res1)
    
    # --- LEG 2: Spatiotemporal ---
    res2 = train_rung("2_Spatiotemporal", lambda: SpatiotemporalResidualCorrector(get_frozen_base_model(), in_channels=18), train_graphs, valid_train, sealed_anchors, phys_cfg, args, device)
    results.append(res2)

    # --- LEG 3: EdgeConditioned ---
    res3 = train_rung("3_EdgeConditioned", lambda: EdgeConditionedResidualCorrector(get_frozen_base_model(), in_channels=18), train_graphs, valid_train, sealed_anchors, phys_cfg, args, device)
    results.append(res3)
    
    # --- LEG 4: StateInjected ---
    res4 = train_rung("4_StateInjected", lambda: StateInjectedResidualCorrector(get_frozen_base_model(), in_channels=18), train_graphs, valid_train, sealed_anchors, phys_cfg, args, device)
    results.append(res4)
    
    # --- LEG 5: MultiTask ---
    res5 = train_rung("5_MultiTask", lambda: MultiTaskResidualCorrector(get_frozen_base_model(), in_channels=18), train_graphs, valid_train, sealed_anchors, phys_cfg, args, device, use_multitask=True)
    results.append(res5)

    # --- LEG 6: MeshGraphNet ---
    res6 = train_rung("6_MeshGraphNet", lambda: MeshGraphNetCorrector(get_frozen_base_model(), in_channels=18), train_graphs, valid_train, sealed_anchors, phys_cfg, args, device)
    results.append(res6)
    
    # --- LEG 7: RGP (GAT + Transformer Hybrid) ---
    res7 = train_rung("7_RGP_Hybrid", lambda: RGPResidualCorrector(get_frozen_base_model(), in_channels=18), train_graphs, valid_train, sealed_anchors, phys_cfg, args, device)
    results.append(res7)

    # Save results
    save_path = Path("outputs/advanced_ml_sweep_results.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    
    print("\n" + "="*50)
    print(" ADVANCED SWEEP SUMMARY")
    print("="*50)
    for r in results:
        print(f" {r['leg']:25s} | Sealed F1: {r['before_sealed_f1']:.4f} -> {r['after_sealed_f1']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
