"""Tune global physical parameters of the differentiable wall model via gradient descent.

Level 1.1: Learns the optimal global physics parameters (da_scale, wake, lss, sgt, tau_low, relax)
across the training cohort using PyTorch autograd and soft Dice/F1 loss.

Usage:
    python scripts/tune_differentiable_wall_model.py --epochs 25 --lr 0.05 --flow pred
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.biochem_gnn.mat_growth_simple import (  # noqa: E402
    WALL_COHORT_V2_GENERALIZATION,
    WALL_COHORT_V2_TRAIN,
)
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.species_pushforward_continuous import resolve_deploy_eval_time_index  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.differentiable_wall_model.differentiable_ode import DifferentiableWallModel  # noqa: E402
from src.differentiable_wall_model.evaluation import evaluate_cohort, evaluate_vessel  # noqa: E402
from src.differentiable_wall_model.losses import CombinedWallClotLoss  # noqa: E402
from src.differentiable_wall_model.parameters import GlobalPhysicsParameters  # noqa: E402
from src.differentiable_wall_model.local_gnn import LocalPhysicsGNN  # noqa: E402
from src.differentiable_wall_model.growth_gno import GrowthGNO  # noqa: E402
from src.differentiable_wall_model.chemical_estimator import ChemicalStateEstimator  # noqa: E402

GRAPH_DIR = Path("data/processed/graphs_biochem_anchors")


def main() -> int:
    ap = argparse.ArgumentParser(description="Tune differentiable wall model parameters")
    ap.add_argument("--epochs", type=int, default=20, help="Number of tuning epochs")
    ap.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    ap.add_argument("--flow", choices=["pred", "gt"], default="pred", help="Flow field source")
    ap.add_argument("--anchors", type=str, default="", help="Optional comma-separated train anchors")
    ap.add_argument("--eval-sealed", action="store_true", default=True, help="Evaluate on sealed cohort")
    ap.add_argument("--save", type=str, default="outputs/tuned_physics_params.json", help="Path to save tuned params")
    ap.add_argument("--model-type", choices=["global", "local"], default="global", help="Use global scalars or local GNN for physics parameters")
    ap.add_argument("--use-growth-gno", action="store_true", help="Use Level 2 Physics-Gated Neural Operator for growth")
    ap.add_argument("--use-chem-estimator", action="store_true", help="Use Level 3 Deep Chemical State Estimator")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Running on device: {device}")

    bio_cfg = BiochemConfig(phase="biochem")
    phys_cfg = PhysicsConfig(phase="biochem")

    train_anchors = (
        [a.strip() for a in args.anchors.split(",") if a.strip()]
        if args.anchors
        else list(WALL_COHORT_V2_TRAIN)
    )

    # Filter available train graphs with T >= 150
    valid_train = []
    for a in train_anchors:
        p = GRAPH_DIR / f"{a}.pt"
        if p.exists():
            d = torch.load(p, map_location="cpu", weights_only=False)
            if int(d.y.shape[0]) >= 150:
                if args.flow == "pred" and getattr(d, "u0_pred", None) is None:
                    continue
                valid_train.append(a)

    print(f"[i] Training cohort size: {len(valid_train)} full-horizon vessels")
    if not valid_train:
        print("[ERR] No valid training vessels found.")
        return 1

    if args.model_type == "global":
        param_mod = GlobalPhysicsParameters()
        print("\n--- Initial Parameters ---")
        for k, v in param_mod.get_effective_parameters().items():
            print(f"  {k:12s}: {v:10.4f}")
    else:
        # Load optimal 1.1 parameters for initialization if available
        init_params = None
        params_path = Path("outputs/tuned_physics_params.json")
        if params_path.exists():
            print(f"[i] Initializing local GNN with global parameters from {params_path}")
            with open(params_path, "r") as f:
                d = json.load(f)
                init_params = d.get("tuned_parameters")
        # 18 channels in KINE_X_SCHEMA
        param_mod = LocalPhysicsGNN(in_channels=18, initial_params=init_params)
        print("\n--- Initializing LocalPhysicsGNN ---")

    gno_mod = GrowthGNO(in_channels=18).to(device) if args.use_growth_gno else None
    chem_mod = ChemicalStateEstimator(in_channels=18).to(device) if args.use_chem_estimator else None

    if gno_mod:
        print("[i] Level 2: GrowthGNO activated")
    if chem_mod:
        print("[i] Level 3: ChemicalStateEstimator activated")

    model = DifferentiableWallModel(bio_cfg=bio_cfg, parameter_provider=param_mod, diffusion_module=gno_mod).to(device)
    model.chem_estimator = chem_mod

    # Baseline evaluation before tuning
    print("\n[i] Evaluating initial baseline...")
    base_eval_tr = evaluate_cohort(model, valid_train, GRAPH_DIR, phys_cfg, flow_source=args.flow, device=device)
    print(f"  Initial Train Mean Deploy Score : {base_eval_tr['mean_score']:.4f} (F1: {base_eval_tr['mean_f1']:.4f})")

    if args.eval_sealed:
        sealed_anchors = list(WALL_COHORT_V2_GENERALIZATION)
        base_eval_sl = evaluate_cohort(model, sealed_anchors, GRAPH_DIR, phys_cfg, flow_source=args.flow, device=device)
        print(f"  Initial Sealed Mean Deploy Score: {base_eval_sl['mean_score']:.4f} (F1: {base_eval_sl['mean_f1']:.4f})")

    loss_fn = CombinedWallClotLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    print(f"\n--- Starting Parameter Optimization ({args.epochs} epochs) ---")
    start_time = time.time()

    # Preload training graphs into CPU memory to stay within 4GB VRAM
    train_graphs = {}
    for a in valid_train:
        d = torch.load(GRAPH_DIR / f"{a}.pt", map_location="cpu", weights_only=False)
        t_eval = resolve_deploy_eval_time_index(int(d.y.shape[0]))
        gt_phi = gt_clot_phi_at_time(d, t_eval, phys_cfg, device=torch.device("cpu")).reshape(-1)
        wall = d.mask_wall.reshape(-1).bool()
        train_graphs[a] = {"data": d, "gt_phi": gt_phi * wall.float(), "wall": wall}

    for ep in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        cohort_loss_val = 0.0

        for a, pack in train_graphs.items():
            d = pack["data"]
            gt = pack["gt_phi"].to(device)
            wall = pack["wall"].to(device)
            out = model(d, flow_source=args.flow, device=device)
            prob = out["prob_clot"]
            l_dict = loss_fn(prob, gt, wall)
            loss_vessel = l_dict["loss"] / len(train_graphs)
            loss_vessel.backward()
            cohort_loss_val += float(l_dict["loss"].item())

        optimizer.step()
        avg_loss = cohort_loss_val / len(train_graphs)

        if ep % 5 == 0 or ep == 1 or ep == args.epochs:
            if args.model_type == "global":
                eff = param_mod.get_effective_parameters()
                print(f"Epoch {ep:3d}/{args.epochs:3d} | Loss: {avg_loss:.5f} | "
                      f"da: {eff['da_scale']:5.2f} | wake: {eff['wake']:5.2f} | lss: {eff['lss']:5.2f} | "
                      f"sgt: {eff['sgt_cgs']:6.1f} | tau: {eff['tau_low']:4.2f} | relax: {eff['relax']:4.2f}")
            else:
                print(f"Epoch {ep:3d}/{args.epochs:3d} | Loss: {avg_loss:.5f} (Local GNN)")

    elapsed = time.time() - start_time
    print(f"\n[OK] Optimization completed in {elapsed:.1f}s")

    if args.model_type == "global":
        print("\n--- Tuned Physical Parameters ---")
        tuned_params = param_mod.get_effective_parameters()
        for k, v in tuned_params.items():
            print(f"  {k:12s}: {v:10.4f}")
    else:
        tuned_params = {"model": "local_gnn"}
        print("\n--- Tuned LocalPhysicsGNN ---")

    # Final Evaluation
    print("\n--- Final Evaluation After Tuning ---")
    final_eval_tr = evaluate_cohort(model, valid_train, GRAPH_DIR, phys_cfg, flow_source=args.flow, device=device)
    print(f"  Final Train Mean Deploy Score  : {final_eval_tr['mean_score']:.4f} (F1: {final_eval_tr['mean_f1']:.4f}) "
          f"[Delta: {final_eval_tr['mean_score'] - base_eval_tr['mean_score']:+.4f}]")

    if args.eval_sealed:
        final_eval_sl = evaluate_cohort(model, sealed_anchors, GRAPH_DIR, phys_cfg, flow_source=args.flow, device=device)
        print(f"  Final Sealed Mean Deploy Score : {final_eval_sl['mean_score']:.4f} (F1: {final_eval_sl['mean_f1']:.4f}) "
              f"[Delta: {final_eval_sl['mean_score'] - base_eval_sl['mean_score']:+.4f}]")

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            "tuned_parameters": tuned_params,
            "train_score_before": base_eval_tr["mean_score"],
            "train_score_after": final_eval_tr["mean_score"],
        }
        if args.eval_sealed:
            save_dict["sealed_score_before"] = base_eval_sl["mean_score"]
            save_dict["sealed_score_after"] = final_eval_sl["mean_score"]
        save_path.write_text(json.dumps(save_dict, indent=2), encoding="utf-8")
        print(f"\n[OK] Saved tuned parameters to {save_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
