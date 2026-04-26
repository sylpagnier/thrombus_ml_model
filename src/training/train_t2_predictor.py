import argparse
import math
import os
import sys
if sys.platform != "win32":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import random
from pathlib import Path

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.config import VesselConfig, PhysicsConfig
from src.architecture.ginodeq import GINO_DEQ
from src.core_physics.physics_kernels import PhysicsKernels
from src.utils.metrics import DynamicLossWeighter, quantify_performance
from src.utils.kinematics_physics_terms import compute_kinematics_physics_terms
from src.utils.samplers import StratifiedAnchorSampler
from src.utils.paths import stage_a_dir, resolve_checkpoint
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau

TIER2_VAL_COMPOSITE_CONTINUITY_SCALE_DEFAULT = 100.0
TIER2_VAL_COMPOSITE_RHEOLOGY_SCALE_DEFAULT = 1.0
TIER2_TRAIN_COUPLED_RHEOLOGY_SCALE_DEFAULT = 1.0


def _env_truthy(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _load_tier1_bootstrap(model: GINO_DEQ, tier1_path: Path, device: str) -> bool:
    if not tier1_path.is_file():
        print(f"⚠️ Tier 1 weights not found at {tier1_path}. Random init will be used.")
        return False
    state_dict = torch.load(tier1_path, map_location=device, weights_only=True)
    encoder_key = next(
        (k for k in state_dict.keys() if k.startswith("encoder.") and k.endswith(".weight")),
        None,
    )
    if encoder_key:
        model_params = dict(model.named_parameters())
        if encoder_key in model_params:
            w_t1 = state_dict[encoder_key]
            w_t2 = model_params[encoder_key]
            if w_t1.shape != w_t2.shape:
                new_w = torch.zeros_like(w_t2)
                n_out = min(w_t1.shape[0], w_t2.shape[0])
                n_in = min(w_t1.shape[1], w_t2.shape[1])
                new_w[:n_out, :n_in] = w_t1[:n_out, :n_in]
                state_dict[encoder_key] = new_w
    model.load_state_dict(state_dict, strict=False)
    print(f"✅ Loaded Tier 1 bootstrap weights from {tier1_path.name}")
    return True


def _assert_tier2_train_split(train_data: list, val_data: list) -> None:
    if not train_data or not val_data:
        raise ValueError("Train or val data is empty after split.")
    if sum(1 for d in train_data if d.is_anchor.any().item()) == 0:
        raise ValueError("Training split has no anchor (COMSOL-labeled) graphs.")


def load_dataset_for_n(current_n: float):
    cfg = VesselConfig(tier="tier2")
    n_subdir = f"n_{current_n:.3f}"
    data_dir = cfg.graph_output_dir / n_subdir

    if not data_dir.exists():
        return []
    file_list = sorted(list(data_dir.glob("vessel_*.pt")))

    dataset = []
    print(f"📂 Loading {len(file_list)} Tier 2 graphs for n={current_n:.3f} from {data_dir.name}...")
    for f in tqdm(file_list, leave=False):
        dataset.append(torch.load(f, weights_only=False))
    return dataset


def setup_coupled_phase(model, loss_weighter, base_lr=1e-4):
    # Ensure all parameters are unfrozen
    for param in model.parameters():
        param.requires_grad = True

    # Pass all model parameters uniformly, just like Tier 1
    opt_params = list(model.parameters()) + list(loss_weighter.parameters())

    return optim.AdamW(opt_params, lr=base_lr, weight_decay=1e-5)


def _tier2_dynamic_loss_weighter(device: str, mom_precision_floor: float, mom_weight_cap: float) -> DynamicLossWeighter:
    floor = max(float(mom_precision_floor), 1e-6)
    cap = max(float(mom_weight_cap), 1.0)
    return DynamicLossWeighter(
        num_losses=2,
        min_log_var=[-math.log(cap), -math.log(50.0)],
        max_log_var=[-math.log(floor), 10.0],
    ).to(device)


def compute_step_loss(
    model, data, kernels, loss_weighter, current_solver, device, carreau_n,
    lambda_phys: float, tier2_kine_p_weight: float = 1.0, coupled_io_scale: float = 6.0,
    train_rheology_scale: float = TIER2_TRAIN_COUPLED_RHEOLOGY_SCALE_DEFAULT
):
    out = model(data, solver=current_solver, anderson_beta=0.8, anderson_warmup_iters=12)
    pred, jac_loss = out if isinstance(out, tuple) else (out, torch.tensor(0.0, device=device))

    terms = compute_kinematics_physics_terms(
        pred, data, kernels, tier="tier2",
        tier2_distillation=False, carreau_n=carreau_n, tier2_kine_p_weight=tier2_kine_p_weight,
    )

    # Pass raw PDE losses to Kendall weighting, then scale by curriculum lambda_phys.
    raw_pde_losses = [terms["l_mom"], terms["l_cont"]]
    weighted_pdes = loss_weighter(raw_pde_losses)

    loss = (
        (lambda_phys * weighted_pdes)
        + (lambda_phys * float(train_rheology_scale) * terms["l_rheo"])
        + (500.0 * terms["l_data_kine"])
        + (500.0 * terms["l_data_mu"])
        + (5.0 * terms["l_bc"])
        + (float(coupled_io_scale) * terms["l_io"])
        + (10.0 * terms["l_wss"])
        + (0.1 * jac_loss)
    )

    return loss, {
        "L_mom": terms["l_mom"].item(), "L_cont": terms["l_cont"].item(),
        "L_rh": terms["l_rheo"].item(), "L_jac": jac_loss.item(), "L_wss": terms["l_wss"].item()
    }


def train_t2_predictor(epochs=80, lr=1e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device being used:", device)

    phys_cfg = PhysicsConfig(tier="tier2")
    kernels = PhysicsKernels(phys_cfg=phys_cfg)
    model = GINO_DEQ(
        in_channels=15, out_channels=5, latent_dim=256, max_iters=25,
        num_fourier_freqs=16, phys_cfg=phys_cfg, activation_fn="silu",
        fourier_base=1.5, use_hard_bcs=True, num_global_tokens=16,
        use_siren_decoder=True, use_width_priors=True,
    ).to(device)

    model_dir = stage_a_dir()
    tier1_path = resolve_checkpoint("a", "tier1_best_physics.pth")
    latest_ckpt_save = model_dir / "tier2_latest_checkpoint.pth"
    resume_training = _env_truthy("TIER2_RESUME")

    loss_weighter = _tier2_dynamic_loss_weighter(device, 0.8, 20.0)

    target_n = phys_cfg.n
    raw_steps = os.environ.get("TIER2_CONTINUATION_STEPS", "0.8,0.6")
    continuation_steps = [float(x.strip()) for x in raw_steps.split(",") if x.strip()]
    n_sequence = []
    for n_val in continuation_steps + [target_n]:
        if not any(abs(n_val - seen) < 1e-8 for seen in n_sequence):
            n_sequence.append(float(n_val))

    # Epoch allocation: ~20% for intermediate stages, rest for final target_n stage
    stage_epochs_alloc = []
    for i in range(len(n_sequence) - 1):
        stage_epochs_alloc.append(max(5, int(epochs * 0.2)))
    stage_epochs_alloc.append(epochs - sum(stage_epochs_alloc))

    def get_stage_info(epoch: int):
        cum_epochs = 0
        for idx, (n_val, alloc) in enumerate(zip(n_sequence, stage_epochs_alloc)):
            if epoch < cum_epochs + alloc:
                return idx, n_val, cum_epochs, alloc
            cum_epochs += alloc
        return len(n_sequence)-1, n_sequence[-1], sum(stage_epochs_alloc[:-1]), stage_epochs_alloc[-1]

    best_val_composite_loss, best_loss = float("inf"), float("inf")
    start_epoch = 0
    resume_optimizer_state = None
    resume_scheduler_state = None
    resume_plateau_scheduler_state = None
    resume_loss_weighter_state = None

    if resume_training and latest_ckpt_save.is_file():
        ckpt = torch.load(latest_ckpt_save, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_loss = float(ckpt.get("best_loss", best_loss))
        best_val_composite_loss = float(ckpt.get("best_val_composite_loss", best_val_composite_loss))
        resume_optimizer_state = ckpt.get("optimizer_state_dict")
        resume_scheduler_state = ckpt.get("scheduler_state_dict")
        resume_plateau_scheduler_state = ckpt.get("plateau_scheduler_state_dict")
        resume_loss_weighter_state = ckpt.get("loss_weighter_state_dict")
        if resume_loss_weighter_state is not None:
            try:
                loss_weighter.load_state_dict(resume_loss_weighter_state)
            except RuntimeError as err:
                print(f"⚠️ Could not restore loss weighter state: {err}. Continuing with fresh weights.")
        print(f"✅ Tier 2 training resume complete at epoch {start_epoch}")
    else:
        _load_tier1_bootstrap(model, tier1_path, device)

    # State tracking variables
    current_stage_idx = -1
    optimizer = None
    scheduler = None
    plateau_scheduler = None
    current_loader = None
    current_val_loader = None
    model.decouple_rheology = False

    for epoch in range(start_epoch, epochs):
        stage_idx, current_n, stage_start_epoch, alloc = get_stage_info(epoch)
        epochs_in_stage = epoch - stage_start_epoch

        # Ramp lambda_phys from 0 to 1 over the first 5 epochs of the current stage
        lambda_phys = min(1.0, max(0.0, epochs_in_stage / 5.0))

        # Re-initialize Optimizer and Schedulers upon entering a new n-stage
        if stage_idx != current_stage_idx:
            print(f"\n🌊 NEW TIER 2 STAGE: n={current_n:.3f} | Epochs {stage_start_epoch} to {stage_start_epoch+alloc-1}")
            current_stage_idx = stage_idx

            # Reset stage-local bests, except when resuming inside the same stage.
            if not (resume_training and epoch == start_epoch):
                best_loss = float("inf")
                best_val_composite_loss = float("inf")

            optimizer = setup_coupled_phase(model, loss_weighter, base_lr=lr)
            if hasattr(loss_weighter, "log_vars"):
                loss_weighter.log_vars.data.zero_()  # Reset PDE precisions

            scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=max(1, 5))
            plateau_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, threshold=5e-4, min_lr=5e-6)

            # Load only current stage data to avoid holding all n-datasets in RAM.
            ds = load_dataset_for_n(current_n)
            anchors = [d for d in ds if d.is_anchor.any().item()]
            physics = [d for d in ds if not d.is_anchor.any().item()]
            split_rng = random.Random(42)
            split_rng.shuffle(anchors)
            split_rng.shuffle(physics)
            split_idx_a, split_idx_p = int(0.9 * len(anchors)), int(0.9 * len(physics))
            train_data_n = anchors[:split_idx_a] + physics[:split_idx_p]
            val_data_n = anchors[split_idx_a:] + physics[split_idx_p:]
            _assert_tier2_train_split(train_data_n, val_data_n)

            sampler_n = StratifiedAnchorSampler(train_data_n, batch_size=1)
            sampler_n.set_warmup_mode(False)
            current_loader = DataLoader(train_data_n, batch_size=1, sampler=sampler_n)
            current_val_loader = DataLoader(val_data_n, batch_size=1, shuffle=False)

            if epoch == start_epoch and resume_optimizer_state is not None:
                try:
                    optimizer.load_state_dict(resume_optimizer_state)
                    if resume_scheduler_state is not None:
                        scheduler.load_state_dict(resume_scheduler_state)
                    if resume_plateau_scheduler_state is not None:
                        plateau_scheduler.load_state_dict(resume_plateau_scheduler_state)
                    print("🔁 Restored optimizer/scheduler states from checkpoint.")
                except Exception as err:
                    print(f"⚠️ Failed to restore optimizer/scheduler states ({err}). Continuing with fresh states.")
                finally:
                    resume_optimizer_state = None
                    resume_scheduler_state = None
                    resume_plateau_scheduler_state = None

        model.train()
        total_loss_epoch = 0.0
        accumulation_steps = 8
        # Freeze loss-weighter updates while lambda_phys is still ramping.
        loss_weighter.requires_grad_(lambda_phys >= 1.0)
        optimizer.zero_grad()

        pbar = tqdm(current_loader, desc=f"Tier 2 Ep {epoch:02d} [n={current_n:.3f}] (λ={lambda_phys:.2f})")
        for batch_idx, data in enumerate(pbar):
            data = data.to(device)

            loss, metrics = compute_step_loss(
                model, data, kernels, loss_weighter, "anderson", device, current_n, lambda_phys=lambda_phys
            )
            loss = loss / accumulation_steps

            if torch.isnan(loss):
                optimizer.zero_grad()
                continue

            loss.backward()

            if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(current_loader)):
                clip_params = [p for g in optimizer.param_groups for p in g["params"]]
                torch.nn.utils.clip_grad_norm_(clip_params, max_norm=0.5)
                optimizer.step()
                optimizer.zero_grad()

            total_loss_epoch += (loss.item() * accumulation_steps)

            pbar.set_postfix({
                "L_tot": f"{(loss.item() * accumulation_steps):.3f}",
                "L_mom": f"{metrics['L_mom']:.3f}",
                "L_cont": f"{metrics['L_cont']:.3f}",
                "L_rh": f"{metrics['L_rh']:.3f}"
            })

        if scheduler.last_epoch < scheduler.total_iters:
            scheduler.step()

        avg_loss = total_loss_epoch / max(1, len(current_loader))
        if avg_loss < best_loss and lambda_phys >= 1.0:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_dir / "tier2_best_loss.pth")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "plateau_scheduler_state_dict": plateau_scheduler.state_dict(),
            "loss_weighter_state_dict": loss_weighter.state_dict(),
            "best_loss": best_loss,
            "best_val_composite_loss": best_val_composite_loss,
        }, latest_ckpt_save)

        # Validation every 2 epochs
        if epoch % 2 == 0:
            scores = quantify_performance(model, current_val_loader, kernels, device, tier="tier2", solver="anderson")
            val_comp = float(
                scores.get("rel_l2", 0)
                + TIER2_VAL_COMPOSITE_CONTINUITY_SCALE_DEFAULT * scores.get("continuity", 0)
                + TIER2_VAL_COMPOSITE_RHEOLOGY_SCALE_DEFAULT * scores.get("rheology", 0)
            )

            print(f"\n📊 [Validation n={current_n:.3f}] Rel L2: {scores.get('rel_l2', float('nan')):.4f}")
            print(f"   |∇·u| mean: {scores.get('continuity', float('nan')):.3e} | Rheo res: {scores.get('rheology', float('nan')):.3e}")
            print(f"   Val composite: {val_comp:.4f}")

            if lambda_phys >= 1.0:
                plateau_scheduler.step(val_comp)
                if val_comp < best_val_composite_loss:
                    best_val_composite_loss = val_comp
                    torch.save(model.state_dict(), model_dir / "tier2_best_physics.pth")
                    print(f"⭐ Saved Best Physics Model")

    torch.save(model.state_dict(), model_dir / "tier2_final.pth")
    print(f"Tier 2 Training Complete. Best val composite: {best_val_composite_loss:.4f} | Best Loss: {best_loss:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--resume", action="store_true")
    p.add_argument("--new", action="store_true")
    args = p.parse_args()
    if args.resume and args.new:
        raise ValueError("Use only one of --resume or --new.")
    if args.resume:
        resume_enabled = True
    elif args.new:
        resume_enabled = False
    else:
        while True:
            raw = input("Training mode [1=resume / 2=start new] [1]: ").strip()
            if raw in ("", "1"):
                resume_enabled = True
                break
            if raw == "2":
                resume_enabled = False
                break
            print("  Enter 1 or 2.")

    os.environ["TIER2_RESUME"] = "1" if resume_enabled else "0"
    print("🔄 Resuming Tier 2 from latest checkpoint." if resume_enabled else "🆕 Starting a new Tier 2 run.")
    train_t2_predictor()