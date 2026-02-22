import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from pathlib import Path
from tqdm import tqdm

from src.phase1.physics.ginodeq import GINO_DEQ
from src.phase1.physics.physics_kernels import PhysicsKernels
from src.config import VesselConfig, PhysicsConfig
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.phase1.utils.samplers import StratifiedAnchorSampler
from src.phase1.utils.metrics import quantify_performance, validate_and_plot

def load_dataset():
    cfg = VesselConfig(tier="tier2")
    data_dir = cfg.graph_output_dir

    if not data_dir.exists():
        print(f"Directory not found: {data_dir}. Please generate Tier 2 data first.")
        return []

    file_list = sorted(list(data_dir.glob("vessel_*.pt")))
    dataset = []
    print(f"📂 Loading {len(file_list)} Tier 2 graphs...")
    for f in tqdm(file_list):
        data = torch.load(f, weights_only=False)
        dataset.append(data)
    return dataset

def train_tier2(epochs=50, lr=2e-5, warm_up_epochs=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GINO_DEQ(in_channels=11, out_channels=4, latent_dim=64, max_iters=15).to(device)

    tier1_path = Path("models/tier1_best_physics.pth")
    if tier1_path.exists():
        model.load_state_dict(torch.load(tier1_path, map_location=device, weights_only=True))
        print("✅ Successfully loaded Tier 1 foundational physics weights.")
    else:
        print("⚠️ Warning: Tier 1 weights not found. Training Tier 2 from scratch.")

    phys_cfg = PhysicsConfig(tier="tier2", re_target=150.0)
    kernels = PhysicsKernels(phys_cfg=phys_cfg)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    dataset = load_dataset()
    if not dataset: return

    train_size = int(0.9 * len(dataset))
    train_data, val_data = dataset[:train_size], dataset[train_size:]

    sampler = StratifiedAnchorSampler(train_data, batch_size=4)
    loader = DataLoader(train_data, batch_size=4, sampler=sampler)
    val_loader = DataLoader(val_data, batch_size=4, shuffle=False)

    best_phys_score = float('inf')
    best_loss = float('inf')
    Path("models").mkdir(exist_ok=True)

    for epoch in range(epochs):
        model.train()
        physics_active = epoch >= warm_up_epochs

        lambda_phys = min(1.0, max(0.0, (epoch - warm_up_epochs) / 20.0))
        lambda_rheo = min(10.0, epoch / 15.0)

        total_loss_epoch = 0.0

        current_solver = "picard" if epoch < 5 else "anderson"
        current_beta = 0.8

        pbar = tqdm(loader, desc=f"Tier 2 Epoch {epoch:02d} [Re={phys_cfg.re_target}]")
        for batch_idx, data in enumerate(pbar):
            data = data.to(device)
            optimizer.zero_grad()
            pred = model(data, solver=current_solver, anderson_beta=current_beta)

            l_data = torch.tensor(0.0, device=device)
            if hasattr(data, 'is_anchor'):
                node_is_anchor = data.is_anchor[data.batch]
                if node_is_anchor.sum() > 0:
                    # Use ground truth viscosity losses for tier2
                    l_data = F.mse_loss(pred[node_is_anchor, :4], data.y[node_is_anchor, :4])

            l_ns = kernels.navier_stokes_residual(pred, data)
            l_bc = kernels.boundary_condition_loss(pred, data)
            l_io = kernels.inlet_outlet_loss(pred, data)
            l_rheo = kernels.rheology_loss(pred, data)

            row, col = data.edge_index

            loss = (lambda_phys * l_ns + 5 * l_bc + 5 * l_io) + \
                   (5.0 * l_data) + (lambda_rheo * l_rheo)

            # --- NaN Check ---
            if torch.isnan(loss):
                print(f"\n⚠️ NaN detected in loss at epoch {epoch}, batch {batch_idx}! Skipping batch.")
                continue

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss_epoch += loss.item()

            # --- Enhanced Postfix Tracking ---
            pbar.set_postfix({
                "L_tot": f"{loss.item():.3f}",
                "L_ns": f"{l_ns.item():.3f}",
                "L_rh": f"{l_rheo.item():.3f}",
                "w_rh": f"{lambda_rheo:.2f}",
                "|g|": f"{grad_norm:.2f}", # Track gradient norm
                "LR": f"{optimizer.param_groups[0]['lr']:.2e}"
            })

        scheduler.step()

        if epoch % 2 == 0:
            scores = quantify_performance(model, val_loader, kernels, device, tier="tier2")
            print(
                f"\n📊 [Validation] Rel L2: {scores.get('rel_l2', 0):.4f} | "
                f"Div: {scores.get('continuity', 0):.3e} | "
                f"Rheo: {scores.get('rheology', 0):.3e} | "
                f"Shear MSE: {scores.get('shear_mse', 0):.3e}"
            )

            phys_score = scores.get('rel_l2', 0) + scores.get('continuity', 0) + scores.get('rheology', 0)
            if phys_score < best_phys_score and physics_active:
                best_phys_score = phys_score
                torch.save(model.state_dict(), "models/tier2_best_physics.pth")
                print("⭐ Saved Best Physics Model")

        avg_loss = total_loss_epoch / len(loader)
        if avg_loss < best_loss and physics_active:
            best_loss = avg_loss
            torch.save(model.state_dict(), "models/tier2_best_loss.pth")

        if epoch % 5 == 0:
            validate_and_plot(model, val_data[0], epoch, device, tier="tier2")

    torch.save(model.state_dict(), "models/tier2_final.pth")
    print(f"Tier 2 Training Complete. Best Physical Score: {best_phys_score:.4f} | Best Loss: {best_loss:.4f}")

if __name__ == "__main__":
    train_tier2()