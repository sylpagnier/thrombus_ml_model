import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from pathlib import Path

# Project specific imports based on your file structure
from src.utils.paths import get_project_root
from src.phase1.data_gen.vessel_generator import VesselGenerator
from src.phase1.data_gen.mesh_to_graph import MeshToGraphComplete
from src.phase1.physics.ginodeq import GINO_DEQ
from src.phase2.ginodeq_tier3 import GINO_DEQ_Tier3
from src.config import PhysicsConfig

# Standard channel indices across all models for kinematics
_CHANNEL = dict(u=0, v=1, p=2, mu_eff=3)


def _plot_field(fig, ax, pos, val, title, cmap, vmin=None, vmax=None):
    """
    Plot a scalar field on an unstructured mesh using tripcolor.
    Includes the dynamic mask to remove artificial convex-hull triangles.
    """
    triang = mtri.Triangulation(pos[:, 0], pos[:, 1])

    # Mask triangles that have abnormally long edges (convex hull artifacts)
    tri_pts = pos[triang.triangles]
    d1 = np.sum((tri_pts[:, 0, :] - tri_pts[:, 1, :]) ** 2, axis=1)
    d2 = np.sum((tri_pts[:, 1, :] - tri_pts[:, 2, :]) ** 2, axis=1)
    d3 = np.sum((tri_pts[:, 2, :] - tri_pts[:, 0, :]) ** 2, axis=1)
    max_edge_sq = np.max(np.vstack([d1, d2, d3]), axis=0)

    mask = max_edge_sq > (np.median(max_edge_sq) * 10.0)
    triang.set_mask(mask)

    tc = ax.tripcolor(triang, val, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=12)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.colorbar(tc, ax=ax, fraction=0.046, pad=0.04)


def run_tier_comparison():
    root = get_project_root()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")

    # ------------------------------------------------------------------
    # 1. Setup Directories
    # ------------------------------------------------------------------
    test_dir = root / "data" / "tier_comparison_test"
    raw_dir = test_dir / "raw_meshes"
    graph_t1_dir = test_dir / "graphs_t1"
    graph_t2_dir = test_dir / "graphs_t2"
    graph_t3_dir = test_dir / "graphs_t3"

    for d in [raw_dir, graph_t1_dir, graph_t2_dir, graph_t3_dir]:
        d.mkdir(parents=True, exist_ok=True)
        # Clear old files
        for f in d.glob("*"): f.unlink()

    # ------------------------------------------------------------------
    # 2. Generate a Single Shared Synthetic Geometry
    # ------------------------------------------------------------------
    print("\n📐 Generating 1 complex synthetic vessel for the comparison...")
    vg = VesselGenerator(tier="tier3", output_dir=raw_dir)  # Use tier 3 to ensure rich geometry
    vg.run_pipeline(n=1, level=1, num_workers=1, seed=42)

    # ------------------------------------------------------------------
    # 3. Convert Mesh to specific Graphs for each Tier
    # ------------------------------------------------------------------
    print("\n🕸️ Converting mesh to graphs for each tier's specific channel requirements...")

    # Tier 1 (15 in_channels)
    mg1 = MeshToGraphComplete(tier="tier1", raw_dir=raw_dir, label_dir=raw_dir, proc_dir=graph_t1_dir)
    mg1.run()

    # Tier 2 (15 in_channels)
    mg2 = MeshToGraphComplete(tier="tier2", raw_dir=raw_dir, label_dir=raw_dir, proc_dir=graph_t2_dir)
    mg2.run()

    # Tier 3 (12 in_channels)
    mg3 = MeshToGraphComplete(tier="tier3", raw_dir=raw_dir, label_dir=raw_dir, proc_dir=graph_t3_dir)
    mg3.run()

    # Load the specific graphs
    try:
        data_t1 = torch.load(list(graph_t1_dir.glob("*.pt"))[0], weights_only=False).to(device)
        data_t2 = torch.load(list(graph_t2_dir.glob("*.pt"))[0], weights_only=False).to(device)
        data_t3 = torch.load(list(graph_t3_dir.glob("*.pt"))[0], weights_only=False).to(device)
        pos = data_t3.x[:, :2].cpu().numpy()  # Position is the same across all
    except IndexError:
        print("⚠️ Failed to generate or load graph files.")
        return

    # ------------------------------------------------------------------
    # 4. Load Models
    # ------------------------------------------------------------------
    print("\n🧠 Loading trained models...")
    model_dir = root / "models"

    # Tier 1 Setup
    model_t1 = GINO_DEQ(in_channels=15, out_channels=5, latent_dim=64, max_iters=15).to(device)
    model_t1.load_state_dict(torch.load(model_dir / "tier1_best_physics.pth", map_location=device, weights_only=True))
    model_t1.eval()

    # Tier 2 Setup (needs proper Non-Newtonian boundaries from config)
    phys_cfg_t2 = PhysicsConfig(tier="tier2")
    model_t2 = GINO_DEQ(
        in_channels=15, out_channels=5, latent_dim=64, max_iters=15,
        mu_inf_nd=(phys_cfg_t2.mu_inf / phys_cfg_t2.mu_ref),
        mu_0_nd=(phys_cfg_t2.mu_0 / phys_cfg_t2.mu_ref)
    ).to(device)
    model_t2.load_state_dict(torch.load(model_dir / "tier2_best_physics.pth", map_location=device, weights_only=True))
    model_t2.eval()

    # Tier 3 Setup
    model_t3 = GINO_DEQ_Tier3(in_channels=12, latent_dim=64, max_outer_iters=3, max_inner_iters=15).to(device)
    model_t3.load_state_dict(torch.load(model_dir / "tier3_best_bio.pth", map_location=device, weights_only=True))
    model_t3.eval()

    # ------------------------------------------------------------------
    # 5. Inference
    # ------------------------------------------------------------------
    print("\n🔮 Running Inference across all Tiers...")
    with torch.no_grad():
        pred_t1 = model_t1(data_t1) if isinstance(model_t1(data_t1), tuple) else model_t1(data_t1)
        pred_t2 = model_t2(data_t2) if isinstance(model_t2(data_t2), tuple) else model_t2(data_t2)
        pred_t3 = model_t3(data_t3) if isinstance(model_t3(data_t3), tuple) else model_t3(data_t3)

    pred_t1_np = pred_t1.cpu().numpy()
    pred_t2_np = pred_t2.cpu().numpy()
    pred_t3_np = pred_t3.cpu().numpy()

    # Extract Fields
    def get_kinematics(pred_np):
        u = pred_np[:, _CHANNEL['u']]
        v = pred_np[:, _CHANNEL['v']]
        vel_mag = np.sqrt(u ** 2 + v ** 2)
        pressure = pred_np[:, _CHANNEL['p']]
        viscosity = pred_np[:, _CHANNEL['mu_eff']]
        return vel_mag, pressure, viscosity

    vel_1, p_1, mu_1 = get_kinematics(pred_t1_np)
    vel_2, p_2, mu_2 = get_kinematics(pred_t2_np)
    vel_3, p_3, mu_3 = get_kinematics(pred_t3_np)

    # Determine global min/max for fair colorbar comparisons across columns
    vel_min, vel_max = min(vel_1.min(), vel_2.min(), vel_3.min()), max(vel_1.max(), vel_2.max(), vel_3.max())
    p_min, p_max = min(p_1.min(), p_2.min(), p_3.min()), max(p_1.max(), p_2.max(), p_3.max())
    mu_min, mu_max = min(mu_1.min(), mu_2.min(), mu_3.min()), max(mu_1.max(), mu_2.max(), mu_3.max())

    # ------------------------------------------------------------------
    # 6. Plotting
    # ------------------------------------------------------------------
    print("🎨 Generating Comparison Plots...")

    # --- FIGURE 1: Kinematic Comparison ---
    fig1, axes1 = plt.subplots(3, 3, figsize=(20, 14))
    fig1.suptitle("Kinematic Evolution: Tier 1 (Newtonian) vs Tier 2 (Non-Newtonian) vs Tier 3 (Bio-Coupled)",
                  fontsize=18, y=0.98)

    columns = ["Tier 1 (Newtonian)", "Tier 2 (Carreau Rheology)", "Tier 3 (Coupled Biochemistry)"]
    for ax, col in zip(axes1[0], columns):
        ax.set_title(col, fontsize=14, fontweight='bold', pad=20)

    # Row 1: Velocity
    _plot_field(fig1, axes1[0, 0], pos, vel_1, "Velocity Mag (m/s)", 'jet', vmin=vel_min, vmax=vel_max)
    _plot_field(fig1, axes1[0, 1], pos, vel_2, "Velocity Mag (m/s)", 'jet', vmin=vel_min, vmax=vel_max)
    _plot_field(fig1, axes1[0, 2], pos, vel_3, "Velocity Mag (m/s)", 'jet', vmin=vel_min, vmax=vel_max)

    # Row 2: Pressure
    _plot_field(fig1, axes1[1, 0], pos, p_1, "Pressure (Pa)", 'coolwarm', vmin=p_min, vmax=p_max)
    _plot_field(fig1, axes1[1, 1], pos, p_2, "Pressure (Pa)", 'coolwarm', vmin=p_min, vmax=p_max)
    _plot_field(fig1, axes1[1, 2], pos, p_3, "Pressure (Pa)", 'coolwarm', vmin=p_min, vmax=p_max)

    # Row 3: Viscosity
    _plot_field(fig1, axes1[2, 0], pos, mu_1, r"Eff. Viscosity ($\mu_{eff}$)", 'viridis', vmin=mu_min, vmax=mu_max)
    _plot_field(fig1, axes1[2, 1], pos, mu_2, r"Eff. Viscosity ($\mu_{eff}$)", 'viridis', vmin=mu_min, vmax=mu_max)
    _plot_field(fig1, axes1[2, 2], pos, mu_3, r"Eff. Viscosity ($\mu_{eff}$)", 'viridis', vmin=mu_min, vmax=mu_max)

    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- FIGURE 2: Tier 3 Biochemical Focus ---
    # Extract Biochemistry channels based on _CHANNEL dictionary indices
    rp_3 = pred_t3_np[:, 4]
    fibrin_3 = pred_t3_np[:, 11]
    mat_s_3 = pred_t3_np[:, 13]

    fig2, axes2 = plt.subplots(1, 3, figsize=(20, 6))
    fig2.suptitle("Tier 3 Focus: Biochemical & Thrombosis Markers", fontsize=18, y=1.05)

    _plot_field(fig2, axes2[0], pos, rp_3, "Resting Platelets", 'Blues')
    _plot_field(fig2, axes2[1], pos, fibrin_3, "Fibrin Concentration (Clotting)", 'Reds')
    _plot_field(fig2, axes2[2], pos, mat_s_3, "Surface Platelet Deposit ($Mat_s$)", 'Oranges')

    fig2.tight_layout()

    # Show both figures simultaneously
    plt.show()


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    run_tier_comparison()