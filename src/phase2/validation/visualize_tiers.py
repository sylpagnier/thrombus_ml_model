import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import argparse
from matplotlib.widgets import Slider, Button
from src.utils.paths import get_project_root
from src.phase1.data_gen.mesh_to_graph import MeshToGraphComplete
from src.phase2.vessel_generator_tier3 import VesselGeneratorTier3
from src.phase2.mesh_to_graph_tier3 import MeshToGraphTier3
from src.phase1.physics.ginodeq import GINO_DEQ
from src.phase2.gnode_tier3 import GNODE_Tier3
from src.config import PhysicsConfig, BiochemConfig, STATE_CHANNEL_MU_EFF_ND

# Standard channel indices across all models for kinematics
_CHANNEL = dict(u=0, v=1, p=2, mu_eff=STATE_CHANNEL_MU_EFF_ND)


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


def _show_tier3_temporal_slider(pos, pred_t3_series_np, custom_times, on_refresh=None):
    u_all = pred_t3_series_np[:, :, _CHANNEL["u"]]
    v_all = pred_t3_series_np[:, :, _CHANNEL["v"]]
    vel_all = np.sqrt(u_all ** 2 + v_all ** 2)
    p_all = pred_t3_series_np[:, :, _CHANNEL["p"]]
    # Tier-3 species channels are stored as log1p(species_nd); convert FI and Mat_s to SI for plotting.
    bio_cfg = BiochemConfig(tier="tier3")
    scales = bio_cfg.get_species_scales(device="cpu").cpu().numpy()
    fib_all = np.expm1(np.clip(pred_t3_series_np[:, :, 12], a_min=0.0, a_max=None)) * scales[8]
    mat_all = np.expm1(np.clip(pred_t3_series_np[:, :, 15], a_min=0.0, a_max=None)) * scales[11]

    # Keep color scales fixed across time to avoid frame-wise contrast artifacts.
    vel_vmin, vel_vmax = float(vel_all.min()), float(vel_all.max())
    p_vmin, p_vmax = float(p_all.min()), float(p_all.max())
    fib_vmin, fib_vmax = float(fib_all.min()), float(fib_all.max())
    mat_vmin, mat_vmax = float(mat_all.min()), float(mat_all.max())

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    plt.subplots_adjust(bottom=0.10, hspace=0.2)
    fig.suptitle(f"Tier 3 Temporal Inspector (t={custom_times[0]:.1f}s)", fontsize=16, fontweight="bold")

    sc1 = axs[0, 0].scatter(pos[:, 0], pos[:, 1], c=vel_all[0], cmap="jet", s=3, vmin=vel_vmin, vmax=vel_vmax)
    axs[0, 0].set_title("Velocity Magnitude (ND)")
    fig.colorbar(sc1, ax=axs[0, 0], label="ND")

    sc2 = axs[0, 1].scatter(pos[:, 0], pos[:, 1], c=p_all[0], cmap="coolwarm", s=3, vmin=p_vmin, vmax=p_vmax)
    axs[0, 1].set_title("Pressure (ND)")
    fig.colorbar(sc2, ax=axs[0, 1], label="ND")

    sc3 = axs[1, 0].scatter(pos[:, 0], pos[:, 1], c=fib_all[0], cmap="Reds", s=3, vmin=fib_vmin, vmax=fib_vmax)
    axs[1, 0].set_title("Fibrin (SI)")
    fig.colorbar(sc3, ax=axs[1, 0], label="mol/m^3")

    sc4 = axs[1, 1].scatter(pos[:, 0], pos[:, 1], c=mat_all[0], cmap="Oranges", s=3, vmin=mat_vmin, vmax=mat_vmax)
    axs[1, 1].set_title("Surface Platelets (Mat_s, SI)")
    fig.colorbar(sc4, ax=axs[1, 1], label="plt/m^2")

    for ax in axs.flat:
        ax.set_aspect("equal")
        ax.axis("off")

    ax_slider = plt.axes([0.2, 0.02, 0.5, 0.03])
    time_slider = Slider(
        ax=ax_slider,
        label="Time Step Index",
        valmin=0,
        valmax=len(custom_times) - 1,
        valinit=0,
        valstep=1,
        color="teal",
    )
    if on_refresh is not None:
        ax_refresh = plt.axes([0.74, 0.015, 0.2, 0.05])
        refresh_button = Button(ax_refresh, "Refresh Geometry", color="lightgray", hovercolor="gainsboro")

        def _on_refresh_click(_):
            on_refresh()

        refresh_button.on_clicked(_on_refresh_click)

    def update(_):
        idx = int(time_slider.val)
        sc1.set_array(vel_all[idx])
        sc2.set_array(p_all[idx])
        sc3.set_array(fib_all[idx])
        sc4.set_array(mat_all[idx])

        fig.suptitle(f"Tier 3 Temporal Inspector (t={custom_times[idx]:.1f}s)", fontsize=16, fontweight="bold")
        fig.canvas.draw_idle()

    time_slider.on_changed(update)
    plt.show()


def run_tier_comparison(regenerate=True, seed=42):
    root = get_project_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    print(f"🎲 Geometry seed: {seed}")
    refresh_state = {"requested": False}

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

    need_regen = regenerate
    if not regenerate:
        has_ready_data = (
            any(raw_dir.glob("*.msh")) and
            any(graph_t1_dir.glob("*.pt")) and
            any(graph_t2_dir.glob("*.pt")) and
            any(graph_t3_dir.glob("*.pt"))
        )
        if not has_ready_data:
            print("⚠️ No existing cached synthetic data found. Regenerating now...")
            need_regen = True

    if need_regen:
        for d in [raw_dir, graph_t1_dir, graph_t2_dir, graph_t3_dir]:
            for f in d.glob("*"):
                if f.is_file():
                    f.unlink()

        print("\n📐 Generating 1 complex synthetic vessel for the comparison...")
        vg = VesselGeneratorTier3(output_dir=raw_dir)
        vg.run_pipeline(n=1, level=1, num_workers=1, seed=seed)

        print("\n🕸️ Converting mesh to graphs for each tier's specific channel requirements...")
        mg1 = MeshToGraphComplete(tier="tier1", raw_dir=raw_dir, label_dir=raw_dir, proc_dir=graph_t1_dir)
        mg1.run()
        mg2 = MeshToGraphComplete(tier="tier2", raw_dir=raw_dir, label_dir=raw_dir, proc_dir=graph_t2_dir)
        mg2.run()
        mg3 = MeshToGraphTier3(raw_dir=raw_dir, label_dir=raw_dir, proc_dir=graph_t3_dir)
        mg3.run()
    else:
        print("\n♻️ Reusing existing single-case synthetic data.")

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
        mu_inf_nd=(phys_cfg_t2.mu_inf / phys_cfg_t2.mu_viscosity_nd_scale),
        mu_0_nd=(phys_cfg_t2.mu_0 / phys_cfg_t2.mu_viscosity_nd_scale)
    ).to(device)
    model_t2.load_state_dict(torch.load(model_dir / "tier2_best_physics.pth", map_location=device, weights_only=True))
    model_t2.eval()

    # Tier 3 Setup (same PhysicsConfig defaults as train_tier3.py / config.py)
    phys_cfg_t3 = PhysicsConfig(tier="tier3")
    bio_cfg = BiochemConfig(tier="tier3")
    model_t3 = GNODE_Tier3(
        phys_cfg=phys_cfg_t3,
        in_channels=12,
        spatial_channels=15,
        latent_dim=64,
        max_inner_iters=10,
        mu_ratio_max=bio_cfg.mu_ratio_max,
        mat_crit=bio_cfg.viscosity_mat_crit,
        fi_crit=bio_cfg.viscosity_fi_crit,
        temp_mat=bio_cfg.viscosity_gnode_temp_mat,
        temp_fi=bio_cfg.viscosity_gnode_temp_fi,
    ).to(device)
    model_t3.load_state_dict(torch.load(model_dir / "tier3_best_bio.pth", map_location=device, weights_only=True))
    model_t3.eval()

    # ------------------------------------------------------------------
    # 5. Inference
    # ------------------------------------------------------------------
    print("\n🔮 Running Inference across all Tiers...")
    with torch.no_grad():
        pred_t1 = model_t1(data_t1) if isinstance(model_t1(data_t1), tuple) else model_t1(data_t1)
        pred_t2 = model_t2(data_t2) if isinstance(model_t2(data_t2), tuple) else model_t2(data_t2)

        # Setup evaluation times for Tier 3 Neural ODE.
        dense_times = bio_cfg.resolve_tier3_times(data_t3, device)
        t_final = float(dense_times[-1].item())
        dt = float((dense_times[1] - dense_times[0]).item())

        # Extrapolate to 1.5x t_final by extending the dense timeline.
        num_extra = int((t_final * 0.5) / dt)
        extended_times = torch.cat([
            dense_times,
            dense_times[-1] + dt * torch.arange(1, num_extra + 1, device=device),
        ])

        # Forward pass over the dense timeline (safer ODE stepping).
        pred_t3_series_dense = model_t3(data_t3, extended_times)

        # Extract the keyframes used by the temporal slider.
        custom_times = [0.0, t_final * 0.33, t_final * 0.66, t_final, t_final * 1.5]
        frame_indices = [
            torch.argmin(torch.abs(extended_times - t)).item()
            for t in custom_times
        ]
        pred_t3_series = pred_t3_series_dense[frame_indices]

        # Extract the final *trained* time step for static Fig 1 comparison.
        idx_t_final = torch.argmin(torch.abs(extended_times - t_final)).item()
        pred_t3 = pred_t3_series_dense[idx_t_final]

    pred_t1_np = pred_t1.cpu().numpy()
    pred_t2_np = pred_t2.cpu().numpy()
    pred_t3_np = pred_t3.cpu().numpy()
    pred_t3_series_np = pred_t3_series.cpu().numpy()

    # ------------------------------------------------------------------
    # 5.5 Extract Fields & Calculate Bounds
    # ------------------------------------------------------------------
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
    vel_min = min(vel_1.min(), vel_2.min(), vel_3.min())
    vel_max = max(vel_1.max(), vel_2.max(), vel_3.max())
    p_min = min(p_1.min(), p_2.min(), p_3.min())
    p_max = max(p_1.max(), p_2.max(), p_3.max())
    mu_min = min(mu_1.min(), mu_2.min(), mu_3.min())
    mu_max = max(mu_1.max(), mu_2.max(), mu_3.max())

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

    # Row 1: Velocity (non-dimensional)
    _plot_field(fig1, axes1[0, 0], pos, vel_1, "Velocity Mag (ND)", 'jet', vmin=vel_min, vmax=vel_max)
    _plot_field(fig1, axes1[0, 1], pos, vel_2, "Velocity Mag (ND)", 'jet', vmin=vel_min, vmax=vel_max)
    _plot_field(fig1, axes1[0, 2], pos, vel_3, "Velocity Mag (ND)", 'jet', vmin=vel_min, vmax=vel_max)

    # Row 2: Pressure (non-dimensional)
    _plot_field(fig1, axes1[1, 0], pos, p_1, "Pressure (ND)", 'coolwarm', vmin=p_min, vmax=p_max)
    _plot_field(fig1, axes1[1, 1], pos, p_2, "Pressure (ND)", 'coolwarm', vmin=p_min, vmax=p_max)
    _plot_field(fig1, axes1[1, 2], pos, p_3, "Pressure (ND)", 'coolwarm', vmin=p_min, vmax=p_max)

    # Row 3: Viscosity
    _plot_field(fig1, axes1[2, 0], pos, mu_1, r"Eff. Viscosity ($\mu_{eff}$)", 'viridis', vmin=mu_min, vmax=mu_max)
    _plot_field(fig1, axes1[2, 1], pos, mu_2, r"Eff. Viscosity ($\mu_{eff}$)", 'viridis', vmin=mu_min, vmax=mu_max)
    _plot_field(fig1, axes1[2, 2], pos, mu_3, r"Eff. Viscosity ($\mu_{eff}$)", 'viridis', vmin=mu_min, vmax=mu_max)

    fig1.tight_layout(rect=(0, 0.03, 1, 0.95))

    ax_refresh_main = fig1.add_axes([0.83, 0.01, 0.15, 0.04])
    refresh_btn_main = Button(ax_refresh_main, "Refresh Geometry", color="lightgray", hovercolor="gainsboro")

    def _request_refresh():
        refresh_state["requested"] = True
        plt.close("all")

    refresh_btn_main.on_clicked(lambda _: _request_refresh())

    print("⏳ Opening interactive Tier 3 temporal slider...")
    _show_tier3_temporal_slider(pos, pred_t3_series_np, custom_times, on_refresh=_request_refresh)
    return refresh_state["requested"]


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    parser = argparse.ArgumentParser(description="Visualize static + temporal behavior across tiers")
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate temporary single-case synthetic data before plotting",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse previously generated temporary data (if available)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the single synthetic case when regeneration is enabled",
    )
    args = parser.parse_args()

    if args.regenerate and args.reuse:
        raise ValueError("Use only one of --regenerate or --reuse")

    if args.reuse:
        regenerate = False
    else:
        regenerate = True

    seed = args.seed
    while True:
        refresh_requested = run_tier_comparison(regenerate=regenerate, seed=seed)
        if not refresh_requested:
            break
        regenerate = True
        seed = int(np.random.default_rng().integers(0, 2**31 - 1))
        print(f"🔁 Refresh requested. Regenerating with new seed: {seed}")