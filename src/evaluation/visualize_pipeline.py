import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import argparse
from matplotlib.widgets import Slider, Button
from src.utils.paths import get_project_root, resolve_checkpoint
from src.data_gen import MeshToGraphComplete, MeshToGraphPhase3, VesselGeneratorPhase3
from src.architecture.ginodeq import GINO_DEQ
from src.architecture.gnode_biochem import GNODE_Phase3
from src.architecture.lora_injection import inject_lora_to_spectral_linears
from src.config import PhysicsConfig, BiochemConfig, STATE_CHANNEL_MU_EFF_ND

# Standard channel indices across all models for kinematics
_CHANNEL = dict(u=0, v=1, p=2, mu_eff=STATE_CHANNEL_MU_EFF_ND)
_KIN_CKPT_CANDIDATES = ("kinematics_best.pth", "kinematics_ckpt_latest.pth", "kinematics_ckpt_100.pth")


def _infer_bio_encoder_prior_dim_from_state_dict(state_dict):
    """Infer extra bio-encoder prior channels from checkpoint tensor shape."""
    key = "bio_encoder.linear.parametrizations.weight.original"
    weight = state_dict.get(key)
    if weight is None or not hasattr(weight, "shape") or len(weight.shape) != 2:
        return None
    # GNODE bio_encoder input = 12 species + 3 kinematics + 15 spatial + prior_dim.
    base_in_features = 30
    inferred = int(weight.shape[1]) - base_in_features
    if inferred < 0:
        return None
    return inferred


def _inject_biochem_kinematic_lora(model, rank=4, alpha=1.0):
    """Match Biochem training: LoRA on kinematic SpectralLinear layers."""
    n_enc = inject_lora_to_spectral_linears(model.kin_encoder, rank=rank, alpha=alpha)
    n_proc = inject_lora_to_spectral_linears(model.kin_processor, rank=rank, alpha=alpha)
    n_dec = inject_lora_to_spectral_linears(model.kinematics_decoder, rank=rank, alpha=alpha)
    print(
        f"   ↳ LoRA injected: kin_encoder={n_enc}, kin_processor={n_proc}, "
        f"kinematics_decoder={n_dec} (rank={rank}, alpha={alpha})"
    )


def _resolve_kinematics_checkpoint():
    for ckpt_name in _KIN_CKPT_CANDIDATES:
        candidate = resolve_checkpoint("a", ckpt_name)
        if candidate.exists():
            return candidate
    expected_dir = resolve_checkpoint("a", _KIN_CKPT_CANDIDATES[0]).parent
    raise FileNotFoundError(
        "No kinematics checkpoint found for visualization. Tried: "
        + ", ".join(str(expected_dir / name) for name in _KIN_CKPT_CANDIDATES)
    )


def _load_single_graph(proc_dir, device, label):
    files = sorted(proc_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No graph files found in {proc_dir} for {label}")
    return torch.load(files[0], weights_only=False).to(device)


def _run_model_once(model, data):
    pred = model(data)
    return pred[0] if isinstance(pred, tuple) else pred


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


def _show_biochem_temporal_slider(pos, pred_biochem_series_np, custom_times, on_refresh=None):
    u_all = pred_biochem_series_np[:, :, _CHANNEL["u"]]
    v_all = pred_biochem_series_np[:, :, _CHANNEL["v"]]
    vel_all = np.sqrt(u_all ** 2 + v_all ** 2)
    p_all = pred_biochem_series_np[:, :, _CHANNEL["p"]]
    # Biochem species channels are stored as log1p(species_nd); convert FI and Mat_s to SI for plotting.
    bio_cfg = BiochemConfig(phase="biochem")
    scales = bio_cfg.get_species_scales(device="cpu").cpu().numpy()
    fib_all = np.expm1(np.clip(pred_biochem_series_np[:, :, 12], a_min=0.0, a_max=None)) * scales[8]
    mat_all = np.expm1(np.clip(pred_biochem_series_np[:, :, 15], a_min=0.0, a_max=None)) * scales[11]

    # Keep color scales fixed across time to avoid frame-wise contrast artifacts.
    vel_vmin, vel_vmax = float(vel_all.min()), float(vel_all.max())
    p_vmin, p_vmax = float(p_all.min()), float(p_all.max())
    fib_vmin, fib_vmax = float(fib_all.min()), float(fib_all.max())
    mat_vmin, mat_vmax = float(mat_all.min()), float(mat_all.max())

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    plt.subplots_adjust(bottom=0.10, hspace=0.2)
    fig.suptitle(f"Biochem Temporal Inspector (t={custom_times[0]:.1f}s)", fontsize=16, fontweight="bold")

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

        fig.suptitle(f"Biochem Temporal Inspector (t={custom_times[idx]:.1f}s)", fontsize=16, fontweight="bold")
        fig.canvas.draw_idle()

    time_slider.on_changed(update)
    plt.show()


def run_phase_comparison(regenerate=True, seed=42):
    root = get_project_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    print(f"🎲 Geometry seed: {seed}")
    refresh_state = {"requested": False}

    # ------------------------------------------------------------------
    # 1. Setup Directories
    # ------------------------------------------------------------------
    test_dir = root / "data" / "phase_comparison_test"
    raw_dir = test_dir / "raw_meshes"
    graph_kine_base_dir = test_dir / "graphs_kine_base"
    graph_biochem_dir = test_dir / "graphs_biochem"

    for d in [raw_dir, graph_kine_base_dir, graph_biochem_dir]:
        d.mkdir(parents=True, exist_ok=True)

    need_regen = regenerate
    if not regenerate:
        has_ready_data = (
            any(raw_dir.glob("*.msh")) and
            any(graph_kine_base_dir.glob("*.pt")) and
            any(graph_biochem_dir.glob("*.pt"))
        )
        if not has_ready_data:
            print("⚠️ No existing cached synthetic data found. Regenerating now...")
            need_regen = True

    if need_regen:
        for d in [raw_dir, graph_kine_base_dir, graph_biochem_dir]:
            for f in d.glob("*"):
                if f.is_file():
                    f.unlink()

        print("\n📐 Generating 1 complex synthetic vessel for the comparison...")
        vg = VesselGeneratorPhase3(output_dir=raw_dir)
        vg.run_pipeline(n=1, level=1, num_workers=1, seed=seed)

        print("\n🕸️ Converting mesh to graphs for each phase's specific channel requirements...")
        mg1 = MeshToGraphComplete(phase="kinematics", raw_dir=raw_dir, label_dir=raw_dir, proc_dir=graph_kine_base_dir)
        mg1.run()
        mg3 = MeshToGraphPhase3(raw_dir=raw_dir, label_dir=raw_dir, proc_dir=graph_biochem_dir)
        mg3.run()
    else:
        print("\n♻️ Reusing existing single-case synthetic data.")

    # Load the specific graphs
    try:
        data_kine_base = _load_single_graph(graph_kine_base_dir, device, "kinematics")
        data_biochem = _load_single_graph(graph_biochem_dir, device, "biochem")
        pos = data_biochem.x[:, :2].cpu().numpy()  # Position is the same across all
    except FileNotFoundError as exc:
        print(f"⚠️ Failed to generate or load graph files: {exc}")
        return False

    # ------------------------------------------------------------------
    # 4. Load Models
    # ------------------------------------------------------------------
    print("\n🧠 Loading trained models...")
    kin_ckpt = _resolve_kinematics_checkpoint()
    print(f"   ↳ Using kinematics checkpoint: {kin_ckpt.name}")
    biochem_ckpt = resolve_checkpoint("b", "biochem_best_bio.pth")
    if not biochem_ckpt.exists():
        raise FileNotFoundError(f"Biochem checkpoint not found: {biochem_ckpt}")
    biochem_state = torch.load(biochem_ckpt, map_location=device, weights_only=True)

    # Kinematics setup
    phys_cfg_kine = PhysicsConfig(phase="kinematics")
    model_kine_base = GINO_DEQ(
        in_channels=15,
        out_channels=5,
        latent_dim=256,
        max_iters=25,
        num_fourier_freqs=16,
        phys_cfg=phys_cfg_kine,
        activation_fn="silu",
        use_hard_bcs=True,
        use_siren_decoder=True,
        use_width_priors=True,
    ).to(device)
    model_kine_base.load_state_dict(torch.load(kin_ckpt, map_location=device, weights_only=True))
    model_kine_base.eval()

    # Biochem Setup (same PhysicsConfig defaults as train_biochem_corrector.py / config.py)
    phys_cfg_biochem = PhysicsConfig(phase="biochem")
    bio_cfg = BiochemConfig(phase="biochem")
    env_prior = int(os.environ.get("BIOCHEM_BIO_ENCODER_PRIOR_DIM", "2"))
    inferred_prior = _infer_bio_encoder_prior_dim_from_state_dict(biochem_state)
    bio_enc_prior = inferred_prior if inferred_prior is not None else env_prior
    print(f"   ↳ Using biochem checkpoint: {biochem_ckpt.name}")
    print(f"   ↳ bio_encoder prior dim: {bio_enc_prior}")
    model_biochem = GNODE_Phase3(
        phys_cfg=phys_cfg_biochem,
        in_channels=12,
        spatial_channels=15,
        latent_dim=64,
        max_inner_iters=10,
        bio_encoder_prior_dim=bio_enc_prior,
        mu_ratio_max=bio_cfg.mu_ratio_max,
        mat_crit=bio_cfg.viscosity_mat_crit,
        fi_crit=bio_cfg.viscosity_fi_crit,
        temp_mat=bio_cfg.viscosity_gnode_temp_mat,
        temp_fi=bio_cfg.viscosity_gnode_temp_fi,
    ).to(device)
    _inject_biochem_kinematic_lora(model_biochem)
    model_biochem.load_state_dict(biochem_state)
    model_biochem.eval()

    # ------------------------------------------------------------------
    # 5. Inference
    # ------------------------------------------------------------------
    print("\n🔮 Running inference across kinematics + biochem...")
    with torch.no_grad():
        pred_kine_base = _run_model_once(model_kine_base, data_kine_base)
        # Setup evaluation times for Biochem Neural ODE.
        dense_times = bio_cfg.resolve_biochem_times(data_biochem, device)
        if dense_times.numel() < 2:
            raise ValueError("Biochem timeline must contain at least two timestamps for rollout visualization.")
        t_final = float(dense_times[-1].item())
        dt = float((dense_times[1] - dense_times[0]).item())
        if dt <= 0.0:
            raise ValueError(f"Invalid biochem timeline step dt={dt}. Expected strictly positive spacing.")

        # Extrapolate to 1.5x t_final by extending the dense timeline.
        num_extra = max(1, int((t_final * 0.5) / dt))
        extended_times = torch.cat([
            dense_times,
            dense_times[-1] + dt * torch.arange(1, num_extra + 1, device=device),
        ])

        # Forward pass over the dense timeline (safer ODE stepping).
        pred_biochem_series_dense = model_biochem(data_biochem, extended_times)

        # Extract the keyframes used by the temporal slider.
        custom_times = [0.0, t_final * 0.33, t_final * 0.66, t_final, t_final * 1.5]
        frame_indices = [
            torch.argmin(torch.abs(extended_times - t)).item()
            for t in custom_times
        ]
        pred_biochem_series = pred_biochem_series_dense[frame_indices]

        # Extract the final *trained* time step for static Fig 1 comparison.
        idx_t_final = torch.argmin(torch.abs(extended_times - t_final)).item()
        pred_biochem = pred_biochem_series_dense[idx_t_final]

    pred_kine_base_np = pred_kine_base.detach().cpu().numpy()
    pred_biochem_np = pred_biochem.detach().cpu().numpy()
    pred_biochem_series_np = pred_biochem_series.detach().cpu().numpy()

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

    vel_kine_base, p_kine_base, mu_kine_base = get_kinematics(pred_kine_base_np)
    vel_biochem, p_biochem, mu_biochem = get_kinematics(pred_biochem_np)

    # Determine global min/max for fair colorbar comparisons across columns
    vel_min = min(vel_kine_base.min(), vel_biochem.min())
    vel_max = max(vel_kine_base.max(), vel_biochem.max())
    p_min = min(p_kine_base.min(), p_biochem.min())
    p_max = max(p_kine_base.max(), p_biochem.max())
    mu_min = min(mu_kine_base.min(), mu_biochem.min())
    mu_max = max(mu_kine_base.max(), mu_biochem.max())

    # ------------------------------------------------------------------
    # 6. Plotting
    # ------------------------------------------------------------------
    print("🎨 Generating Comparison Plots...")

    # --- FIGURE 1: Kinematic Comparison ---
    fig1, axes1 = plt.subplots(3, 2, figsize=(15, 14))
    fig1.suptitle("Kinematic Evolution: Kine vs Biochem (Coupled)",
                  fontsize=18, y=0.98)

    columns = ["Kine", "Biochem (Coupled)"]
    for ax, col in zip(axes1[0], columns):
        ax.set_title(col, fontsize=14, fontweight='bold', pad=20)

    # Row 1: Velocity (non-dimensional)
    _plot_field(fig1, axes1[0, 0], pos, vel_kine_base, "Velocity Mag (ND)", 'jet', vmin=vel_min, vmax=vel_max)
    _plot_field(fig1, axes1[0, 1], pos, vel_biochem, "Velocity Mag (ND)", 'jet', vmin=vel_min, vmax=vel_max)

    # Row 2: Pressure (non-dimensional)
    _plot_field(fig1, axes1[1, 0], pos, p_kine_base, "Pressure (ND)", 'coolwarm', vmin=p_min, vmax=p_max)
    _plot_field(fig1, axes1[1, 1], pos, p_biochem, "Pressure (ND)", 'coolwarm', vmin=p_min, vmax=p_max)

    # Row 3: Viscosity
    _plot_field(fig1, axes1[2, 0], pos, mu_kine_base, r"Eff. Viscosity ($\mu_{eff}$)", 'viridis', vmin=mu_min, vmax=mu_max)
    _plot_field(fig1, axes1[2, 1], pos, mu_biochem, r"Eff. Viscosity ($\mu_{eff}$)", 'viridis', vmin=mu_min, vmax=mu_max)

    fig1.tight_layout(rect=(0, 0.03, 1, 0.95))

    ax_refresh_main = fig1.add_axes([0.83, 0.01, 0.15, 0.04])
    refresh_btn_main = Button(ax_refresh_main, "Refresh Geometry", color="lightgray", hovercolor="gainsboro")

    def _request_refresh():
        refresh_state["requested"] = True
        plt.close("all")

    refresh_btn_main.on_clicked(lambda _: _request_refresh())

    print("⏳ Opening interactive Biochem temporal slider...")
    _show_biochem_temporal_slider(pos, pred_biochem_series_np, custom_times, on_refresh=_request_refresh)
    return refresh_state["requested"]


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    parser = argparse.ArgumentParser(description="Visualize static + temporal behavior across phases")
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
        refresh_requested = run_phase_comparison(regenerate=regenerate, seed=seed)
        if not refresh_requested:
            break
        regenerate = True
        seed = int(np.random.default_rng().integers(0, 2**31 - 1))
        print(f"🔁 Refresh requested. Regenerating with new seed: {seed}")