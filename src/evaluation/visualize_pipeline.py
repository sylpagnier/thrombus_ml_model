import contextlib
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import argparse
from matplotlib.widgets import Slider, Button
from src.utils.paths import get_project_root, resolve_checkpoint
from src.data_gen import MeshToGraphComplete, MeshToGraphPhase3, VesselGeneratorPhase3
from src.architecture.ginodeq import GINO_DEQ
from src.architecture.gnode_biochem import GNODE_Phase3, _SPECIES_LOG1P_MAX, _SPECIES_LOG1P_MIN
from src.architecture.lora_injection import inject_lora_to_spectral_linears
from src.config import PhysicsConfig, BiochemConfig, STATE_CHANNEL_MU_EFF_ND
from src.utils.nondim import to_t_nd

# Standard channel indices across all models for kinematics
_CHANNEL = dict(u=0, v=1, p=2, mu_eff=STATE_CHANNEL_MU_EFF_ND)
_KIN_CKPT_CANDIDATES = ("kinematics_best.pth", "kinematics_ckpt_latest.pth", "kinematics_ckpt_100.pth")
_BIOCHEM_CKPT_CANDIDATES = (
    "biochem_teacher_best.pth",
    "biochem_best_bio.pth",
    "biochem_latest_checkpoint.pth",
)


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


def _infer_latent_dim_from_state_dict(state_dict) -> int | None:
    """Match ``train_biochem_corrector``: infer GNODE width from saved tensors."""
    w = state_dict.get("kin_encoder.0.weight")
    if w is not None and hasattr(w, "shape") and len(w.shape) == 2:
        return int(w.shape[0])
    w = state_dict.get("bio_encoder.linear.parametrizations.weight.original")
    if w is not None and hasattr(w, "shape") and len(w.shape) == 2:
        return int(w.shape[0])
    return None


@contextlib.contextmanager
def _viz_biochem_ode_speedups() -> Iterator[None]:
    """Plain ``odeint`` + coarser RK for visualization unless the user already set these env vars."""
    keys = ("BIOCHEM_ODEINT_USE_ADJOINT", "BIOCHEM_ADJOINT_RK4_SUBSTEPS")
    saved = {k: os.environ.get(k) for k in keys}
    try:
        if saved["BIOCHEM_ODEINT_USE_ADJOINT"] is None:
            os.environ["BIOCHEM_ODEINT_USE_ADJOINT"] = "0"
        if saved["BIOCHEM_ADJOINT_RK4_SUBSTEPS"] is None:
            os.environ["BIOCHEM_ADJOINT_RK4_SUBSTEPS"] = "8"
        yield
    finally:
        for k, old in saved.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


def _filter_compatible_state_dict(
    source_state_dict: Dict[str, torch.Tensor],
    target_state_dict: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """Keep only checkpoint tensors whose key exists and shape matches the live model."""
    compatible: Dict[str, torch.Tensor] = {}
    skipped: List[str] = []
    for key, value in source_state_dict.items():
        target_value = target_state_dict.get(key, None)
        if target_value is None:
            skipped.append(key)
            continue
        if tuple(value.shape) != tuple(target_value.shape):
            skipped.append(key)
            continue
        compatible[key] = value
    return compatible, skipped


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


def _load_torch_checkpoint(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _checkpoint_state_dict(raw: Any) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """Return (metadata, state_dict) from a flat or nested biochem checkpoint."""
    if isinstance(raw, dict) and "model_state_dict" in raw:
        meta = {k: v for k, v in raw.items() if k != "model_state_dict"}
        state = raw["model_state_dict"]
        if isinstance(state, dict):
            return meta, state
    if isinstance(raw, dict):
        return {}, raw
    raise TypeError(f"Unsupported checkpoint type: {type(raw)!r}")


def _resolve_biochem_checkpoint(explicit: Optional[str] = None) -> Path:
    """Prefer teacher-best weights; override via CLI or ``VIZ_BIOCHEM_CHECKPOINT``."""
    name = (explicit or os.environ.get("VIZ_BIOCHEM_CHECKPOINT") or "").strip()
    if name:
        path = Path(name)
        if path.is_file():
            return path.resolve()
        resolved = resolve_checkpoint("b", name)
        if resolved.exists():
            return resolved
        raise FileNotFoundError(f"Biochem checkpoint not found: {name}")
    for ckpt_name in _BIOCHEM_CKPT_CANDIDATES:
        candidate = resolve_checkpoint("b", ckpt_name)
        if candidate.exists():
            return candidate
    expected_dir = resolve_checkpoint("b", _BIOCHEM_CKPT_CANDIDATES[0]).parent
    raise FileNotFoundError(
        "No biochem checkpoint found for visualization. Tried: "
        + ", ".join(str(expected_dir / n) for n in _BIOCHEM_CKPT_CANDIDATES)
        + " (train with STOP_AFTER_TEACHER=1 to write biochem_teacher_best.pth)."
    )


def _load_single_graph(proc_dir, device, label):
    files = sorted(proc_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No graph files found in {proc_dir} for {label}")
    return torch.load(files[0], weights_only=False).to(device)


def _run_model_once(model, data):
    pred = model(data)
    return pred[0] if isinstance(pred, tuple) else pred


def _biochem_rheology_fields(
    model: GNODE_Phase3,
    data,
    pred_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """COMSOL-style gelation: Carreau ``μ_blood`` + triggers ``μ1(Mat)``, ``μ2(FI)`` (same as ``GNODE_Phase3``)."""
    device = pred_t.device
    dtype = pred_t.dtype
    u_nd = pred_t[:, 0:1]
    v_nd = pred_t[:, 1:2]
    mu_inf = model.phys_cfg.mu_inf
    mu_0 = model.phys_cfg.mu_0
    lam = model.phys_cfg.lam
    n_idx = model.phys_cfg.n
    u_ref = data.u_ref.view(-1, 1).to(device=device, dtype=dtype)
    d_bar = data.d_bar.view(-1, 1).to(device=device, dtype=dtype)

    du_dx_nd = torch.sparse.mm(data.G_x, u_nd)
    du_dy_nd = torch.sparse.mm(data.G_y, u_nd)
    dv_dx_nd = torch.sparse.mm(data.G_x, v_nd)
    dv_dy_nd = torch.sparse.mm(data.G_y, v_nd)

    scale_grad = u_ref / d_bar
    gamma_dot = torch.sqrt(
        2 * ((du_dx_nd * scale_grad) ** 2 + (dv_dy_nd * scale_grad) ** 2)
        + ((du_dy_nd * scale_grad) + (dv_dx_nd * scale_grad)) ** 2
        + 1e-8
    )

    mu_blood = mu_inf + (mu_0 - mu_inf) * torch.pow(1.0 + (lam * gamma_dot) ** 2, (n_idx - 1.0) / 2.0)

    sp_safe = torch.clamp(
        pred_t[:, 4:16],
        min=torch.tensor(_SPECIES_LOG1P_MIN, device=device, dtype=dtype),
        max=torch.tensor(_SPECIES_LOG1P_MAX, device=device, dtype=dtype),
    )
    species_si = model.species_log_nd_to_si(sp_safe)
    fi_si = species_si[:, 8:9]
    mat_si = species_si[:, 11:12]
    mu1 = model.mu1_sigmoid(mat_si)
    mu2 = model.mu2_sigmoid(fi_si)
    mu_blood_mu1 = mu_blood * mu1
    return (
        mu_blood.squeeze(-1),
        mu1.squeeze(-1),
        mu2.squeeze(-1),
        mu_blood_mu1.squeeze(-1),
    )


def _rheology_series_numpy(
    model: GNODE_Phase3,
    data,
    pred_series_np: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``pred_series_np`` shaped ``[T, N, C]`` → ``(mu_blood*mu1 [T,N], mu2 [T,N])`` in SI / dimensionless."""
    device = data.x.device
    dtype = torch.float32
    t_steps = int(pred_series_np.shape[0])
    n = int(pred_series_np.shape[1])
    out_m1 = np.zeros((t_steps, n), dtype=np.float64)
    out_m2 = np.zeros((t_steps, n), dtype=np.float64)
    with torch.no_grad():
        for ti in range(t_steps):
            pred_t = torch.from_numpy(pred_series_np[ti]).to(device=device, dtype=dtype)
            _, _, mu2, mb_m1 = _biochem_rheology_fields(model, data, pred_t)
            out_m1[ti] = mb_m1.detach().cpu().numpy()
            out_m2[ti] = mu2.detach().cpu().numpy()
    return out_m1, out_m2


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


def _show_biochem_temporal_slider(
    pos,
    pred_biochem_series_np,
    custom_times,
    model_biochem: GNODE_Phase3,
    data_biochem,
    on_refresh=None,
):
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

    mb_mu1_all, mu2_all = _rheology_series_numpy(model_biochem, data_biochem, pred_biochem_series_np)
    m1_vmin, m1_vmax = float(mb_mu1_all.min()), float(mb_mu1_all.max())
    if m1_vmax <= m1_vmin + 1e-18:
        m1_vmax = m1_vmin + 1e-12
    mu2_cap = float(model_biochem.mu_ratio_max)

    fig, axs = plt.subplots(3, 2, figsize=(14, 14))
    plt.subplots_adjust(bottom=0.10, hspace=0.22)
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

    sc5 = axs[2, 0].scatter(
        pos[:, 0], pos[:, 1], c=mb_mu1_all[0], cmap="magma", s=3, vmin=m1_vmin, vmax=m1_vmax
    )
    axs[2, 0].set_title(r"$\mu_{blood}\times\mu_1$(Mat) [Pa·s]")
    fig.colorbar(sc5, ax=axs[2, 0], label="Pa·s")

    sc6 = axs[2, 1].scatter(pos[:, 0], pos[:, 1], c=mu2_all[0], cmap="Reds", s=3, vmin=0.0, vmax=mu2_cap)
    axs[2, 1].set_title(r"$\mu_2$ trigger (FI) [−]")
    fig.colorbar(sc6, ax=axs[2, 1], label="−")

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
        sc5.set_array(mb_mu1_all[idx])
        sc6.set_array(mu2_all[idx])

        fig.suptitle(f"Biochem Temporal Inspector (t={custom_times[idx]:.1f}s)", fontsize=16, fontweight="bold")
        fig.canvas.draw_idle()

    time_slider.on_changed(update)
    plt.show()


def run_phase_comparison(regenerate=True, seed=42, biochem_checkpoint: Optional[str] = None):
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
        mg1.run(max_files=1)
        mg3 = MeshToGraphPhase3(raw_dir=raw_dir, label_dir=raw_dir, proc_dir=graph_biochem_dir)
        mg3.run(max_files=1)
    else:
        print("\n♻️ Reusing existing single-case synthetic data.")

    # Load the specific graphs
    try:
        data_kine_base = _load_single_graph(graph_kine_base_dir, device, "kinematics")
        data_biochem = _load_single_graph(graph_biochem_dir, device, "biochem")
        pos = data_biochem.x[:, :2].cpu().numpy()  # Position is the same across all
        n_nodes = int(data_biochem.x.shape[0])
        n_edges = int(data_biochem.edge_index.shape[1])
        print(f"   ↳ Graph: {n_nodes} nodes, {n_edges} edges", flush=True)
    except FileNotFoundError as exc:
        print(f"⚠️ Failed to generate or load graph files: {exc}")
        return False

    # ------------------------------------------------------------------
    # 4. Load Models
    # ------------------------------------------------------------------
    print("\n🧠 Loading trained models...")
    kin_ckpt = _resolve_kinematics_checkpoint()
    print(f"   ↳ Using kinematics checkpoint: {kin_ckpt.name}")
    biochem_ckpt = _resolve_biochem_checkpoint(biochem_checkpoint)
    biochem_meta, biochem_state = _checkpoint_state_dict(_load_torch_checkpoint(biochem_ckpt))

    # Kinematics setup (optional faster DEQ for visualization — default matches training)
    kin_max_iters = int(os.environ.get("VIZ_KIN_MAX_ITERS", "25"))
    kin_max_iters = max(5, min(80, kin_max_iters))
    phys_cfg_kine = PhysicsConfig(phase="kinematics")
    model_kine_base = GINO_DEQ(
        in_channels=15,
        out_channels=5,
        latent_dim=256,
        max_iters=kin_max_iters,
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
    latent_env = os.environ.get("BIOCHEM_LATENT_DIM", "").strip()
    if latent_env:
        latent_dim = max(8, int(latent_env))
    else:
        latent_dim = _infer_latent_dim_from_state_dict(biochem_state) or 256
    print(f"   ↳ Using biochem checkpoint: {biochem_ckpt.name}")
    if biochem_meta.get("checkpoint_role") == "teacher_best":
        t_ep = biochem_meta.get("best_epoch", -1)
        t_mae = biochem_meta.get("val_mu_log_mae")
        ep_s = f" ep={int(t_ep):02d}" if isinstance(t_ep, int) and t_ep >= 0 else ""
        mae_s = f" val_logMAE={float(t_mae):.4f}" if t_mae is not None else ""
        print(f"   ↳ teacher-best checkpoint{ep_s}{mae_s}")
    print(f"   ↳ bio_encoder prior dim: {bio_enc_prior}")
    print(f"   ↳ latent_dim: {latent_dim}")
    _viz_inner = os.environ.get("VIZ_BIOCHEM_MAX_INNER_ITERS", "").strip()
    biochem_inner_iters = int(_viz_inner) if _viz_inner else 10
    biochem_inner_iters = max(3, min(25, biochem_inner_iters))
    model_biochem = GNODE_Phase3(
        phys_cfg=phys_cfg_biochem,
        in_channels=12,
        spatial_channels=15,
        latent_dim=latent_dim,
        max_inner_iters=biochem_inner_iters,
        bio_encoder_prior_dim=bio_enc_prior,
        mu_ratio_max=bio_cfg.mu_ratio_max,
        mat_crit=bio_cfg.viscosity_mat_crit,
        fi_crit=bio_cfg.viscosity_fi_crit,
        temp_mat=bio_cfg.viscosity_gnode_temp_mat,
        temp_fi=bio_cfg.viscosity_gnode_temp_fi,
    ).to(device)
    _inject_biochem_kinematic_lora(model_biochem)
    compatible_bio, skipped_bio = _filter_compatible_state_dict(biochem_state, model_biochem.state_dict())
    model_biochem.load_state_dict(compatible_bio, strict=False)
    if skipped_bio:
        print(f"   ↳ Skipped {len(skipped_bio)} checkpoint key(s) (no target or shape mismatch).")
    model_biochem.eval()

    # ------------------------------------------------------------------
    # 5. Inference
    # ------------------------------------------------------------------
    print("\n🔮 Running inference across kinematics + biochem...", flush=True)

    def _cuda_sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    with torch.no_grad():
        t_k0 = time.perf_counter()
        print("   ↳ Kinematics (GINO-DEQ)…", flush=True)
        pred_kine_base = _run_model_once(model_kine_base, data_kine_base)
        _cuda_sync()
        print(f"   ↳ Kinematics done in {time.perf_counter() - t_k0:.1f}s", flush=True)

        # ``GNODE_Phase3`` expects non-dimensional times (same as training: ``to_t_nd(..., bio_cfg.t_final)``).
        dense_times_si_full = bio_cfg.resolve_biochem_times(data_biochem, device)
        if dense_times_si_full.numel() < 2:
            raise ValueError("Biochem timeline must contain at least two timestamps for rollout visualization.")
        t_ref = float(bio_cfg.t_final)
        t_final_si = float(dense_times_si_full[-1].item())
        n_full = int(dense_times_si_full.numel())
        # Each macro step runs a full DEQ-style kinematics solve + an ODE segment — interactive viz
        # must subsample the COMSOL-sized grid (often ~60+ steps) or a single run can take many minutes.
        n_cap_raw = (os.environ.get("VIZ_BIOCHEM_MACRO_STEPS") or "16").strip()
        if n_cap_raw == "0" or (n_cap_raw.lower() in ("full", "all")):
            n_macro_use = n_full
        else:
            try:
                n_cap = max(4, min(512, int(n_cap_raw)))
            except ValueError:
                n_cap = 16
            n_macro_use = min(n_full, n_cap)
        dense_times_si = torch.linspace(
            0.0, t_final_si, steps=n_macro_use, device=device, dtype=torch.float32
        )
        dense_times = to_t_nd(dense_times_si, t_ref)
        dt_si = float((dense_times_si[1] - dense_times_si[0]).item())
        if dt_si <= 0.0:
            raise ValueError(f"Invalid biochem timeline step dt_si={dt_si}. Expected strictly positive spacing.")
        dt_nd = float((dense_times[1] - dense_times[0]).item())
        if dt_nd <= 0.0:
            raise ValueError(f"Invalid ND timeline step dt_nd={dt_nd}. Expected strictly positive spacing.")

        num_extra = max(1, int((t_final_si * 0.5) / dt_si))
        extended_times = torch.cat([            dense_times,
            dense_times[-1] + dt_nd * torch.arange(1, num_extra + 1, device=device, dtype=dense_times.dtype),
        ])
        print(
            f"   ↳ Biochem rollout: {extended_times.numel()} macro knots "
            f"(using {n_macro_use}/{n_full} time samples; set VIZ_BIOCHEM_MACRO_STEPS=0 for full density)",
            flush=True,
        )

        t_b0 = time.perf_counter()
        with _viz_biochem_ode_speedups():
            pred_biochem_series_dense = model_biochem(data_biochem, extended_times)
        _cuda_sync()
        print(f"   ↳ Biochem trajectory done in {time.perf_counter() - t_b0:.1f}s", flush=True)

        # Extract the keyframes used by the temporal slider (labels stay in SI seconds for the UI).
        custom_times = [0.0, t_final_si * 0.33, t_final_si * 0.66, t_final_si, t_final_si * 1.5]
        custom_times_nd = [t / t_ref for t in custom_times]
        frame_indices = [
            torch.argmin(torch.abs(extended_times - t_nd)).item()
            for t_nd in custom_times_nd
        ]
        pred_biochem_series = pred_biochem_series_dense[frame_indices]

        # Extract the final *trained* time step for static Fig 1 comparison.
        idx_t_final = torch.argmin(torch.abs(extended_times - (t_final_si / t_ref))).item()
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
    # Do not share μ_eff color limits across columns: biochem adds large gelation spikes on the same
    # ND channel scale as kine, so a joint max (often set by biochem) washes out Carreau structure
    # in the kinematics panel and makes it look falsely Newtonian/uniform.
    mu_k_min, mu_k_max = float(mu_kine_base.min()), float(mu_kine_base.max())
    mu_b_min, mu_b_max = float(mu_biochem.min()), float(mu_biochem.max())
    _eps = 1e-9
    if mu_k_max <= mu_k_min + _eps:
        mu_k_max = mu_k_min + max(abs(mu_k_min), 1.0) * 1e-6
    if mu_b_max <= mu_b_min + _eps:
        mu_b_max = mu_b_min + max(abs(mu_b_min), 1.0) * 1e-6

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

    # Row 3: Viscosity (independent scales — see ``mu_*`` mins/maxes above)
    _plot_field(fig1, axes1[2, 0], pos, mu_kine_base, r"Eff. Viscosity ($\mu_{eff,nd}$) kine", 'viridis', vmin=mu_k_min, vmax=mu_k_max)
    _plot_field(fig1, axes1[2, 1], pos, mu_biochem, r"Eff. Viscosity ($\mu_{eff,nd}$) biochem", 'viridis', vmin=mu_b_min, vmax=mu_b_max)

    fig1.tight_layout(rect=(0, 0.03, 1, 0.95))

    # --- FIGURE 2: Biochem COMSOL-style viscosity triggers (final rollout time) ---
    with torch.no_grad():
        pred_bt = torch.from_numpy(pred_biochem_np).to(
            device=data_biochem.x.device, dtype=torch.float32
        )
        _, _, mu2_t, mb_m1 = _biochem_rheology_fields(model_biochem, data_biochem, pred_bt)
    mb_m1_np = mb_m1.detach().cpu().numpy()
    mu2_np = mu2_t.detach().cpu().numpy()
    m1s_vmin, m1s_vmax = float(mb_m1_np.min()), float(mb_m1_np.max())
    if m1s_vmax <= m1s_vmin + 1e-18:
        m1s_vmax = m1s_vmin + 1e-12
    fig2, axs2 = plt.subplots(1, 2, figsize=(14, 5.5))
    fig2.suptitle(
        "Biochem: COMSOL-style gelation (final rollout time; same μ1/μ2 as GNODE_Phase3)",
        fontsize=14,
        fontweight="bold",
    )
    _plot_field(
        fig2,
        axs2[0],
        pos,
        mb_m1_np,
        r"$\mu_{blood}\times\mu_1$(Mat) [Pa·s]",
        "magma",
        vmin=m1s_vmin,
        vmax=m1s_vmax,
    )
    _plot_field(
        fig2,
        axs2[1],
        pos,
        mu2_np,
        r"$\mu_2$ trigger (FI) [−]",
        "Reds",
        vmin=0.0,
        vmax=float(model_biochem.mu_ratio_max),
    )
    fig2.tight_layout(rect=(0, 0.03, 1, 0.92))

    ax_refresh_main = fig1.add_axes([0.83, 0.01, 0.15, 0.04])
    refresh_btn_main = Button(ax_refresh_main, "Refresh Geometry", color="lightgray", hovercolor="gainsboro")

    def _request_refresh():
        refresh_state["requested"] = True
        plt.close("all")

    refresh_btn_main.on_clicked(lambda _: _request_refresh())

    print("⏳ Opening interactive Biochem temporal slider...")
    _show_biochem_temporal_slider(
        pos,
        pred_biochem_series_np,
        custom_times,
        model_biochem,
        data_biochem,
        on_refresh=_request_refresh,
    )
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
    parser.add_argument(
        "--biochem-checkpoint",
        type=str,
        default=None,
        help=(
            "Biochem weights file (default: biochem_teacher_best.pth, else biochem_best_bio.pth). "
            "Override with path or filename under outputs/biochem."
        ),
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
        refresh_requested = run_phase_comparison(
            regenerate=regenerate,
            seed=seed,
            biochem_checkpoint=args.biochem_checkpoint,
        )
        if not refresh_requested:
            break
        regenerate = True
        seed = int(np.random.default_rng().integers(0, 2**31 - 1))
        print(f"🔁 Refresh requested. Regenerating with new seed: {seed}")