import contextlib
import os
import time
from dataclasses import dataclass
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
from src.architecture.gnode_biochem import (
    apply_biochem_forward_policy_from_checkpoint_meta,
    biochem_forward_policy_from_checkpoint_meta,
    format_biochem_forward_policy_summary,
)
from src.architecture.kinematics_model_config import (
    build_gino_deq_from_ctor,
    kinematics_checkpoint_tensors,
    resolve_gino_deq_ctor_kwargs,
)
from src.architecture.gnode_biochem import (
    GNODE_Phase3,
    _SPECIES_LOG1P_MAX,
    _SPECIES_LOG1P_MIN,
    _biochem_mu_disable_explicit_gelation,
    _biochem_mu_simple_log_residual_enabled,
    biochem_explicit_gelation_terms,
    resolve_gnode_phase3_ctor_kwargs,
)
from src.architecture.lora_injection import inject_lora_to_spectral_linears
from src.config import PhysicsConfig, BiochemConfig, STATE_CHANNEL_MU_EFF_ND, VesselConfig
from src.utils.nondim import to_t_nd
from src.utils.channel_schema import infer_missing_schema

# Standard channel indices across all models for kinematics
_CHANNEL = dict(u=0, v=1, p=2, mu_eff=STATE_CHANNEL_MU_EFF_ND)
_KIN_CKPT_CANDIDATES = ("kinematics_best.pth", "kinematics_ckpt_latest.pth", "kinematics_ckpt_100.pth")
_BIOCHEM_CKPT_CANDIDATES = (
    "biochem_teacher_best_high_mu.pth",
    "biochem_teacher_last.pth",
    "biochem_teacher_best.pth",
    "biochem_best_high_mu.pth",
    "biochem_latest_checkpoint.pth",
)
_BIOCHEM_TEACHER_CKPT_CANDIDATES = (
    "biochem_teacher_best_high_mu.pth",
    "biochem_teacher_last.pth",
)
# Names written by ``train_biochem_corrector.py``.
_BIOCHEM_CKPT_ROLE_BY_NAME: Dict[str, str] = {
    "biochem_teacher_best_high_mu.pth": "teacher_best_high_mu",
    "biochem_teacher_last.pth": "teacher_last",
    "biochem_teacher_best.pth": "teacher_best_all_legacy",
    "biochem_best_high_mu.pth": "corrector_best_high_mu",
    "biochem_best_bio.pth": "corrector_best_bio_legacy",
    "biochem_latest_checkpoint.pth": "corrector_latest",
}
_BIOCHEM_TEACHER_CKPT_ROLES = frozenset(
    {"teacher_best_high_mu", "teacher_last", "teacher_best", "teacher_best_all_legacy"}
)
_BIOCHEM_CKPT_INVENTORY_EXTRA = ("biochem_best_bio.pth",)
_DEFAULT_VAL_ANCHOR_STEM = "patient007"
# Legacy COMSOL display (species sigmoids); biochem rollout uses stored ``mu_eff`` when ablated.
_MU_DYNAMIC_SI_LABEL = r"$\mu_b \times (\mu_1(\mathrm{Mat}) + \mu_2(\mathrm{FI}))$ [Pa·s]"
_MU1_PRODUCT_SI_LABEL = r"$\mu_{blood}\times\mu_1$(Mat) [Pa·s]"
_MU2_TRIGGER_LABEL = r"$\mu_2$ trigger (FI) [−]"
# COMSOL Surface default "WaveLightClassic" ≈ matplotlib blue–white–red (``bwr``).
_MU_VIZ_CMAP_DEFAULT = "bwr"
# COMSOL WaveLightClassic legend on ``mu_b*(mu2+mu1)`` exports (patient007 t_final reference).
_MU_VIZ_VMIN_DEFAULT = 0.04
_MU_VIZ_VMAX_DEFAULT = 0.10


def _viz_mu_cmap() -> str:
    """Colormap for dynamic μ and μ₁-product panels (override: ``VIZ_MU_CMAP``, e.g. ``magma``)."""
    raw = (os.environ.get("VIZ_MU_CMAP") or _MU_VIZ_CMAP_DEFAULT).strip()
    return raw or _MU_VIZ_CMAP_DEFAULT


def _viz_mu_clim_fixed() -> bool:
    """True unless ``VIZ_MU_CLIM=auto`` (data min/max across panels)."""
    raw = (os.environ.get("VIZ_MU_CLIM") or "fixed").strip().lower()
    return raw not in ("auto", "data", "dynamic")


def _viz_mu_si_clim(*arrays: np.ndarray) -> Tuple[float, float]:
    """
    Color limits [Pa·s] for μ panels.

    Default: fixed COMSOL-style window (``VIZ_MU_VMIN`` / ``VIZ_MU_VMAX``, else 0.04–0.10).
    Set ``VIZ_MU_CLIM=auto`` to span the passed arrays (legacy behavior).
    """
    if _viz_mu_clim_fixed():
        vmin_s = (os.environ.get("VIZ_MU_VMIN") or "").strip()
        vmax_s = (os.environ.get("VIZ_MU_VMAX") or "").strip()
        vmin = float(vmin_s) if vmin_s else _MU_VIZ_VMIN_DEFAULT
        vmax = float(vmax_s) if vmax_s else _MU_VIZ_VMAX_DEFAULT
        if vmax <= vmin:
            vmax = vmin + 1e-12
        return vmin, vmax
    nonempty = [a for a in arrays if a.size > 0]
    if not nonempty:
        return _MU_VIZ_VMIN_DEFAULT, _MU_VIZ_VMAX_DEFAULT
    vmin = min(float(a.min()) for a in nonempty)
    vmax = max(float(a.max()) for a in nonempty)
    if vmax <= vmin + 1e-18:
        vmax = vmin + 1e-12
    return vmin, vmax


def _biochem_mu_viz_labels() -> Dict[str, str]:
    """Panel titles aligned with the active forward ``μ_eff`` path."""
    if _biochem_mu_simple_log_residual_enabled():
        mu_dyn = r"$\mu_{\mathrm{eff}}$ (rollout, $\mu_{\mathrm{kin}}\,e^{\Delta\log\mu}$) [Pa·s]"
        gel_suffix = "effective μ₁/μ₂ in forward = 0 (simple log residual)"
    elif _biochem_mu_disable_explicit_gelation():
        mu_dyn = r"$\mu_{\mathrm{eff}}$ (rollout) [Pa·s]"
        gel_suffix = "effective μ₁/μ₂ in forward = 0"
    else:
        mu_dyn = r"$\mu_{\mathrm{eff}}$ (rollout) [Pa·s]"
        gel_suffix = "effective μ₁/μ₂ in forward"
    return {
        "mu_dynamic": mu_dyn,
        "mu1_product": r"$\mu_{\mathrm{blood}}\times\mu_1$ (effective) [Pa·s]",
        "mu2_trigger": r"$\mu_2$ (effective in forward) [−]",
        "gelation_suffix": gel_suffix,
    }


def _rollout_mu_eff_si_numpy(phys_cfg: PhysicsConfig, pred_np: np.ndarray) -> np.ndarray:
    """Stored rollout viscosity channel (ND → SI); matches kinematic coupling in forward."""
    mu_ch = STATE_CHANNEL_MU_EFF_ND
    mu_nd = torch.from_numpy(pred_np[:, mu_ch]).float().view(-1, 1)
    return phys_cfg.viscosity_nd_to_si(mu_nd).detach().cpu().numpy().reshape(-1)


def _carreau_mu_blood_torch(
    model: GNODE_Phase3,
    data,
    pred_t: torch.Tensor,
) -> torch.Tensor:
    """Shear-thinning baseline ``μ_b`` from decoded ``u,v`` (SI)."""
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
    mu_blood = mu_inf + (mu_0 - mu_inf) * torch.pow(
        1.0 + (lam * gamma_dot) ** 2, (n_idx - 1.0) / 2.0
    )
    return mu_blood.squeeze(-1)


@dataclass(frozen=True)
class BiochemCheckpointChoice:
    path: Path
    role: str
    explicit: bool
    fell_back_from_teacher: bool


def _biochem_ckpt_search_dir() -> Path:
    return resolve_checkpoint("b", _BIOCHEM_CKPT_CANDIDATES[0]).parent


def _format_biochem_ckpt_inventory() -> str:
    ckpt_dir = _biochem_ckpt_search_dir()
    lines = [f"  directory: {ckpt_dir}"]
    for name in _BIOCHEM_CKPT_CANDIDATES + _BIOCHEM_CKPT_INVENTORY_EXTRA:
        path = ckpt_dir / name
        role = _BIOCHEM_CKPT_ROLE_BY_NAME.get(name, "unknown")
        if path.is_file():
            lines.append(f"  - {name}  [{role}]  (present)")
        else:
            lines.append(f"  - {name}  [{role}]  (missing)")
    return "\n".join(lines)


def _infer_role_from_meta(meta: Dict[str, Any], filename: str) -> str:
    role = (meta.get("checkpoint_role") or "").strip()
    if role:
        return role
    return _BIOCHEM_CKPT_ROLE_BY_NAME.get(filename, "custom")


def _print_biochem_checkpoint_banner(choice: BiochemCheckpointChoice, meta: Dict[str, Any]) -> None:
    """Explain which ``train_biochem_corrector`` artifact is driving visualization."""
    role = _infer_role_from_meta(meta, choice.path.name)
    role_labels = {
        "teacher_best_high_mu": "global-best teacher (lowest val mu_log_mae_high_mu)",
        "teacher_last": "most recent teacher run (backup)",
        "teacher_best_all_legacy": "legacy global teacher all-truth (biochem_teacher_best.pth)",
        "corrector_best_high_mu": "corrector global high-μ (legacy filename)",
        "corrector_best_bio_legacy": "legacy corrector best (composite; deprecated)",
        "corrector_latest": "full corrector — latest resume snapshot",
        "custom": "user-specified path",
    }
    print("\n📦 Biochem checkpoint selection (from train_biochem_corrector.py):")
    print(f"   ↳ file: {choice.path.resolve()}")
    print(f"   ↳ role: {role} — {role_labels.get(role, role_labels['custom'])}")
    if choice.explicit:
        print("   ↳ source: --biochem-checkpoint or VIZ_BIOCHEM_CHECKPOINT")
    elif role in _BIOCHEM_TEACHER_CKPT_ROLES:
        print(
            "   ↳ source: teacher checkpoint preference "
            f"({' → '.join(_BIOCHEM_TEACHER_CKPT_CANDIDATES)})"
        )
    else:
        print(
            "   ↳ source: default preference order "
            f"({' → '.join(_BIOCHEM_CKPT_CANDIDATES)}) — first existing file wins"
        )
    t_ep = meta.get("best_epoch", -1)
    t_mae = meta.get("val_mu_log_mae")
    t_high = meta.get("val_mu_log_mae_high_mu")
    run_note = (meta.get("run_note") or "").strip()
    if isinstance(t_ep, int) and t_ep >= 0:
        print(f"   ↳ saved at teacher/corrector epoch: {int(t_ep)}")
    if t_mae is not None:
        try:
            print(f"   ↳ val mu_log_mae all (stored in ckpt): {float(t_mae):.4f}")
        except (TypeError, ValueError):
            pass
    if t_high is not None:
        try:
            print(f"   ↳ val mu_log_mae high-μ (stored in ckpt): {float(t_high):.4f}")
        except (TypeError, ValueError):
            pass
    if run_note:
        print(f"   ↳ run_note: {run_note}")
    print("   ↳ on-disk inventory:")
    for line in _format_biochem_ckpt_inventory().splitlines():
        print(line)


def _anchor_graph_dir() -> Path:
    return Path(VesselConfig(phase="biochem_anchors").graph_output_dir)


def _list_anchor_graph_paths() -> List[Path]:
    anchor_dir = _anchor_graph_dir()
    if not anchor_dir.exists():
        return []
    return sorted(anchor_dir.glob("*.pt"))


def _list_anchor_stems() -> List[str]:
    return [p.stem for p in _list_anchor_graph_paths()]


def _default_val_anchor_stem(stems: List[str]) -> str:
    if not stems:
        raise FileNotFoundError(
            f"No anchor graphs found under {_anchor_graph_dir()}. "
            "Export COMSOL anchors first (see src/data_gen/pipeline_biochem.py) "
            "or pass --synthetic to visualize a generated vessel."
        )
    if _DEFAULT_VAL_ANCHOR_STEM in stems:
        return _DEFAULT_VAL_ANCHOR_STEM
    n = len(stems)
    split_idx = int(0.9 * n)
    split_idx = max(1, min(split_idx, n - 1))
    return stems[split_idx]


def _resolve_anchor_stem(explicit: Optional[str] = None) -> str:
    env_stem = (os.environ.get("VIZ_ANCHOR_STEM") or "").strip()
    stem = (explicit or env_stem or "").strip()
    stems = _list_anchor_stems()
    if stem:
        if stem.endswith(".pt"):
            stem = Path(stem).stem
        if stem not in stems:
            raise FileNotFoundError(
                f"Anchor graph '{stem}' not found under {_anchor_graph_dir()}. "
                f"Available: {', '.join(stems) or '(none)'}"
            )
        return stem
    return _default_val_anchor_stem(stems)


def _load_anchor_graph(stem: str, device: torch.device):
    path = _anchor_graph_dir() / f"{stem}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Anchor graph not found: {path}")
    data = torch.load(path, weights_only=False)
    data = infer_missing_schema(data, phase_hint="biochem")
    return data.to(device), path


def _graph_has_comsol_trajectory(data) -> bool:
    if not hasattr(data, "y") or data.y is None or not torch.is_tensor(data.y):
        return False
    if data.y.dim() == 3 and data.y.shape[0] >= 1 and data.y.shape[-1] >= 16:
        return True
    return data.y.dim() == 2 and data.y.shape[-1] >= 16


def _nearest_time_indices(times_si: torch.Tensor, query_times: List[float]) -> List[int]:
    t = times_si.reshape(-1).to(dtype=torch.float32)
    return [int(torch.argmin(torch.abs(t - float(q))).item()) for q in query_times]


def _extract_state_series_np(data, frame_indices: List[int]) -> np.ndarray:
    """COMSOL / anchor labels as ``[T, N, C]`` numpy (steady graphs broadcast to each frame)."""
    y = data.y
    if y.dim() == 2:
        y0 = y.detach().cpu().numpy()
        return np.stack([y0 for _ in frame_indices], axis=0)
    if y.dim() != 3:
        raise ValueError(f"Unsupported anchor y shape {tuple(y.shape)}; expected [T,N,C] or [N,C].")
    idx = [max(0, min(int(i), y.shape[0] - 1)) for i in frame_indices]
    return y[idx].detach().cpu().numpy()


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


def _viz_fast_enabled(explicit: Optional[bool] = None) -> bool:
    if explicit is not None:
        return explicit
    return os.environ.get("VIZ_FAST", "1").strip().lower() not in ("0", "false", "no", "off")


def _slider_keyframe_times_si(
    dense_times_si_full: torch.Tensor,
    t_final_si: float,
    extend_mult: float,
) -> List[float]:
    """Slider keyframes aligned to the anchor/export grid (avoids arbitrary 33/66% gaps)."""
    n = int(dense_times_si_full.numel())
    if n >= 4:
        fracs = (0.0, 0.33, 0.66, 1.0)
        idxs = [min(n - 1, max(0, int(round(f * (n - 1))))) for f in fracs]
        times = [float(dense_times_si_full[i].item()) for i in idxs]
    else:
        times = [0.0, t_final_si * 0.33, t_final_si * 0.66, t_final_si]
    times.append(float(t_final_si * extend_mult))
    return times


@contextlib.contextmanager
def _viz_biochem_ode_speedups() -> Iterator[None]:
    """Plain ``odeint`` + coarser RK + COMSOL-like ODE steps for visualization unless already set."""
    keys = ("BIOCHEM_ODEINT_USE_ADJOINT", "BIOCHEM_ADJOINT_RK4_SUBSTEPS", "BIOCHEM_ODE_MAX_STEP_S")
    saved = {k: os.environ.get(k) for k in keys}
    try:
        if saved["BIOCHEM_ODEINT_USE_ADJOINT"] is None:
            os.environ["BIOCHEM_ODEINT_USE_ADJOINT"] = "0"
        if saved["BIOCHEM_ADJOINT_RK4_SUBSTEPS"] is None:
            os.environ["BIOCHEM_ADJOINT_RK4_SUBSTEPS"] = "4"
        if saved["BIOCHEM_ODE_MAX_STEP_S"] is None:
            os.environ["BIOCHEM_ODE_MAX_STEP_S"] = "150"
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
    n_proc_extra = 0
    if hasattr(model, "kin_processor_extra"):
        n_proc_extra = inject_lora_to_spectral_linears(
            model.kin_processor_extra, rank=rank, alpha=alpha
        )
    n_dec = 0
    if getattr(model, "kinematics_decoder", None) is not None:
        n_dec = inject_lora_to_spectral_linears(model.kinematics_decoder, rank=rank, alpha=alpha)
    print(
        f"   ↳ LoRA injected: kin_encoder={n_enc}, kin_processor={n_proc + n_proc_extra}, "
        f"kinematics_decoder={n_dec} (rank={rank}, alpha={alpha}; SIREN ckpts use decoder=0)"
    )


def _try_resolve_kinematics_checkpoint() -> Optional[Path]:
    for ckpt_name in _KIN_CKPT_CANDIDATES:
        candidate = resolve_checkpoint("a", ckpt_name)
        if candidate.exists():
            return candidate
    return None


def _resolve_kinematics_checkpoint() -> Path:
    found = _try_resolve_kinematics_checkpoint()
    if found is not None:
        return found
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


def _resolve_biochem_checkpoint(
    explicit: Optional[str] = None,
    *,
    teacher_only: bool = False,
) -> BiochemCheckpointChoice:
    """Prefer best high-μ, then latest resume, then global teacher-best; override via CLI."""
    require_teacher = teacher_only or (
        os.environ.get("VIZ_BIOCHEM_REQUIRE_TEACHER", "").strip().lower() in ("1", "true", "yes", "on")
    )
    name = (explicit or os.environ.get("VIZ_BIOCHEM_CHECKPOINT") or "").strip()
    if name:
        path = Path(name)
        if path.is_file():
            resolved = path.resolve()
        else:
            resolved = resolve_checkpoint("b", name)
            if not resolved.exists():
                raise FileNotFoundError(f"Biochem checkpoint not found: {name}")
        role = _BIOCHEM_CKPT_ROLE_BY_NAME.get(resolved.name, "custom")
        if require_teacher and role not in _BIOCHEM_TEACHER_CKPT_ROLES:
            raise FileNotFoundError(
                f"--teacher-only but checkpoint '{resolved.name}' (role={role}) is not a teacher artifact. "
                f"Use one of: {', '.join(_BIOCHEM_TEACHER_CKPT_CANDIDATES)}."
            )
        return BiochemCheckpointChoice(
            path=resolved,
            role=role,
            explicit=True,
            fell_back_from_teacher=False,
        )

    search_names = _BIOCHEM_TEACHER_CKPT_CANDIDATES if require_teacher else _BIOCHEM_CKPT_CANDIDATES
    chosen: Optional[Path] = None
    for ckpt_name in search_names:
        candidate = resolve_checkpoint("b", ckpt_name)
        if candidate.exists():
            chosen = candidate
            break
    if chosen is None:
        expected_dir = _biochem_ckpt_search_dir()
        hint = (
            "Train teacher with BIOCHEM_STOP_AFTER_TEACHER=1 (writes biochem_teacher_last.pth + bests)."
            if require_teacher
            else "Train teacher or corrector to produce outputs/biochem/*.pth checkpoints."
        )
        raise FileNotFoundError(
            "No biochem checkpoint found for visualization. Tried: "
            + ", ".join(str(expected_dir / n) for n in search_names)
            + f". {hint}\n{_format_biochem_ckpt_inventory()}"
        )
    role = _BIOCHEM_CKPT_ROLE_BY_NAME.get(chosen.name, "custom")
    return BiochemCheckpointChoice(
        path=chosen,
        role=role,
        explicit=False,
        fell_back_from_teacher=False,
    )


def _load_single_graph(proc_dir, device, label):
    files = sorted(proc_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No graph files found in {proc_dir} for {label}")
    return torch.load(files[0], weights_only=False).to(device)


def _run_model_once(model, data):
    pred = model(data)
    return pred[0] if isinstance(pred, tuple) else pred


def _comsol_style_rheology_fields(
    model: GNODE_Phase3,
    data,
    pred_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """COMSOL reference display: Carreau ``μ_b`` × ``(1+μ₁+μ₂)`` from species sigmoids."""
    device = pred_t.device
    dtype = pred_t.dtype
    mu_blood = _carreau_mu_blood_torch(model, data, pred_t)
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
    mu_dynamic_si = mu_blood.unsqueeze(-1) * (1.0 + mu1 + mu2)
    return (
        mu_blood,
        mu1.squeeze(-1),
        mu2.squeeze(-1),
        mu_dynamic_si.squeeze(-1),
    )


def _biochem_rollout_rheology_fields(
    model: GNODE_Phase3,
    data,
    pred_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Biochem panels: stored ``μ_eff`` rollout + effective explicit μ₁/μ₂ (respects ablation flags)."""
    mu_dynamic_si = model.phys_cfg.viscosity_nd_to_si(
        pred_t[:, STATE_CHANNEL_MU_EFF_ND : STATE_CHANNEL_MU_EFF_ND + 1]
    ).squeeze(-1)
    mu_blood = _carreau_mu_blood_torch(model, data, pred_t)
    sp_safe = torch.clamp(
        pred_t[:, 4:16],
        min=_SPECIES_LOG1P_MIN,
        max=_SPECIES_LOG1P_MAX,
    )
    species_si = model.species_log_nd_to_si(sp_safe)
    mu1, mu2 = biochem_explicit_gelation_terms(
        model, species_si[:, 8:9], species_si[:, 11:12]
    )
    return (
        mu_blood,
        mu1.squeeze(-1),
        mu2.squeeze(-1),
        mu_dynamic_si,
    )


def _rheology_series_numpy(
    model: GNODE_Phase3,
    data,
    pred_series_np: np.ndarray,
    *,
    biochem_rollout: bool = True,
) -> np.ndarray:
    """``pred_series_np`` ``[T,N,C]`` → ``μ_eff`` in SI ``[T,N]`` (rollout channel or COMSOL-style)."""
    device = data.x.device
    dtype = torch.float32
    t_steps = int(pred_series_np.shape[0])
    n = int(pred_series_np.shape[1])
    out = np.zeros((t_steps, n), dtype=np.float64)
    with torch.no_grad():
        for ti in range(t_steps):
            if biochem_rollout:
                pred_t = torch.from_numpy(pred_series_np[ti]).to(device=device, dtype=dtype)
                _, _, _, mu_dyn = _biochem_rollout_rheology_fields(model, data, pred_t)
                out[ti] = mu_dyn.detach().cpu().numpy()
            else:
                out[ti] = _rollout_mu_eff_si_numpy(model.phys_cfg, pred_series_np[ti])
    return out


def _rheology_trigger_series_numpy(
    model: GNODE_Phase3,
    data,
    pred_series_np: np.ndarray,
    *,
    biochem_rollout: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """``[T,N,C]`` → ``(μ_blood×μ_1, μ_2)`` effective (biochem) or COMSOL-style sigmoids."""
    device = data.x.device
    dtype = torch.float32
    t_steps = int(pred_series_np.shape[0])
    n = int(pred_series_np.shape[1])
    mb_mu1 = np.zeros((t_steps, n), dtype=np.float64)
    mu2_out = np.zeros((t_steps, n), dtype=np.float64)
    fields_fn = _biochem_rollout_rheology_fields if biochem_rollout else _comsol_style_rheology_fields
    with torch.no_grad():
        for ti in range(t_steps):
            pred_t = torch.from_numpy(pred_series_np[ti]).to(device=device, dtype=dtype)
            mu_blood, mu1, mu2, _ = fields_fn(model, data, pred_t)
            mb_mu1[ti] = (mu_blood * mu1).detach().cpu().numpy()
            mu2_out[ti] = mu2.detach().cpu().numpy()
    return mb_mu1, mu2_out


def _trigger_fields_numpy(
    model: GNODE_Phase3,
    data,
    pred_np: np.ndarray,
    *,
    biochem_rollout: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Single-frame ``[N,C]`` → ``(μ_blood×μ_1, μ_2)`` for gelation comparison figure."""
    device = data.x.device
    pred_t = torch.from_numpy(pred_np).to(device=device, dtype=torch.float32)
    fields_fn = _biochem_rollout_rheology_fields if biochem_rollout else _comsol_style_rheology_fields
    with torch.no_grad():
        mu_blood, mu1, mu2, _ = fields_fn(model, data, pred_t)
    return (
        (mu_blood * mu1).detach().cpu().numpy(),
        mu2.detach().cpu().numpy(),
    )


def _show_mu_trigger_comparison_figure(
    pos: np.ndarray,
    model_biochem: GNODE_Phase3,
    mb_mu1_ml: np.ndarray,
    mu2_ml: np.ndarray,
    *,
    case_label: str = "",
    time_label: str = "",
    mb_mu1_comsol: Optional[np.ndarray] = None,
    mu2_comsol: Optional[np.ndarray] = None,
    mu_labels: Optional[Dict[str, str]] = None,
) -> None:
    """Static gelation triggers at final rollout time; optional COMSOL vs ML 2×2 grid."""
    when = f" at {time_label}" if time_label else " at final time"
    labels = mu_labels or _biochem_mu_viz_labels()
    mu1_lbl = labels["mu1_product"]
    mu2_lbl = labels["mu2_trigger"]
    gel_suffix = labels["gelation_suffix"]
    mu2_cap = float(model_biochem.mu_ratio_max)
    m1_arrays: List[np.ndarray] = [mb_mu1_ml]
    if mb_mu1_comsol is not None:
        m1_arrays.append(mb_mu1_comsol)
    m1_vmin, m1_vmax = _viz_mu_si_clim(*m1_arrays)

    show_comsol = mb_mu1_comsol is not None and mu2_comsol is not None
    if show_comsol:
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        title = (
            f"Gelation triggers{when}: COMSOL vs Biochem ({case_label}; {gel_suffix})"
            if case_label
            else f"Gelation triggers{when}: COMSOL vs Biochem ({gel_suffix})"
        )
        panels = (
            (axs[0, 0], mb_mu1_comsol, f"COMSOL: {_MU1_PRODUCT_SI_LABEL}", _viz_mu_cmap(), m1_vmin, m1_vmax),
            (axs[0, 1], mu2_comsol, f"COMSOL: {_MU2_TRIGGER_LABEL}", "Reds", 0.0, mu2_cap),
            (axs[1, 0], mb_mu1_ml, f"Biochem: {mu1_lbl}", _viz_mu_cmap(), m1_vmin, m1_vmax),
            (axs[1, 1], mu2_ml, f"Biochem: {mu2_lbl}", "Reds", 0.0, mu2_cap),
        )
        for ax, values, subtitle, cmap, vmin, vmax in panels:
            _plot_field(fig, ax, pos, values, subtitle, cmap, vmin=vmin, vmax=vmax)
    else:
        fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
        title = (
            f"Gelation triggers{when} ({case_label}; {gel_suffix})"
            if case_label
            else f"Gelation triggers{when} ({gel_suffix})"
        )
        _plot_field(
            fig,
            axs[0],
            pos,
            mb_mu1_ml,
            mu1_lbl,
            _viz_mu_cmap(),
            vmin=m1_vmin,
            vmax=m1_vmax,
        )
        _plot_field(
            fig,
            axs[1],
            pos,
            mu2_ml,
            mu2_lbl,
            "Reds",
            vmin=0.0,
            vmax=mu2_cap,
        )
    fig.tight_layout(rect=(0, 0.03, 1, 0.90))
    _set_figure_suptitle(fig, title, fontsize=14)


def _mesh_axis_limits(pos: np.ndarray, pad_frac: float = 0.04) -> Tuple[float, float, float, float]:
    x0, x1 = float(pos[:, 0].min()), float(pos[:, 0].max())
    y0, y1 = float(pos[:, 1].min()), float(pos[:, 1].max())
    dx = max((x1 - x0) * pad_frac, 1e-9)
    dy = max((y1 - y0) * pad_frac, 1e-9)
    return x0 - dx, x1 + dx, y0 - dy, y1 + dy


def _set_figure_suptitle(
    fig,
    title: str,
    *,
    fontsize: float = 14,
    subplot_top: float = 0.88,
    title_y: float = 0.96,
) -> None:
    """Reserve headroom so the main title stays visible when the window is maximized."""
    fig.subplots_adjust(top=subplot_top)
    fig.suptitle(title, fontsize=fontsize, fontweight="bold", y=title_y)


def _show_kinematics_static_figure(
    pos: np.ndarray,
    vel: np.ndarray,
    pressure: np.ndarray,
    mu_eff: np.ndarray,
    *,
    case_label: str = "",
) -> None:
    """Steady GINO-DEQ solve (not time-dependent) — separate from the biochem temporal slider."""
    xlo, xhi, ylo, yhi = _mesh_axis_limits(pos)
    xspan = max(xhi - xlo, 1e-9)
    yspan = max(yhi - ylo, 1e-9)
    panel_aspect = yspan / xspan
    panel_h = 4.2
    panel_w = min(panel_h * panel_aspect, 5.5)
    fig, axes = plt.subplots(1, 3, figsize=(panel_w * 3.35, panel_h + 0.55))
    title = f"Kinematics (GINO-DEQ), steady — {case_label}" if case_label else "Kinematics (GINO-DEQ), steady"
    panels = (
        (vel, "|u| (ND)", "jet"),
        (pressure, "p (ND)", "coolwarm"),
        (mu_eff, r"$\mu_{eff}$ (ND)", "viridis"),
    )
    for ax, (values, subtitle, cmap) in zip(np.atleast_1d(axes), panels):
        _plot_field(
            fig,
            ax,
            pos,
            values,
            subtitle,
            cmap,
            tight_axes=True,
            xlim=(xlo, xhi),
            ylim=(ylo, yhi),
        )
    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.06, wspace=0.14)
    _set_figure_suptitle(fig, title, fontsize=15, subplot_top=0.86, title_y=0.95)


def _vel_pressure_from_series(series_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    u = series_np[:, :, _CHANNEL["u"]]
    v = series_np[:, :, _CHANNEL["v"]]
    vel = np.sqrt(u ** 2 + v ** 2)
    p = series_np[:, :, _CHANNEL["p"]]
    return vel, p


def _plot_field(
    fig,
    ax,
    pos,
    val,
    title,
    cmap,
    vmin=None,
    vmax=None,
    *,
    tight_axes: bool = False,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
):
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

    tc = ax.tripcolor(triang, val, cmap=cmap, shading="gouraud", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11, pad=6)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if tight_axes:
        ax.set_aspect("equal", adjustable="box")
    else:
        ax.set_aspect("equal")
    ax.axis("off")
    fig.colorbar(tc, ax=ax, fraction=0.042, pad=0.02, shrink=0.88)


def _species_si_from_series(series_np: np.ndarray, scales: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    fib = np.expm1(np.clip(series_np[:, :, 12], a_min=0.0, a_max=None)) * scales[8]
    mat = np.expm1(np.clip(series_np[:, :, 15], a_min=0.0, a_max=None)) * scales[11]
    return fib, mat


def _show_biochem_temporal_slider(
    pos,
    pred_biochem_series_np,
    custom_times,
    model_biochem: GNODE_Phase3,
    data_biochem,
    on_refresh=None,
    comsol_series_np: Optional[np.ndarray] = None,
    title_prefix: str = "Biochem Temporal Inspector",
    refresh_label: str = "Refresh",
):
    vel_all, _ = _vel_pressure_from_series(pred_biochem_series_np)
    bio_cfg = BiochemConfig(phase="biochem")
    mu_labels = _biochem_mu_viz_labels()
    scales = bio_cfg.get_species_scales(device="cpu").cpu().numpy()
    fib_all, mat_all = _species_si_from_series(pred_biochem_series_np, scales)
    mu_dyn_all = _rheology_series_numpy(
        model_biochem, data_biochem, pred_biochem_series_np, biochem_rollout=True
    )
    mb_mu1_all, mu2_all = _rheology_trigger_series_numpy(
        model_biochem, data_biochem, pred_biochem_series_np, biochem_rollout=True
    )
    mu2_cap = float(model_biochem.mu_ratio_max)

    show_comsol = comsol_series_np is not None
    ncols = 2 if show_comsol else 1
    col_b, col_c = 0, (1 if show_comsol else None)

    comsol_vel = comsol_fib = comsol_mat = comsol_mu_dyn = None
    comsol_mb_mu1 = comsol_mu2 = None
    if show_comsol:
        comsol_vel, _ = _vel_pressure_from_series(comsol_series_np)
        comsol_fib, comsol_mat = _species_si_from_series(comsol_series_np, scales)
        comsol_mu_dyn = _rheology_series_numpy(
            model_biochem, data_biochem, comsol_series_np, biochem_rollout=False
        )
        comsol_mb_mu1, comsol_mu2 = _rheology_trigger_series_numpy(
            model_biochem, data_biochem, comsol_series_np, biochem_rollout=False
        )

    vel_vmin = float(vel_all.min())
    vel_vmax = float(vel_all.max())
    if show_comsol:
        vel_vmin = min(vel_vmin, float(comsol_vel.min()))
        vel_vmax = max(vel_vmax, float(comsol_vel.max()))

    fib_vmin = float(min(fib_all.min(), comsol_fib.min() if show_comsol else fib_all.min()))
    fib_vmax = float(max(fib_all.max(), comsol_fib.max() if show_comsol else fib_all.max()))
    mat_vmin = float(min(mat_all.min(), comsol_mat.min() if show_comsol else mat_all.min()))
    mat_vmax = float(max(mat_all.max(), comsol_mat.max() if show_comsol else mat_all.max()))
    mu_clim_arrays: List[np.ndarray] = [mu_dyn_all.reshape(-1)]
    m1_clim_arrays: List[np.ndarray] = [mb_mu1_all.reshape(-1)]
    if show_comsol:
        mu_clim_arrays.append(comsol_mu_dyn.reshape(-1))
        m1_clim_arrays.append(comsol_mb_mu1.reshape(-1))
    mu_vmin, mu_vmax = _viz_mu_si_clim(*mu_clim_arrays)
    m1_vmin, m1_vmax = _viz_mu_si_clim(*m1_clim_arrays)

    t0_label = f"t={custom_times[0]:.1f}s"

    def _init_scatter(fig, ax, values, title, cmap, vmin, vmax):
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=values[0], cmap=cmap, s=3, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        return sc

    # --- Species (SI): separate figure so the main inspector stays compact ---
    fig_sp, axs_sp = plt.subplots(2, ncols, figsize=(7 * ncols, 8))
    if ncols == 1:
        axs_sp = np.asarray(axs_sp).reshape(2, 1)
    _set_figure_suptitle(fig_sp, f"{title_prefix} — Species (SI) ({t0_label})", fontsize=14)
    scatters_sp: Dict[str, Any] = {}
    scatters_sp["fib_b"] = _init_scatter(
        fig_sp, axs_sp[0, col_b], fib_all, "Biochem: Fibrin (SI)", "Reds", fib_vmin, fib_vmax
    )
    scatters_sp["mat_b"] = _init_scatter(
        fig_sp,
        axs_sp[1, col_b],
        mat_all,
        "Biochem: Surface Platelets (SI)",
        "Oranges",
        mat_vmin,
        mat_vmax,
    )
    if show_comsol:
        scatters_sp["fib_c"] = _init_scatter(
            fig_sp, axs_sp[0, col_c], comsol_fib, "COMSOL: Fibrin (SI)", "Reds", fib_vmin, fib_vmax
        )
        scatters_sp["mat_c"] = _init_scatter(
            fig_sp,
            axs_sp[1, col_c],
            comsol_mat,
            "COMSOL: Surface Platelets (SI)",
            "Oranges",
            mat_vmin,
            mat_vmax,
        )

    # --- Main: flow + effective viscosity + gelation triggers ---
    nrows_main = 4
    fig, axs = plt.subplots(nrows_main, ncols, figsize=(7 * ncols, 15))
    if ncols == 1:
        axs = np.asarray(axs).reshape(nrows_main, 1)
    plt.subplots_adjust(bottom=0.08, top=0.90, hspace=0.22)
    _set_figure_suptitle(
        fig, f"{title_prefix} — Flow & rheology ({t0_label})", fontsize=16, subplot_top=0.90, title_y=0.97
    )

    scatters: Dict[str, Any] = {}
    scatters["vel_b"] = _init_scatter(
        fig, axs[0, col_b], vel_all, "Biochem: |u| (ND)", "jet", vel_vmin, vel_vmax
    )
    scatters["mu_b"] = _init_scatter(
        fig, axs[1, col_b], mu_dyn_all, f"Biochem: {mu_labels['mu_dynamic']}", _viz_mu_cmap(), mu_vmin, mu_vmax
    )
    scatters["m1_b"] = _init_scatter(
        fig, axs[2, col_b], mb_mu1_all, f"Biochem: {mu_labels['mu1_product']}", _viz_mu_cmap(), m1_vmin, m1_vmax
    )
    scatters["m2_b"] = _init_scatter(
        fig, axs[3, col_b], mu2_all, f"Biochem: {mu_labels['mu2_trigger']}", "Reds", 0.0, mu2_cap
    )

    if show_comsol:
        scatters["vel_c"] = _init_scatter(
            fig, axs[0, col_c], comsol_vel, "COMSOL: |u| (ND)", "jet", vel_vmin, vel_vmax
        )
        scatters["mu_c"] = _init_scatter(
            fig,
            axs[1, col_c],
            comsol_mu_dyn,
            f"COMSOL: {mu_labels['mu_dynamic']}",
            _viz_mu_cmap(),
            mu_vmin,
            mu_vmax,
        )
        scatters["m1_c"] = _init_scatter(
            fig, axs[2, col_c], comsol_mb_mu1, f"COMSOL: {_MU1_PRODUCT_SI_LABEL}", _viz_mu_cmap(), m1_vmin, m1_vmax
        )
        scatters["m2_c"] = _init_scatter(
            fig, axs[3, col_c], comsol_mu2, f"COMSOL: {_MU2_TRIGGER_LABEL}", "Reds", 0.0, mu2_cap
        )

    n_frames = len(custom_times)
    fig.subplots_adjust(bottom=0.14)
    fig_sp.subplots_adjust(bottom=0.14)

    def _make_time_slider(parent_fig, bottom: float = 0.03) -> Slider:
        ax_slider = parent_fig.add_axes([0.15, bottom, 0.55, 0.03])
        return Slider(
            ax=ax_slider,
            label="Time",
            valmin=0,
            valmax=max(0, n_frames - 1),
            valinit=0,
            valstep=1,
            color="teal",
        )

    time_slider_main = _make_time_slider(fig)
    time_slider_sp = _make_time_slider(fig_sp)
    sliders = [time_slider_main, time_slider_sp]

    if on_refresh is not None:
        ax_refresh = fig.add_axes([0.74, 0.02, 0.2, 0.05])
        refresh_button = Button(ax_refresh, refresh_label, color="lightgray", hovercolor="gainsboro")
        refresh_button.on_clicked(lambda _: on_refresh())

    def _apply_frame(idx: int) -> None:
        idx = int(max(0, min(idx, n_frames - 1)))
        t_lbl = f"t={custom_times[idx]:.1f}s"
        scatters["vel_b"].set_array(vel_all[idx])
        scatters["mu_b"].set_array(mu_dyn_all[idx])
        scatters["m1_b"].set_array(mb_mu1_all[idx])
        scatters["m2_b"].set_array(mu2_all[idx])
        scatters_sp["fib_b"].set_array(fib_all[idx])
        scatters_sp["mat_b"].set_array(mat_all[idx])
        if show_comsol:
            scatters["vel_c"].set_array(comsol_vel[idx])
            scatters["mu_c"].set_array(comsol_mu_dyn[idx])
            scatters["m1_c"].set_array(comsol_mb_mu1[idx])
            scatters["m2_c"].set_array(comsol_mu2[idx])
            scatters_sp["fib_c"].set_array(comsol_fib[idx])
            scatters_sp["mat_c"].set_array(comsol_mat[idx])
        _set_figure_suptitle(
            fig, f"{title_prefix} — Flow & rheology ({t_lbl})", fontsize=16, subplot_top=0.88, title_y=0.96
        )
        _set_figure_suptitle(
            fig_sp,
            f"{title_prefix} — Species (SI) ({t_lbl})",
            fontsize=14,
            subplot_top=0.88,
            title_y=0.96,
        )
        fig.canvas.draw_idle()
        fig_sp.canvas.draw_idle()

    def _on_slider_change(val) -> None:
        idx = int(round(float(val)))
        for s in sliders:
            if int(round(float(s.val))) != idx:
                s.eventson = False
                s.set_val(idx)
                s.eventson = True
        _apply_frame(idx)

    for s in sliders:
        s.on_changed(_on_slider_change)

    if n_frames <= 1:
        print(
            "   ⚠️ Temporal slider: only one keyframe in this rollout "
            f"(n={n_frames}); use --full-viz or VIZ_BIOCHEM_MACRO_STEPS=full for more steps."
        )
    else:
        t_end = custom_times[-1]
        print(
            f"   ↳ Temporal slider: {n_frames} keyframes (0…{n_frames - 1}), "
            f"t ∈ [0, {t_end:.1f}] s — bottom of Species or Flow & rheology window."
        )
    plt.show()


def run_phase_comparison(
    source: str = "anchor",
    regenerate: bool = True,
    seed: int = 42,
    biochem_checkpoint: Optional[str] = None,
    anchor_stem: Optional[str] = None,
    teacher_only: bool = False,
    fast_viz: Optional[bool] = None,
):
    root = get_project_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_anchor = source != "synthetic"
    print(f"🖥️ Using device: {device}")
    if use_anchor:
        print(f"📍 Data source: COMSOL anchor graph")
    else:
        print(f"🎲 Geometry seed: {seed}")
    fast_mode = _viz_fast_enabled(fast_viz)
    print(
        f"⚡ Fast visualization mode: {'ON' if fast_mode else 'OFF'} "
        f"(dense COMSOL-spaced rollout; use --full-viz or VIZ_FAST=0 for high-fidelity)"
    )
    refresh_state = {"requested": False}

    data_kine_base = None
    anchor_path: Optional[Path] = None
    case_label = ""

    if use_anchor:
        stem = _resolve_anchor_stem(anchor_stem)
        case_label = stem
        print(f"   ↳ Anchor stem: {stem} ({_anchor_graph_dir()})")
        try:
            data_biochem, anchor_path = _load_anchor_graph(stem, device)
        except FileNotFoundError as exc:
            print(f"⚠️ {exc}")
            return False
        if not _graph_has_comsol_trajectory(data_biochem):
            print(
                f"⚠️ Anchor graph '{stem}' has no usable COMSOL labels in data.y. "
                "Re-export with extract_biochem_comsol_data.py or pick another stem."
            )
            return False
    else:
        # ------------------------------------------------------------------
        # Synthetic single-case setup (legacy behavior)
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
                any(raw_dir.glob("*.msh"))
                and any(graph_kine_base_dir.glob("*.pt"))
                and any(graph_biochem_dir.glob("*.pt"))
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
            mg1 = MeshToGraphComplete(
                phase="kinematics", raw_dir=raw_dir, label_dir=raw_dir, proc_dir=graph_kine_base_dir
            )
            mg1.run(max_files=1)
            mg3 = MeshToGraphPhase3(raw_dir=raw_dir, label_dir=raw_dir, proc_dir=graph_biochem_dir)
            mg3.run(max_files=1)
        else:
            print("\n♻️ Reusing existing single-case synthetic data.")

        try:
            data_kine_base = _load_single_graph(graph_kine_base_dir, device, "kinematics")
            data_biochem = _load_single_graph(graph_biochem_dir, device, "biochem")
            case_label = "synthetic"
        except FileNotFoundError as exc:
            print(f"⚠️ Failed to generate or load graph files: {exc}")
            return False

    pos = data_biochem.x[:, :2].cpu().numpy()
    n_nodes = int(data_biochem.x.shape[0])
    n_edges = int(data_biochem.edge_index.shape[1])
    print(f"   ↳ Graph: {n_nodes} nodes, {n_edges} edges", flush=True)

    # ------------------------------------------------------------------
    # 4. Load Models
    # ------------------------------------------------------------------
    print("\n🧠 Loading trained models...")
    biochem_choice = _resolve_biochem_checkpoint(biochem_checkpoint, teacher_only=teacher_only)
    biochem_ckpt = biochem_choice.path
    biochem_meta, biochem_state = _checkpoint_state_dict(_load_torch_checkpoint(biochem_ckpt))
    _print_biochem_checkpoint_banner(biochem_choice, biochem_meta)
    fp = biochem_forward_policy_from_checkpoint_meta(biochem_meta)
    if fp is not None:
        apply_biochem_forward_policy_from_checkpoint_meta(biochem_meta, quiet=True)
    else:
        print(
            "   ⚠️ No forward_policy in checkpoint — μ rollout uses current shell env. "
            "Re-train/save teacher to embed policy, or set BIOCHEM_* manually.",
            flush=True,
        )

    model_kine_base = None
    kin_ckpt = _try_resolve_kinematics_checkpoint()
    if kin_ckpt is not None:
        print(f"   ↳ Using kinematics checkpoint: {kin_ckpt.name}")
        kin_default_iters = "12" if fast_mode else "25"
        kin_max_iters = int(os.environ.get("VIZ_KIN_MAX_ITERS", kin_default_iters))
        kin_max_iters = max(5, min(80, kin_max_iters))
        phys_cfg_kine = PhysicsConfig(phase="kinematics")
        kin_raw = _load_torch_checkpoint(kin_ckpt)
        kin_meta, kin_state = kinematics_checkpoint_tensors(kin_raw)
        kin_ctor = resolve_gino_deq_ctor_kwargs(kin_meta, kin_state)
        kin_ctor = dict(kin_ctor)
        kin_ctor["max_iters"] = kin_max_iters
        if kin_meta.get("model_config"):
            print("   ↳ kinematics GINO_DEQ from checkpoint model_config.")
        else:
            print(
                "   ↳ kinematics GINO_DEQ from weight/reference inference "
                "(re-save kinematics_best.pth to embed model_config)."
            )
        model_kine_base = build_gino_deq_from_ctor(phys_cfg_kine, kin_ctor).to(device)
        model_kine_base.load_state_dict(kin_state, strict=False)
        model_kine_base.eval()
    elif not use_anchor:
        print("   ↳ No kinematics checkpoint found; skipping GINO-DEQ column.", flush=True)

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
    print(f"   ↳ bio_encoder prior dim: {bio_enc_prior}")
    print(f"   ↳ latent_dim: {latent_dim}")
    _viz_inner = os.environ.get("VIZ_BIOCHEM_MAX_INNER_ITERS", "").strip()
    biochem_inner_iters = int(_viz_inner) if _viz_inner else (6 if fast_mode else 10)
    biochem_inner_iters = max(3, min(25, biochem_inner_iters))
    ctor = resolve_gnode_phase3_ctor_kwargs(
        biochem_meta,
        biochem_state,
        bio_encoder_prior_dim_default=bio_enc_prior,
        latent_dim_default=latent_dim,
        max_inner_iters_default=biochem_inner_iters,
    )
    if biochem_meta.get("model_config"):
        fp_note = format_biochem_forward_policy_summary(fp)
        if fp_note:
            print(
                f"   ↳ biochem GNODE from checkpoint model_config + forward_policy ({fp_note})."
            )
        else:
            print("   ↳ biochem GNODE from checkpoint model_config (saved by train_biochem_corrector).")
    else:
        print(
            f"   ↳ biochem GNODE from weight inference (legacy ckpt): "
            f"siren={int(ctor['use_siren_decoder'])} fourier={int(ctor['num_fourier_freqs'])} "
            f"hard_bcs={int(ctor['use_hard_bcs'])} — re-run teacher save to embed model_config."
        )
    model_biochem = GNODE_Phase3(
        phys_cfg=phys_cfg_biochem,
        in_channels=int(ctor["in_channels"]),
        spatial_channels=int(ctor["spatial_channels"]),
        latent_dim=int(ctor["latent_dim"]),
        max_inner_iters=int(ctor["max_inner_iters"]),
        bio_encoder_prior_dim=int(ctor["bio_encoder_prior_dim"]),
        mu_ratio_max=bio_cfg.mu_ratio_max,
        mat_crit=bio_cfg.viscosity_mat_crit,
        fi_crit=bio_cfg.viscosity_fi_crit,
        temp_mat=bio_cfg.viscosity_gnode_temp_mat,
        temp_fi=bio_cfg.viscosity_gnode_temp_fi,
        num_fourier_freqs=int(ctor["num_fourier_freqs"]),
        use_siren_decoder=bool(ctor["use_siren_decoder"]),
        gnode_layers=int(ctor["gnode_layers"]),
        use_hard_bcs=bool(ctor["use_hard_bcs"]),
    ).to(device)
    _inject_biochem_kinematic_lora(model_biochem)
    compatible_bio, skipped_bio = _filter_compatible_state_dict(biochem_state, model_biochem.state_dict())
    model_biochem.load_state_dict(compatible_bio, strict=False)
    if skipped_bio:
        print(f"   ↳ Skipped {len(skipped_bio)} checkpoint key(s) (no target or shape mismatch).")
        siren_skipped = [k for k in skipped_bio if k.startswith("siren_decoder.")]
        if siren_skipped and ctor["use_siren_decoder"]:
            print(
                f"   ⚠️ siren_decoder not loaded ({len(siren_skipped)} keys) — biochem |u| will look wrong. "
                "Set BIOCHEM_USE_SIREN=1 and re-run, or use a checkpoint trained with SIREN."
            )
    model_biochem.eval()

    # ------------------------------------------------------------------
    # 5. Inference
    # ------------------------------------------------------------------
    print("\n🔮 Running inference...", flush=True)

    def _cuda_sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    comsol_series_np: Optional[np.ndarray] = None
    comsol_final_np: Optional[np.ndarray] = None

    pred_kine_on_biochem_mesh = None
    with torch.no_grad():
        if model_kine_base is not None:
            t_k0 = time.perf_counter()
            kine_data = data_kine_base if data_kine_base is not None else data_biochem
            print("   ↳ Kinematics (GINO-DEQ) on visualization mesh…", flush=True)
            pred_kine_on_biochem_mesh = _run_model_once(model_kine_base, kine_data)
            _cuda_sync()
            print(f"   ↳ Kinematics done in {time.perf_counter() - t_k0:.1f}s", flush=True)
        pred_kine_base = pred_kine_on_biochem_mesh

        # ``GNODE_Phase3`` expects non-dimensional times (same as training: ``to_t_nd(..., bio_cfg.t_final)``).
        dense_times_si_full = bio_cfg.resolve_biochem_times(data_biochem, device)
        if dense_times_si_full.numel() < 2:
            raise ValueError("Biochem timeline must contain at least two timestamps for rollout visualization.")
        t_ref = float(bio_cfg.t_final)
        t_final_si = float(dense_times_si_full[-1].item())
        n_full = int(dense_times_si_full.numel())
        default_mode = "dense"
        time_mode = (os.environ.get("VIZ_BIOCHEM_TIME_MODE", default_mode) or default_mode).strip().lower()
        extend_mult_default = 1.2 if fast_mode else 1.5
        try:
            extend_mult = float(os.environ.get("VIZ_BIOCHEM_EXTEND_MULT", str(extend_mult_default)))
        except ValueError:
            extend_mult = extend_mult_default
        extend_mult = max(1.0, extend_mult)
        custom_times = _slider_keyframe_times_si(dense_times_si_full, t_final_si, extend_mult)

        if time_mode in ("keyframe", "keyframes", "sparse"):
            rollout_times_si = torch.tensor(custom_times, device=device, dtype=torch.float32)
            rollout_times = to_t_nd(rollout_times_si, t_ref)
            print(
                f"   ↳ Biochem rollout: {rollout_times.numel()} keyframes "
                f"(mode={time_mode}; set VIZ_BIOCHEM_TIME_MODE=dense for higher fidelity)",
                flush=True,
            )
        else:
            # Each macro step runs a full DEQ-style kinematics solve + an ODE segment — interactive viz
            # must subsample the COMSOL-sized grid (often ~60+ steps) or a single run can take many minutes.
            n_cap_default = "12" if fast_mode else "16"
            n_cap_raw = (os.environ.get("VIZ_BIOCHEM_MACRO_STEPS") or n_cap_default).strip()
            if n_cap_raw == "0" or (n_cap_raw.lower() in ("full", "all")):
                n_macro_use = n_full
            else:
                try:
                    n_cap = max(4, min(512, int(n_cap_raw)))
                except ValueError:
                    n_cap = int(n_cap_default)
                n_macro_use = min(n_full, n_cap)
            if n_macro_use >= n_full:
                dense_times_si = dense_times_si_full
            else:
                idx = torch.linspace(0, n_full - 1, steps=n_macro_use, device=device).round().long()
                dense_times_si = dense_times_si_full[idx]
            dense_times = to_t_nd(dense_times_si, t_ref)
            dt_si = float((dense_times_si[1] - dense_times_si[0]).item())
            if dt_si <= 0.0:
                raise ValueError(f"Invalid biochem timeline step dt_si={dt_si}. Expected strictly positive spacing.")
            dt_nd = float((dense_times[1] - dense_times[0]).item())
            if dt_nd <= 0.0:
                raise ValueError(f"Invalid ND timeline step dt_nd={dt_nd}. Expected strictly positive spacing.")
            extra_frac_default = 0.2 if fast_mode else 0.5
            try:
                extra_frac = float(os.environ.get("VIZ_BIOCHEM_EXTRA_FRACTION", str(extra_frac_default)))
            except ValueError:
                extra_frac = extra_frac_default
            extra_frac = max(0.0, extra_frac)
            num_extra = max(1, int((t_final_si * extra_frac) / max(dt_si, 1e-9)))
            rollout_times = torch.cat([
                dense_times,
                dense_times[-1] + dt_nd * torch.arange(1, num_extra + 1, device=device, dtype=dense_times.dtype),
            ])
            print(
                f"   ↳ Biochem rollout: {rollout_times.numel()} macro knots "
                f"(mode={time_mode}; using {n_macro_use}/{n_full} base samples)",
                flush=True,
            )

        y_oracle = None
        if (os.environ.get("BIOCHEM_K10G_ORACLE_CLOTS") or "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            if hasattr(data_biochem, "y") and data_biochem.y is not None and data_biochem.y.shape[0] >= 2:
                n_oracle = min(int(rollout_times.numel()), int(data_biochem.y.shape[0]))
                y_oracle = data_biochem.y[:n_oracle].to(device=device)
                rollout_times = rollout_times[:n_oracle]
                print(
                    f"   ↳ K10g oracle clots: μ_eff in wall-adjacent band from GT (n_steps={n_oracle})",
                    flush=True,
                )
            else:
                print("   ⚠️ BIOCHEM_K10G_ORACLE_CLOTS=1 but data.y missing; open-loop rollout.", flush=True)

        t_b0 = time.perf_counter()
        with _viz_biochem_ode_speedups():
            pred_biochem_series_dense = model_biochem(
                data_biochem,
                rollout_times,
                y_true_trajectory=y_oracle,
            )
        _cuda_sync()
        print(f"   ↳ Biochem trajectory done in {time.perf_counter() - t_b0:.1f}s", flush=True)

        # Extract the keyframes used by the temporal slider (labels stay in SI seconds for the UI).
        custom_times_nd = [t / t_ref for t in custom_times]
        frame_indices = [torch.argmin(torch.abs(rollout_times - t_nd)).item() for t_nd in custom_times_nd]
        pred_biochem_series = pred_biochem_series_dense[frame_indices]

        # Extract the final *trained* time step for static Fig 1 comparison.
        idx_t_final = torch.argmin(torch.abs(rollout_times - (t_final_si / t_ref))).item()
        pred_biochem = pred_biochem_series_dense[idx_t_final]

        if use_anchor:
            comsol_times_si = bio_cfg.resolve_biochem_times(data_biochem, device)
            comsol_frame_indices = _nearest_time_indices(comsol_times_si, custom_times)
            comsol_series_np = _extract_state_series_np(data_biochem, comsol_frame_indices)
            idx_comsol_final = _nearest_time_indices(comsol_times_si, [t_final_si])[0]
            if data_biochem.y.dim() == 3:
                comsol_final_np = data_biochem.y[idx_comsol_final].detach().cpu().numpy()
            else:
                comsol_final_np = data_biochem.y.detach().cpu().numpy()
            print(
                f"   ↳ COMSOL reference: {int(comsol_times_si.numel())} export steps; "
                f"slider aligned to model keyframes",
                flush=True,
            )

    pred_biochem_np = pred_biochem.detach().cpu().numpy()
    pred_biochem_series_np = pred_biochem_series.detach().cpu().numpy()
    pred_kine_base_np = pred_kine_base.detach().cpu().numpy() if pred_kine_base is not None else None

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

    vel_biochem, p_biochem, mu_biochem = get_kinematics(pred_biochem_np)

    if use_anchor and comsol_final_np is not None:
        vel_ref, p_ref, mu_ref = get_kinematics(comsol_final_np)
        vel_comsol, p_comsol, mu_comsol = vel_ref, p_ref, mu_ref
    else:
        vel_comsol = p_comsol = mu_comsol = None

    if pred_kine_base_np is not None:
        vel_kine_base, p_kine_base, mu_kine_base = get_kinematics(pred_kine_base_np)
    elif use_anchor and comsol_final_np is not None:
        vel_kine_base, p_kine_base, mu_kine_base = vel_comsol, p_comsol, mu_comsol
    else:
        raise RuntimeError("No reference kinematics available for visualization.")

    # ------------------------------------------------------------------
    # 6. Plotting
    # ------------------------------------------------------------------
    t_final_si = float(bio_cfg.resolve_biochem_times(data_biochem, device)[-1].item())
    time_label = f"t_final≈{t_final_si:.0f}s (last COMSOL export step)"
    print("🎨 Generating comparison plots...")
    print(
        "   Figure guide (matplotlib order):\n"
        "     • Steady GINO-DEQ — time-independent Stage-A snapshot (not biochem rollout t_final)\n"
        f"     • Dynamic μ_eff — biochem rollout at {time_label}\n"
        f"     • Gelation triggers — same instant as μ_eff panel; COMSOL row = GT species, Biochem row = terms in forward\n"
        "     • Temporal inspector — Species + Flow/rheology with Time slider (keyframes, not only t_final)"
    )

    if pred_kine_base_np is not None and model_kine_base is not None:
        _show_kinematics_static_figure(
            pos, vel_kine_base, p_kine_base, mu_kine_base, case_label=case_label
        )

    # --- FIGURE 2: rollout μ_eff at final time (stored channel; matches forward coupling) ---
    mu_labels = _biochem_mu_viz_labels()
    mu_dyn_np = _rollout_mu_eff_si_numpy(model_biochem.phys_cfg, pred_biochem_np)

    comsol_mu_dyn_np = None
    if use_anchor and comsol_final_np is not None:
        comsol_mu_dyn_np = _rollout_mu_eff_si_numpy(model_biochem.phys_cfg, comsol_final_np)

    mu_clim_arrays: List[np.ndarray] = [mu_dyn_np]
    if comsol_mu_dyn_np is not None:
        mu_clim_arrays.append(comsol_mu_dyn_np)
    mu_si_vmin, mu_si_vmax = _viz_mu_si_clim(*mu_clim_arrays)

    n_rheo_cols = 2 if comsol_mu_dyn_np is not None else 1
    fig2, axs2 = plt.subplots(1, n_rheo_cols, figsize=(7 * n_rheo_cols, 5.5))
    axs2 = np.atleast_1d(axs2)
    rheo_title = f"Dynamic viscosity at {time_label} ({case_label}; {mu_labels['mu_dynamic']})"
    col_i = 0
    if comsol_mu_dyn_np is not None:
        _plot_field(
            fig2,
            axs2[col_i],
            pos,
            comsol_mu_dyn_np,
            f"COMSOL: {mu_labels['mu_dynamic']}",
            _viz_mu_cmap(),
            vmin=mu_si_vmin,
            vmax=mu_si_vmax,
        )
        col_i += 1
    _plot_field(
        fig2,
        axs2[col_i],
        pos,
        mu_dyn_np,
        f"Biochem: {mu_labels['mu_dynamic']}",
        _viz_mu_cmap(),
        vmin=mu_si_vmin,
        vmax=mu_si_vmax,
    )
    fig2.tight_layout(rect=(0, 0.03, 1, 0.90))
    _set_figure_suptitle(fig2, rheo_title, fontsize=14)

    mb_mu1_ml, mu2_ml = _trigger_fields_numpy(
        model_biochem, data_biochem, pred_biochem_np, biochem_rollout=True
    )
    mb_mu1_comsol = mu2_comsol = None
    if use_anchor and comsol_final_np is not None:
        mb_mu1_comsol, mu2_comsol = _trigger_fields_numpy(
            model_biochem, data_biochem, comsol_final_np, biochem_rollout=False
        )
    _show_mu_trigger_comparison_figure(
        pos,
        model_biochem,
        mb_mu1_ml,
        mu2_ml,
        case_label=case_label,
        time_label=time_label,
        mb_mu1_comsol=mb_mu1_comsol,
        mu2_comsol=mu2_comsol,
        mu_labels=mu_labels,
    )

    def _request_refresh():
        refresh_state["requested"] = True
        plt.close("all")

    slider_refresh_label = "Next Anchor" if use_anchor else "Refresh geometry"
    print("⏳ Opening interactive Biochem temporal slider...")
    _show_biochem_temporal_slider(
        pos,
        pred_biochem_series_np,
        custom_times,
        model_biochem,
        data_biochem,
        on_refresh=_request_refresh,
        comsol_series_np=comsol_series_np,
        title_prefix=f"Biochem Temporal Inspector ({case_label})",
        refresh_label=slider_refresh_label,
    )
    return refresh_state["requested"]


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    parser = argparse.ArgumentParser(
        description=(
            "Visualize biochem model vs COMSOL anchor labels (default) or a freshly generated synthetic vessel."
        )
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate/use a synthetic vessel under data/phase_comparison_test instead of a COMSOL anchor graph",
    )
    parser.add_argument(
        "--anchor",
        type=str,
        default=None,
        metavar="STEM",
        help=(
            "COMSOL anchor stem to visualize (default: patient007 if present, else first held-out anchor). "
            "Override with VIZ_ANCHOR_STEM."
        ),
    )
    parser.add_argument(
        "--list-anchors",
        action="store_true",
        help="List available anchor graph stems and exit",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="With --synthetic: regenerate temporary single-case synthetic data before plotting",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="With --synthetic: reuse previously generated temporary data (if available)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the synthetic case when regeneration is enabled",
    )
    parser.add_argument(
        "--biochem-checkpoint",
        type=str,
        default=None,
        help=(
            "Biochem weights file. Default: biochem_teacher_best_high_mu.pth → "
            "biochem_teacher_last.pth (then legacy fallbacks). Override via path or VIZ_BIOCHEM_CHECKPOINT."
        ),
    )
    parser.add_argument(
        "--teacher-only",
        action="store_true",
        help=(
            "Teacher checkpoints only: biochem_teacher_best_high_mu.pth → biochem_teacher_last.pth. "
            "Same as VIZ_BIOCHEM_REQUIRE_TEACHER=1."
        ),
    )
    parser.add_argument(
        "--full-viz",
        action="store_true",
        help="Disable fast viz (dense rollout up to 16+ macro steps, finer ODE; sets VIZ_FAST=0).",
    )
    args = parser.parse_args()

    if args.full_viz:
        os.environ["VIZ_FAST"] = "0"

    if args.list_anchors:
        stems = _list_anchor_stems()
        anchor_dir = _anchor_graph_dir()
        if not stems:
            print(f"No anchor graphs found under {anchor_dir}")
        else:
            default_stem = _default_val_anchor_stem(stems)
            print(f"Anchor graphs in {anchor_dir}:")
            for stem in stems:
                mark = " (default)" if stem == default_stem else ""
                print(f"  - {stem}{mark}")
        raise SystemExit(0)

    if args.regenerate and args.reuse:
        raise ValueError("Use only one of --regenerate or --reuse")

    source = "synthetic" if args.synthetic else "anchor"
    regenerate = args.regenerate if args.regenerate or args.reuse else True
    if args.reuse:
        regenerate = False

    seed = args.seed
    anchor_stems = _list_anchor_stems()
    anchor_idx = 0
    if source == "anchor" and args.anchor:
        if args.anchor.endswith(".pt"):
            args_anchor = Path(args.anchor).stem
        else:
            args_anchor = args.anchor
        if args_anchor in anchor_stems:
            anchor_idx = anchor_stems.index(args_anchor)
        else:
            anchor_idx = 0
    elif source == "anchor":
        default_stem = _resolve_anchor_stem(None)
        anchor_idx = anchor_stems.index(default_stem) if default_stem in anchor_stems else 0

    while True:
        current_anchor = anchor_stems[anchor_idx] if source == "anchor" and anchor_stems else args.anchor
        refresh_requested = run_phase_comparison(
            source=source,
            regenerate=regenerate,
            seed=seed,
            biochem_checkpoint=args.biochem_checkpoint,
            anchor_stem=current_anchor,
            teacher_only=args.teacher_only,
            fast_viz=not args.full_viz,
        )
        if not refresh_requested:
            break
        if source == "anchor":
            if not anchor_stems:
                break
            anchor_idx = (anchor_idx + 1) % len(anchor_stems)
            print(f"🔁 Next anchor: {anchor_stems[anchor_idx]}")
        else:
            regenerate = True
            seed = int(np.random.default_rng().integers(0, 2**31 - 1))
            print(f"🔁 Refresh requested. Regenerating synthetic vessel with new seed: {seed}")