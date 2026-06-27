from __future__ import annotations

import contextlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import argparse
from matplotlib.widgets import Slider, Button
from src.utils.paths import get_project_root, resolve_checkpoint
from src.data_gen import MeshToGraphComplete, MeshToGraphPhase3, VesselGeneratorPhase3
from src.architecture.ginodeq import GINO_DEQ
from src.architecture.kinematics_model_config import (
    build_gino_deq_from_ctor,
    kinematics_checkpoint_tensors,
    resolve_gino_deq_ctor_kwargs,
)
from src.architecture.lora_injection import inject_lora_to_spectral_linears
from src.config import PhysicsConfig, BiochemConfig, STATE_CHANNEL_MU_EFF_ND, VesselConfig
from src.utils import species_channels
from src.core_physics.clot_phi_simple import (
    build_clot_phi_model,
    build_clot_phi_step,
    carreau_mu_si_from_uv,
    clot_phi_hybrid_enabled,
    log_blend_mu_eff_si,
    mu_eff_from_delta_log_si,
)
from src.core_physics.clot_phi_mu_inject import (
    assemble_committed_mu_map,
    biochem_mlp_mu_map_enabled,
    committed_mu_mesh_from_clot_model,
    compute_mlp_commit_gates_at_rollout_frame,
    expand_seed_growth_allowed_mask,
    init_deploy_supervision_vision_mask,
    mlp_deploy_no_commit_at_t0,
    mlp_deploy_vision_grow_enabled,
    mlp_deploy_vision_restrict_enabled,
    mlp_mu_map_bulk_mode,
    mlp_mu_map_mask_mode,
    mlp_mu_map_phi_gate_enabled,
    mu_map_carreau_baseline_si,
    resolve_clot_trigger_gate,
    resolve_mu_map_baselines_si,
)
from src.inference.deploy_mu_map_env import wire_deploy_mu_map
from src.evaluation.clot_phi_checkpoint_env import (
    apply_clot_phi_config_from_checkpoint,
    apply_clot_phi_eval_defaults,
)
from src.utils.nondim import to_t_nd
from src.utils.channel_schema import infer_missing_schema
from src.utils.kinematics_paths import kinematics_graph_rheology_dir

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
# Locked passive / step-2 teacher artifacts (STOP_AFTER_TEACHER=1 saves).
_BIOCHEM_PASSIVE_TEACHER_CKPT_NAMES: Dict[str, str] = {
    "biochem_teacher_passive_mu_unlock_best.pth": "passive_mu_unlock_best",
    "biochem_teacher_passive_xy_locked.pth": "passive_xy_locked",
    "biochem_teacher_passive_align_locked.pth": "passive_align_locked",
    "biochem_teacher_passive_species_locked.pth": "passive_species_locked",
    "biochem_teacher_passive_m3_locked.pth": "passive_m3_locked",
}
# Names written by ``train_biochem_corrector.py``.
_BIOCHEM_CKPT_ROLE_BY_NAME: Dict[str, str] = {
    "biochem_teacher_best_high_mu.pth": "teacher_best_high_mu",
    "biochem_teacher_last.pth": "teacher_last",
    "biochem_teacher_best.pth": "teacher_best_all_legacy",
    "biochem_best_high_mu.pth": "corrector_best_high_mu",
    "biochem_best_bio.pth": "corrector_best_bio_legacy",
    "biochem_latest_checkpoint.pth": "corrector_latest",
    **_BIOCHEM_PASSIVE_TEACHER_CKPT_NAMES,
}
_BIOCHEM_TEACHER_CKPT_ROLES = frozenset(
    {"teacher_best_high_mu", "teacher_last", "teacher_best", "teacher_best_all_legacy"}
)
_BIOCHEM_CORRECTOR_CKPT_ROLES = frozenset(
    {"corrector_best_high_mu", "corrector_best_bio_legacy", "corrector_latest", "corrector_best"}
)
_BIOCHEM_CKPT_INVENTORY_EXTRA = ("biochem_best_bio.pth",) + tuple(_BIOCHEM_PASSIVE_TEACHER_CKPT_NAMES)
_DEFAULT_VAL_ANCHOR_STEM = "patient007"
_CLOT_PHI_CKPT_CANDIDATES = (
    "outputs/biochem/clot_baseline/clot_phi_best.pth",
    "outputs/biochem/passive_species_focus_compare/gnode12_lane_a_clotphi/clot_phi_best.pth",
    "outputs/biochem/clot_phi_best.pth",
)
# Legacy COMSOL display (species sigmoids); biochem rollout uses stored ``mu_eff`` when ablated.
_MU_DYNAMIC_SI_LABEL = r"$\mu_b \times (\mu_1(\mathrm{Mat}) + \mu_2(\mathrm{FI}))$ [Pa·s]"
_MU1_PRODUCT_SI_LABEL = r"$\mu_{blood}\times\mu_1$(Mat) [Pa·s]"
_MU2_TRIGGER_LABEL = r"$\mu_2$ trigger (FI) [-]"
# COMSOL Surface default "WaveLightClassic" ~ matplotlib blue–white–red (``bwr``).
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
    """Panel titles for biochem mu fields (GraphSAGE deploy)."""
    return {
        "mu_dynamic": r"$\mu_{\mathrm{eff}}$ (deploy) [Pa·s]",
        "mu1_product": r"$\mu_{\mathrm{blood}}\times\mu_1$ (effective) [Pa·s]",
        "mu2_trigger": r"$\mu_2$ (effective in forward) [-]",
        "gelation_suffix": "GraphSAGE deploy mu",
    }


def _rollout_mu_eff_si_numpy(phys_cfg: PhysicsConfig, pred_np: np.ndarray) -> np.ndarray:
    """Stored rollout viscosity channel (ND -> SI); matches kinematic coupling in forward."""
    mu_ch = STATE_CHANNEL_MU_EFF_ND
    mu_nd = torch.from_numpy(pred_np[:, mu_ch]).float().view(-1, 1)
    return phys_cfg.viscosity_nd_to_si(mu_nd).detach().cpu().numpy().reshape(-1)


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


def _is_teacher_checkpoint_role(role: str) -> bool:
    """True for teacher-only artifacts (including passive locked ckpts), not corrector snapshots."""
    if role in _BIOCHEM_CORRECTOR_CKPT_ROLES:
        return False
    if role in _BIOCHEM_TEACHER_CKPT_ROLES:
        return True
    if role.startswith("teacher_"):
        return True
    if role.startswith("passive_"):
        return True
    return False


def _resolve_checkpoint_role(path: Path) -> str:
    role = _BIOCHEM_CKPT_ROLE_BY_NAME.get(path.name, "custom")
    if role != "custom":
        return role
    try:
        meta, _ = _checkpoint_state_dict(_load_torch_checkpoint(path))
        return _infer_role_from_meta(meta, path.name)
    except (OSError, TypeError, ValueError):
        return "custom"


def _print_biochem_checkpoint_banner(choice: BiochemCheckpointChoice, meta: Dict[str, Any]) -> None:
    """Explain which ``train_biochem_corrector`` artifact is driving visualization."""
    role = _infer_role_from_meta(meta, choice.path.name)
    role_labels = {
        "teacher_best_high_mu": "global-best teacher (lowest val mu_log_mae_high_mu)",
        "teacher_last": "most recent teacher run (backup)",
        "teacher_best_all_legacy": "legacy global teacher all-truth (biochem_teacher_best.pth)",
        "corrector_best_high_mu": "corrector global high-mu (legacy filename)",
        "corrector_best_bio_legacy": "legacy corrector best (composite; deprecated)",
        "corrector_latest": "full corrector — latest resume snapshot",
        "custom": "user-specified path",
    }
    print("\n[i]  Biochem checkpoint selection (from train_biochem_corrector.py):")
    print(f"   ->  file: {choice.path.resolve()}")
    print(f"   ->  role: {role} — {role_labels.get(role, role_labels['custom'])}")
    if choice.explicit:
        print("   ->  source: --biochem-checkpoint or VIZ_BIOCHEM_CHECKPOINT")
    elif role in _BIOCHEM_TEACHER_CKPT_ROLES:
        print(
            "   ->  source: teacher checkpoint preference "
            f"({' -> '.join(_BIOCHEM_TEACHER_CKPT_CANDIDATES)})"
        )
    else:
        print(
            "   ->  source: default preference order "
            f"({' -> '.join(_BIOCHEM_CKPT_CANDIDATES)}) — first existing file wins"
        )
    t_ep = meta.get("best_epoch", -1)
    t_mae = meta.get("val_mu_log_mae")
    t_high = meta.get("val_mu_log_mae_high_mu")
    run_note = (meta.get("run_note") or "").strip()
    if isinstance(t_ep, int) and t_ep >= 0:
        print(f"   ->  saved at teacher/corrector epoch: {int(t_ep)}")
    if t_mae is not None:
        try:
            print(f"   ->  val mu_log_mae all (stored in ckpt): {float(t_mae):.4f}")
        except (TypeError, ValueError):
            pass
    if t_high is not None:
        try:
            print(f"   ->  val mu_log_mae high-mu (stored in ckpt): {float(t_high):.4f}")
        except (TypeError, ValueError):
            pass
    if run_note:
        print(f"   ->  run_note: {run_note}")
    print("   ->  on-disk inventory:")
    for line in _format_biochem_ckpt_inventory().splitlines():
        print(line)


def _anchor_graph_dir() -> Path:
    return Path(VesselConfig(phase="biochem_anchors").graph_output_dir)


def _list_anchor_stems() -> List[str]:
    anchor_dir = _anchor_graph_dir()
    if not anchor_dir.is_dir():
        return []
    return sorted(p.stem for p in anchor_dir.glob("*.pt"))


def _default_val_anchor_stem(stems: List[str]) -> str:
    env = (os.environ.get("VIZ_ANCHOR_STEM") or "").strip()
    if env and env in stems:
        return env
    if _DEFAULT_VAL_ANCHOR_STEM in stems:
        return _DEFAULT_VAL_ANCHOR_STEM
    return stems[0] if stems else ""


def _parser_default_anchor_stem() -> str:
    """CLI default for ``--anchor`` (patient007 when present, else first biochem anchor)."""
    stems = _list_anchor_stems()
    if stems:
        return _default_val_anchor_stem(stems)
    return _DEFAULT_VAL_ANCHOR_STEM


def _resolve_anchor_stem(explicit: Optional[str]) -> str:
    stems = _list_anchor_stems()
    if not stems:
        raise FileNotFoundError(f"No anchor graphs under {_anchor_graph_dir()}")
    if explicit:
        stem = Path(explicit).stem if str(explicit).endswith(".pt") else str(explicit).strip()
        if stem in stems:
            return stem
        raise FileNotFoundError(f"Anchor stem '{stem}' not found under {_anchor_graph_dir()}")
    return _default_val_anchor_stem(stems)


def _load_graph_pt(path: Path, device: torch.device, *, phase_hint: str):
    if not path.is_file():
        raise FileNotFoundError(f"Graph not found: {path}")
    data = torch.load(path, map_location=device, weights_only=False)
    data = infer_missing_schema(data, phase_hint=phase_hint)
    if _should_refresh_kinematics_node_x(data, path=path, phase_hint=phase_hint):
        from src.data_gen.lib.node_feature_assembly import (
            kinematics_uv_prior_max,
            refresh_kinematics_node_x_on_graph,
        )

        stem = path.stem
        refreshed = refresh_kinematics_node_x_on_graph(
            data,
            stem=stem,
            y_time_index=int(os.environ.get("KINEMATICS_PRIOR_Y_TIME_INDEX", "0")),
        )
        if refreshed:
            prior_max = kinematics_uv_prior_max(data.x)
            print(
                f"   ->  Refreshed kine priors on {stem} "
                f"(uv_prior max={prior_max:.4f}; centerline Poiseuille + inlet BC)"
            )
    return data


def _should_refresh_kinematics_node_x(data, *, path: Path, phase_hint: str) -> bool:
    if os.environ.get("KINEMATICS_SKIP_PRIOR_REFRESH", "0").strip() in ("1", "true", "yes"):
        return False
    if os.environ.get("KINEMATICS_FORCE_PRIOR_REFRESH", "0").strip() in ("1", "true", "yes"):
        return True
    hint = (phase_hint or "").lower()
    is_anchor = "biochem" in hint or "biochem_anchors" in path.as_posix()
    if not is_anchor:
        return False
    if not hasattr(data, "x") or data.x is None:
        return False
    if int(data.x.shape[1]) < 18:
        return False
    from src.data_gen.lib.node_feature_assembly import kinematics_uv_prior_max

    return kinematics_uv_prior_max(data.x) <= 1e-3


def _load_anchor_graph(stem: str, device: torch.device):
    path = _anchor_graph_dir() / f"{stem}.pt"
    return _load_graph_pt(path, device, phase_hint="biochem"), path


def _graph_has_comsol_trajectory(data) -> bool:
    if not hasattr(data, "y") or data.y is None:
        return False
    y = data.y
    if y.dim() == 2 and y.shape[1] >= 3:
        return float(y[:, :3].norm().item()) > 1e-6
    if y.dim() == 3 and y.shape[2] >= 3:
        return float(y[:, :3].norm().item()) > 1e-6
    return False


def _kinematics_graph_dir(rheology: str = "newtonian") -> Path:
    return kinematics_graph_rheology_dir(rheology)


def _kinematics_anchor_graph_path(stem: str, rheology: str = "carreau") -> Path:
    from src.utils.kinematics_paths import resolve_kinematics_anchor_graph

    resolved = resolve_kinematics_anchor_graph(stem, rheology=rheology)
    if resolved is not None:
        return resolved
    from src.utils.kinematics_paths import kinematics_anchor_graph_dir

    return kinematics_anchor_graph_dir(rheology=rheology) / f"{stem}.pt"


def _load_kinematics_gino_deq(device: torch.device) -> GINO_DEQ:
    kin_ckpt = _resolve_kinematics_checkpoint()
    print(f"   ->  Kinematics checkpoint: {kin_ckpt}")
    kin_raw = torch.load(kin_ckpt, map_location=device, weights_only=False)
    kin_meta, kin_state = kinematics_checkpoint_tensors(kin_raw)
    ctor = resolve_gino_deq_ctor_kwargs(kin_meta, kin_state)
    kin_default_iters = int(ctor.get("max_iters", 25))
    kin_max_iters = int(os.environ.get("VIZ_KIN_MAX_ITERS", kin_default_iters))
    kin_max_iters = max(5, min(80, kin_max_iters))
    ctor["max_iters"] = kin_max_iters
    phys_cfg_kine = PhysicsConfig(phase="kinematics")
    model = build_gino_deq_from_ctor(phys_cfg_kine, ctor).to(device)
    print(f"   ->  GINO-DEQ max_iters={kin_max_iters} solver={os.environ.get('VIZ_KIN_SOLVER', 'anderson')}")
    model.load_state_dict(kin_state, strict=False)
    model.eval()
    return model


def _steady_kine_target_tensor(data, time_index: int = -1) -> Optional[torch.Tensor]:
    if not hasattr(data, "y") or data.y is None:
        return None
    y = data.y
    if y.dim() == 3:
        t = int(time_index) if time_index >= 0 else int(y.shape[0]) + int(time_index)
        t = max(0, min(t, y.shape[0] - 1))
        return y[t, :, :5]
    if y.dim() == 2 and y.shape[1] >= 5:
        return y[:, :5]
    return None


def _fields_from_kine_state(state_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = state_np[:, _CHANNEL["u"]]
    v = state_np[:, _CHANNEL["v"]]
    vel = np.sqrt(u ** 2 + v ** 2)
    pressure = state_np[:, _CHANNEL["p"]]
    viscosity = state_np[:, _CHANNEL["mu_eff"]]
    return vel, pressure, viscosity


def _rel_l2_uvp(pred: np.ndarray, tgt: np.ndarray) -> float:
    diff = pred[:, :3] - tgt[:, :3]
    return float(np.linalg.norm(diff) / (np.linalg.norm(tgt[:, :3]) + 1e-8))


def _show_steady_kinematics_pred_vs_gt(
    pos: np.ndarray,
    pred_np: np.ndarray,
    gt_np: Optional[np.ndarray],
    *,
    case_label: str,
    cohort: str,
    rel_l2: Optional[float] = None,
    gt_row_label: str = "COMSOL GT (t=0)",
) -> None:
    """Steady GINO-DEQ: model prediction vs labels (when available)."""
    vel_p, p_p, mu_p = _fields_from_kine_state(pred_np)
    has_gt = gt_np is not None and gt_np.shape[0] == pred_np.shape[0]
    nrows = 2 if has_gt else 1
    xlo, xhi, ylo, yhi = _mesh_axis_limits(pos)
    xspan = max(xhi - xlo, 1e-9)
    yspan = max(yhi - ylo, 1e-9)
    panel_aspect = yspan / xspan
    panel_h = 3.8
    panel_w = min(panel_h * panel_aspect, 5.2)
    fig, axes = plt.subplots(nrows, 3, figsize=(panel_w * 3.35, panel_h * nrows + 0.7))
    axes = np.atleast_2d(axes)

    rel_note = f" | rel_L2(uvp)={rel_l2:.3f}" if rel_l2 is not None else ""
    gt_note = f" vs {gt_row_label}" if has_gt else ""
    title = f"Steady kinematics — {cohort} / {case_label}{gt_note}{rel_note}"

    row_specs = [(vel_p, p_p, mu_p, "GINO-DEQ pred")]
    vel_g = p_g = mu_g = None
    if has_gt:
        vel_g, p_g, mu_g = _fields_from_kine_state(gt_np)
        row_specs.append((vel_g, p_g, mu_g, gt_row_label))

    col_titles = ("|u| (ND)", "p (ND)", r"$\mu_{eff}$ (ND)")
    cmaps = ("jet", "coolwarm", "viridis")
    if has_gt and vel_g is not None:
        vel_lo, vel_hi = 0.0, float(np.percentile(vel_g, 99.5))
        if vel_hi <= vel_lo:
            vel_hi = float(vel_g.max()) + 1e-12
        p_lo, p_hi = float(np.percentile(p_g, 1.0)), float(np.percentile(p_g, 99.0))
        if p_hi <= p_lo:
            p_lo, p_hi = float(p_g.min()), float(p_g.max()) + 1e-12
        mu_lo, mu_hi = float(np.percentile(mu_g, 1.0)), float(np.percentile(mu_g, 99.0))
        if mu_hi <= mu_lo:
            mu_lo, mu_hi = float(mu_g.min()), float(mu_g.max()) + 1e-12
        col_clims = [(vel_lo, vel_hi), (p_lo, p_hi), (mu_lo, mu_hi)]
    else:
        col_clims = [(None, None), (None, None), (None, None)]

    for row_i, (vel, pres, mu, row_lbl) in enumerate(row_specs):
        for col_i, (values, col_lbl, cmap) in enumerate(zip((vel, pres, mu), col_titles, cmaps)):
            vmin, vmax = col_clims[col_i]
            _plot_field(
                fig,
                axes[row_i, col_i],
                pos,
                values,
                f"{row_lbl}: {col_lbl}",
                cmap,
                tight_axes=True,
                xlim=(xlo, xhi),
                ylim=(ylo, yhi),
                vmin=vmin,
                vmax=vmax,
            )
    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.06, top=0.88, wspace=0.14, hspace=0.28)
    _set_figure_suptitle(fig, title, fontsize=14, subplot_top=0.84, title_y=0.96)


def run_steady_kinematics_viz(
    *,
    cases: List[Tuple[str, str, Path]],
    time_index: int = -1,
    device: Optional[torch.device] = None,
) -> None:
    """Run steady GINO-DEQ on one or more graphs; matplotlib pred vs GT per case."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Steady kinematics-only viz on {device}")
    model = _load_kinematics_gino_deq(device)
    for cohort, label, graph_path in cases:
        phase_hint = "biochem" if cohort == "patient" else "kinematics"
        data = _load_graph_pt(graph_path, device, phase_hint=phase_hint)
        with torch.no_grad():
            pred = _run_model_once(model, data)
        pred_np = pred.detach().cpu().numpy()
        tgt = _steady_kine_target_tensor(data, time_index=time_index)
        gt_np = tgt.detach().cpu().numpy() if tgt is not None else None
        rel = _rel_l2_uvp(pred_np, gt_np) if gt_np is not None else None
        if rel is not None:
            print(f"   {label}: rel_L2(uvp)={rel:.4f} (time_index={time_index})")
        pos = data.x[:, :2].detach().cpu().numpy()
        _show_steady_kinematics_pred_vs_gt(
            pos, pred_np, gt_np, case_label=label, cohort=cohort, rel_l2=rel
        )
    print("[i] Close each figure window to exit (or advance if your backend is interactive).")
    plt.show()


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


_VIZ_SIM_END_DEFAULT_S = 30000.0


def _resolve_viz_sim_end_si(
    explicit: Optional[float],
    *,
    t_final_si: float,
    prompt: bool = True,
) -> float:
    """Simulation horizon [s]; default 30000, at least COMSOL t_final."""
    floor = max(float(t_final_si), 1.0)
    if explicit is not None:
        return max(float(explicit), floor)
    raw = (os.environ.get("VIZ_SIM_END_S") or "").strip()
    if raw:
        try:
            return max(float(raw), floor)
        except ValueError:
            pass
    # Legacy multiplier env (deprecated): VIZ_BIOCHEM_EXTEND_MULT * t_final
    mult_raw = (os.environ.get("VIZ_BIOCHEM_EXTEND_MULT") or "").strip()
    if mult_raw and t_final_si > 0:
        try:
            return max(float(mult_raw) * t_final_si, floor)
        except ValueError:
            pass
    if prompt and sys.stdin.isatty():
        try:
            line = input(
                f"Simulation end time [s] (COMSOL export ends ~{t_final_si:.0f}; default {_VIZ_SIM_END_DEFAULT_S:.0f}): "
            ).strip()
            if line:
                return max(float(line), floor)
        except (ValueError, EOFError):
            pass
    return max(_VIZ_SIM_END_DEFAULT_S, floor)


def _slider_keyframe_times_si(
    dense_times_si_full: torch.Tensor,
    t_final_si: float,
    t_end_si: float,
) -> List[float]:
    """Slider keyframes aligned to the anchor/export grid (avoids arbitrary 33/66% gaps)."""
    n = int(dense_times_si_full.numel())
    if n >= 4:
        fracs = (0.0, 0.33, 0.66, 1.0)
        idxs = [min(n - 1, max(0, int(round(f * (n - 1))))) for f in fracs]
        times = [float(dense_times_si_full[i].item()) for i in idxs]
    else:
        times = [0.0, t_final_si * 0.33, t_final_si * 0.66, t_final_si]
    t_end = max(float(t_end_si), float(t_final_si))
    if t_end > t_final_si + 1e-6:
        times.append(t_end)
        for frac in (0.25, 0.5, 0.75):
            t_ex = t_final_si + frac * (t_end - t_final_si)
            if t_ex > t_final_si + 1e-6 and t_ex < t_end - 1e-6:
                times.append(float(t_ex))
    return sorted({float(t) for t in times})


@dataclass
class _BiochemVizRolloutPlan:
    """Rollout schedule + rerun hook for the interactive temporal inspector."""

    data_biochem: Any
    model_biochem: GNODE_Phase3
    device: torch.device
    bio_cfg: BiochemConfig
    dense_times_si_full: torch.Tensor
    t_ref: float
    t_final_si: float
    fast_mode: bool
    y_oracle: Optional[torch.Tensor]
    t_end_si: float
    use_anchor: bool
    time_mode: str = "dense"

    def build_custom_times_si(self) -> List[float]:
        return _slider_keyframe_times_si(self.dense_times_si_full, self.t_final_si, self.t_end_si)

    def build_rollout_times_nd(self) -> torch.Tensor:
        device = self.device
        t_ref = self.t_ref
        t_final_si = self.t_final_si
        n_full = int(self.dense_times_si_full.numel())
        t_end_si = max(float(self.t_end_si), float(t_final_si))

        if self.time_mode in ("keyframe", "keyframes", "sparse"):
            custom_times_si = self.build_custom_times_si()
            rollout_times_si = torch.tensor(custom_times_si, device=device, dtype=torch.float32)
            return to_t_nd(rollout_times_si, t_ref)

        n_cap_default = "12" if self.fast_mode else "16"
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
            dense_times_si = self.dense_times_si_full
        else:
            idx = torch.linspace(0, n_full - 1, steps=n_macro_use, device=device).round().long()
            dense_times_si = self.dense_times_si_full[idx]
        dense_times = to_t_nd(dense_times_si, t_ref)
        dt_si = float((dense_times_si[1] - dense_times_si[0]).item())
        if dt_si <= 0.0:
            raise ValueError(f"Invalid biochem timeline step dt_si={dt_si}. Expected strictly positive spacing.")
        dt_nd = float((dense_times[1] - dense_times[0]).item())
        if dt_nd <= 0.0:
            raise ValueError(f"Invalid ND timeline step dt_nd={dt_nd}. Expected strictly positive spacing.")
        last_si = float(dense_times_si[-1].item())
        if t_end_si > last_si + 1e-6:
            num_extra = max(1, int((t_end_si - last_si) / max(dt_si, 1e-9)))
        else:
            num_extra = 0
        if num_extra > 0:
            return torch.cat(
                [
                    dense_times,
                    dense_times[-1] + dt_nd * torch.arange(1, num_extra + 1, device=device, dtype=dense_times.dtype),
                ]
            )
        return dense_times

    def run_keyframed(self) -> Tuple[np.ndarray, List[float], Optional[np.ndarray], torch.Tensor]:
        device = self.device
        rollout_times = self.build_rollout_times_nd()
        if self.y_oracle is not None:
            n_oracle = min(int(rollout_times.numel()), int(self.y_oracle.shape[0]))
            y_oracle = self.y_oracle[:n_oracle].to(device=device)
            rollout_times = rollout_times[:n_oracle]
        else:
            y_oracle = None
        t0 = time.perf_counter()
        with torch.no_grad():
            with _viz_biochem_ode_speedups():
                pred_dense = self.model_biochem(
                    self.data_biochem,
                    rollout_times,
                    y_true_trajectory=y_oracle,
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
        print(
            f"   ->  Biochem rollout: {int(rollout_times.numel())} macro knots, "
            f"sim_end~{self.t_end_si:.0f}s (COMSOL t_final~{self.t_final_si:.0f}s) "
            f"in {time.perf_counter() - t0:.1f}s",
            flush=True,
        )
        custom_times = self.build_custom_times_si()
        custom_times_nd = [t / self.t_ref for t in custom_times]
        frame_indices = [torch.argmin(torch.abs(rollout_times - t_nd)).item() for t_nd in custom_times_nd]
        pred_series_np = pred_dense[frame_indices].detach().cpu().numpy()
        idx_t_final = torch.argmin(torch.abs(rollout_times - (self.t_final_si / self.t_ref))).item()
        pred_t_final = pred_dense[idx_t_final]
        comsol_series_np = None
        if self.use_anchor and hasattr(self.data_biochem, "y") and self.data_biochem.y is not None:
            comsol_times_si = self.bio_cfg.resolve_biochem_times(self.data_biochem, device)
            comsol_frame_indices = _nearest_time_indices(comsol_times_si, custom_times)
            comsol_series_np = _extract_state_series_np(self.data_biochem, comsol_frame_indices)
        return pred_series_np, custom_times, comsol_series_np, pred_t_final


def _carreau_mu_frame_numpy(
    phys_cfg: PhysicsConfig,
    data,
    pred_np: np.ndarray,
) -> np.ndarray:
    device = data.x.device
    pred_t = torch.from_numpy(pred_np).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        mu_c = carreau_mu_si_from_uv(data, pred_t[:, 0], pred_t[:, 1], phys_cfg)
    return mu_c.reshape(-1).detach().cpu().numpy()


def _committed_mu_single_frame_numpy(
    data,
    pred_np: np.ndarray,
    time_index: int,
    *,
    clot_model: torch.nn.Module | None,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
) -> np.ndarray:
    if clot_model is None:
        return _carreau_mu_frame_numpy(phys_cfg, data, pred_np)
    device = data.x.device
    pred_t = torch.from_numpy(pred_np).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        mu_si, _mu_c, _phi, _mu_mlp = committed_mu_mesh_from_clot_model(
            clot_model,
            data,
            int(time_index),
            u_nd=pred_t[:, 0],
            v_nd=pred_t[:, 1],
            species_log=pred_t[:, species_channels.SPECIES_BLOCK],
            phys_cfg=phys_cfg,
            bio_cfg=bio_cfg,
            device=device,
        )
    return mu_si.detach().cpu().numpy()


def _committed_mu_mesh_series_numpy(
    data,
    pred_series_np: np.ndarray,
    time_indices: List[int],
    *,
    clot_model: torch.nn.Module,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
) -> np.ndarray:
    """Full-mesh v2 mu from rollout [u,v] + standalone clot MLP (not GNODE channel 3)."""
    device = data.x.device
    dtype = torch.float32
    t_steps = int(pred_series_np.shape[0])
    out = np.zeros((t_steps, int(pred_series_np.shape[1])), dtype=np.float64)
    with torch.no_grad():
        for ti in range(t_steps):
            pred_t = torch.from_numpy(pred_series_np[ti]).to(device=device, dtype=dtype)
            y_idx = int(time_indices[ti]) if ti < len(time_indices) else ti
            mu_si, mu_c, phi, mu_mlp = committed_mu_mesh_from_clot_model(
                clot_model,
                data,
                y_idx,
                u_nd=pred_t[:, 0],
                v_nd=pred_t[:, 1],
                species_log=pred_t[:, species_channels.SPECIES_BLOCK],
                phys_cfg=phys_cfg,
                bio_cfg=bio_cfg,
                device=device,
            )
            out[ti] = mu_si.detach().cpu().numpy()
            if ti == 0:
                step0 = build_clot_phi_step(
                    data,
                    y_idx,
                    phys_cfg,
                    bio_cfg,
                    device,
                    u_nd_override=pred_t[:, 0],
                    v_nd_override=pred_t[:, 1],
                )
                gate0 = resolve_clot_trigger_gate(
                    phi,
                    step0.mu_c_si,
                    mu_mlp,
                    region=step0.region,
                    mu_gt_cap_si=step0.mu_gt_cap,
                    phys_cfg=phys_cfg,
                )
                clot_frac = float(gate0.reshape(-1).mean().item())
                print(
                    f"   [i]  v2 mu mesh t=0: mask={mlp_mu_map_mask_mode()} bulk={mlp_mu_map_bulk_mode()} "
                    f"mu_c mean={float(mu_c.mean()):.4f} Pa*s "
                    f"mu_out mean={float(mu_si.mean()):.4f} clot_frac={clot_frac:.4f}",
                    flush=True,
                )
    return out


def _temporal_series_arrays(
    model_biochem: GNODE_Phase3,
    data_biochem,
    pred_biochem_series_np: np.ndarray,
    comsol_series_np: Optional[np.ndarray],
    *,
    time_indices: Optional[List[int]] = None,
    clot_model: Optional[torch.nn.Module] = None,
) -> Dict[str, Any]:
    vel_all, _ = _vel_pressure_from_series(pred_biochem_series_np)
    bio_cfg = BiochemConfig(phase="biochem")
    mu_labels = _biochem_mu_viz_labels()
    scales = bio_cfg.get_species_scales(device="cpu").cpu().numpy()
    fib_all, mat_all = _species_si_from_series(pred_biochem_series_np, scales)
    mu_dyn_all = _rheology_series_numpy(
        model_biochem,
        data_biochem,
        pred_biochem_series_np,
        biochem_rollout=True,
        bio_cfg=bio_cfg,
        time_indices=time_indices,
        clot_model=clot_model,
    )
    mb_mu1_all, mu2_all = _rheology_trigger_series_numpy(
        model_biochem, data_biochem, pred_biochem_series_np, biochem_rollout=True
    )
    out: Dict[str, Any] = {
        "vel_all": vel_all,
        "fib_all": fib_all,
        "mat_all": mat_all,
        "mu_dyn_all": mu_dyn_all,
        "mb_mu1_all": mb_mu1_all,
        "mu2_all": mu2_all,
        "mu_labels": mu_labels,
        "mu2_cap": float(model_biochem.mu_ratio_max),
        "show_legacy_gel": not _biochem_mu_disable_explicit_gelation(),
    }
    show_comsol = comsol_series_np is not None
    out["show_comsol"] = show_comsol
    if show_comsol:
        comsol_vel, _ = _vel_pressure_from_series(comsol_series_np)
        comsol_fib, comsol_mat = _species_si_from_series(comsol_series_np, scales)
        comsol_mu_dyn = _rheology_series_numpy(
            model_biochem, data_biochem, comsol_series_np, biochem_rollout=False
        )
        comsol_mb_mu1, comsol_mu2 = _rheology_trigger_series_numpy(
            model_biochem, data_biochem, comsol_series_np, biochem_rollout=False
        )
        out.update(
            comsol_vel=comsol_vel,
            comsol_fib=comsol_fib,
            comsol_mat=comsol_mat,
            comsol_mu_dyn=comsol_mu_dyn,
            comsol_mb_mu1=comsol_mb_mu1,
            comsol_mu2=comsol_mu2,
        )
    vel_vmin = float(vel_all.min())
    vel_vmax = float(vel_all.max())
    if show_comsol:
        vel_vmin = min(vel_vmin, float(out["comsol_vel"].min()))
        vel_vmax = max(vel_vmax, float(out["comsol_vel"].max()))
    fib_vmin = float(min(fib_all.min(), out["comsol_fib"].min() if show_comsol else fib_all.min()))
    fib_vmax = float(max(fib_all.max(), out["comsol_fib"].max() if show_comsol else fib_all.max()))
    mat_vmin = float(min(mat_all.min(), out["comsol_mat"].min() if show_comsol else mat_all.min()))
    mat_vmax = float(max(mat_all.max(), out["comsol_mat"].max() if show_comsol else mat_all.max()))
    mu_clim_arrays: List[np.ndarray] = [mu_dyn_all.reshape(-1)]
    m1_clim_arrays: List[np.ndarray] = [mb_mu1_all.reshape(-1)]
    if show_comsol:
        mu_clim_arrays.append(out["comsol_mu_dyn"].reshape(-1))
        m1_clim_arrays.append(out["comsol_mb_mu1"].reshape(-1))
    out["vel_vmin"] = vel_vmin
    out["vel_vmax"] = vel_vmax
    out["fib_vmin"] = fib_vmin
    out["fib_vmax"] = fib_vmax
    out["mat_vmin"] = mat_vmin
    out["mat_vmax"] = mat_vmax
    out["mu_vmin"], out["mu_vmax"] = _viz_mu_si_clim(*mu_clim_arrays)
    out["m1_vmin"], out["m1_vmax"] = _viz_mu_si_clim(*m1_clim_arrays)
    return out


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
        f"   ->  LoRA injected: kin_encoder={n_enc}, kin_processor={n_proc + n_proc_extra}, "
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
        role = _resolve_checkpoint_role(resolved)
        if require_teacher and not _is_teacher_checkpoint_role(role):
            raise FileNotFoundError(
                f"--teacher-only but checkpoint '{resolved.name}' (role={role}) is not a teacher artifact. "
                f"Use one of: {', '.join(_BIOCHEM_TEACHER_CKPT_CANDIDATES + tuple(_BIOCHEM_PASSIVE_TEACHER_CKPT_NAMES))}."
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
    solver = os.environ.get("VIZ_KIN_SOLVER", "anderson").strip().lower() or "anderson"
    beta = float(os.environ.get("VIZ_KIN_ANDERSON_BETA", "0.8"))
    warmup = int(os.environ.get("VIZ_KIN_ANDERSON_WARMUP", "5"))
    pred = model(
        data,
        solver=solver,
        anderson_beta=beta,
        anderson_warmup_iters=max(0, warmup),
    )
    return pred[0] if isinstance(pred, tuple) else pred


def _carreau_mu_blood_torch(
    model: GNODE_Phase3,
    data,
    pred_t: torch.Tensor,
) -> torch.Tensor:
    """Carreau shear-thinning blood viscosity [Pa*s] from ND ``u,v`` on the biochem mesh."""
    u_nd = pred_t[:, _CHANNEL["u"]]
    v_nd = pred_t[:, _CHANNEL["v"]]
    return carreau_mu_si_from_uv(data, u_nd, v_nd, model.phys_cfg)


def _comsol_style_rheology_fields(
    model: GNODE_Phase3,
    data,
    pred_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """COMSOL reference display: Carreau ``μ_b`` x ``(1+μ₁+μ₂)`` from species sigmoids."""
    device = pred_t.device
    dtype = pred_t.dtype
    mu_blood = _carreau_mu_blood_torch(model, data, pred_t)
    sp_safe = torch.clamp(
        pred_t[:, species_channels.SPECIES_BLOCK],
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
        pred_t[:, species_channels.SPECIES_BLOCK],
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
    bio_cfg: BiochemConfig | None = None,
    time_indices: List[int] | None = None,
    clot_model: Optional[torch.nn.Module] = None,
) -> np.ndarray:
    """``pred_series_np`` ``[T,N,C]`` -> ``μ_eff`` in SI ``[T,N]`` (rollout channel or COMSOL-style)."""
    if biochem_rollout and biochem_mlp_mu_map_enabled() and bio_cfg is not None and time_indices is not None:
        if clot_model is not None:
            return _committed_mu_mesh_series_numpy(
                data,
                pred_series_np,
                time_indices,
                clot_model=clot_model,
                phys_cfg=model.phys_cfg,
                bio_cfg=bio_cfg,
            )
        print(
            "   [WARN]  BIOCHEM_MLP_MU_MAP=1 but no clot MLP for dynamic mu; using Carreau-only from [u,v].",
            flush=True,
        )
        t_steps = int(pred_series_np.shape[0])
        out = np.zeros((t_steps, int(pred_series_np.shape[1])), dtype=np.float64)
        for ti in range(t_steps):
            out[ti] = _carreau_mu_frame_numpy(model.phys_cfg, data, pred_series_np[ti])
        return out

    device = data.x.device
    dtype = torch.float32
    t_steps = int(pred_series_np.shape[0])
    out = np.zeros((t_steps, int(pred_series_np.shape[1])), dtype=np.float64)
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
    """``[T,N,C]`` -> ``(μ_bloodxμ_1, μ_2)`` effective (biochem) or COMSOL-style sigmoids."""
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
    """Single-frame ``[N,C]`` -> ``(μ_bloodxμ_1, μ_2)`` for gelation comparison figure."""
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
    """Static gelation triggers at final rollout time; optional COMSOL vs ML 2x2 grid."""
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


def _resolve_clot_phi_checkpoint(user_path: Optional[str] = None) -> Optional[Path]:
    """Deploy clot-phi readout ckpt (MLP on teacher rollout features)."""
    if (os.environ.get("VIZ_NO_CLOT_PHI") or "").strip().lower() in ("1", "true", "yes", "on"):
        return None
    root = get_project_root()
    if user_path:
        p = Path(user_path)
        if not p.is_absolute():
            p = root / p
        if p.is_file():
            return p
    env = (os.environ.get("VIZ_CLOT_PHI_CHECKPOINT") or "").strip()
    if env:
        p = Path(env)
        if not p.is_absolute():
            p = root / p
        if p.is_file():
            return p
    for rel in _CLOT_PHI_CKPT_CANDIDATES:
        p = root / rel
        if p.is_file():
            return p
    return None


def _load_clot_phi_model(ckpt_path: Path, device: torch.device) -> Tuple[torch.nn.Module, dict]:
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = raw.get("config") or {}
    apply_clot_phi_config_from_checkpoint(cfg)
    apply_clot_phi_eval_defaults()
    os.environ.setdefault("CLOT_PHI_DGAMMA_FEATURE_TIME", "current")
    in_dim = int(cfg.get("in_dim", 6))
    hidden = int(cfg.get("hidden", 32))
    model = build_clot_phi_model(in_dim=in_dim, hidden=hidden).to(device)
    model.load_state_dict(raw["model_state_dict"])
    model.eval()
    return model, cfg


@torch.no_grad()
def _clot_phi_mu_at_frame(
    data_gt,
    data_pred,
    clot_model: torch.nn.Module,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    time_index: int,
    *,
    show_gt: bool,
    prev_mu_eff_si: torch.Tensor | None = None,
    allowed_commit_mask: torch.Tensor | None = None,
    pin_display_region_to_allowed: bool = False,
    macro_step_index: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mu_gt_cap, mu_pred, region_mask) for one time slice."""
    step_pr = build_clot_phi_step(data_pred, time_index, phys_cfg, bio_cfg, device)
    u_pr = step_pr.u_flow_nd.reshape(-1)
    v_pr = step_pr.v_flow_nd.reshape(-1)
    mu_c_bulk, mu_c_mlp = resolve_mu_map_baselines_si(data_pred, u_pr, v_pr, phys_cfg)
    if clot_phi_hybrid_enabled() and hasattr(clot_model, "forward_delta_log_mu"):
        phi_pr = torch.sigmoid(clot_model.forward_logits(step_pr.features)).reshape(-1)
        mu_mlp = mu_eff_from_delta_log_si(
            mu_c_mlp, clot_model.forward_delta_log_mu(step_pr.features)
        )
    else:
        phi_pr = clot_model(step_pr.features).reshape(-1)
        mu_mlp = log_blend_mu_eff_si(mu_c_mlp, phi_pr)
    if biochem_mlp_mu_map_enabled() and mlp_mu_map_phi_gate_enabled():
        mu_pr = assemble_committed_mu_map(
            mu_c_bulk,
            mu_mlp,
            phi_pr,
            region=step_pr.region,
            mu_gt_cap_si=step_pr.mu_gt_cap,
            phys_cfg=phys_cfg,
            graph_data=data_pred,
            prev_mu_eff_si=prev_mu_eff_si,
            u_nd=u_pr,
            v_nd=v_pr,
            bio_cfg=bio_cfg,
            allowed_commit_mask=allowed_commit_mask,
        ).reshape(-1)
        if (
            macro_step_index is not None
            and mlp_deploy_no_commit_at_t0()
            and int(macro_step_index) == 0
        ):
            mu_pr = mu_c_bulk.reshape(-1).detach().cpu().numpy()
        else:
            mu_pr = mu_pr.detach().cpu().numpy()
    else:
        mu_pr = mu_mlp.reshape(-1).detach().cpu().numpy()
    if pin_display_region_to_allowed and allowed_commit_mask is not None:
        m = allowed_commit_mask.detach().cpu().numpy().astype(bool)
    else:
        m = step_pr.region.detach().cpu().numpy().astype(bool)
    mu_pr_np = np.asarray(mu_pr, dtype=np.float64)
    if show_gt:
        step_gt = build_clot_phi_step(data_gt, time_index, phys_cfg, bio_cfg, device)
        mu_gt_np = step_gt.mu_gt_cap.detach().cpu().numpy()
    else:
        mu_gt_np = np.zeros_like(mu_pr_np)
    return mu_gt_np, mu_pr_np, m


@torch.no_grad()
def _precompute_clot_phi_mu_series(
    data_gt,
    clot_model: torch.nn.Module,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    pred_series_np: np.ndarray,
    custom_times: List[float],
    *,
    show_gt: bool,
    t_ref_times_si: torch.Tensor,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Per-keyframe MLP mu maps (neighbor band)."""
    data_pred = data_gt.clone()
    mu_gt_frames: List[np.ndarray] = []
    mu_pr_frames: List[np.ndarray] = []
    region_frames: List[np.ndarray] = []
    track_vision = mlp_deploy_vision_restrict_enabled()
    allowed: torch.Tensor | None = None
    for idx, t_si in enumerate(custom_times):
        ti = int(_nearest_time_indices(t_ref_times_si, [float(t_si)])[0])
        frame = pred_series_np[idx]
        data_pred.y[ti, :, 0:3] = torch.as_tensor(
            frame[:, 0:3], device=data_pred.y.device, dtype=data_pred.y.dtype
        )
        data_pred.y[ti, :, species_channels.SPECIES_BLOCK] = torch.as_tensor(
            frame[:, species_channels.SPECIES_BLOCK], device=data_pred.y.device, dtype=data_pred.y.dtype
        )
        prev_mu = None
        if idx > 0:
            prev_frame = pred_series_np[idx - 1]
            prev_mu = phys_cfg.viscosity_nd_to_si(
                torch.from_numpy(prev_frame[:, STATE_CHANNEL_MU_EFF_ND]).to(
                    device=device, dtype=torch.float32
                )
            )
        if track_vision and allowed is None:
            allowed = init_deploy_supervision_vision_mask(
                data_gt,
                device,
                ti,
                phys_cfg=phys_cfg,
                bio_cfg=bio_cfg,
            )
        if biochem_mlp_mu_map_enabled():
            mu_pr_np = _rollout_mu_eff_si_numpy(phys_cfg, frame)
            if show_gt:
                step_gt = build_clot_phi_step(data_gt, ti, phys_cfg, bio_cfg, device)
                mu_gt_np = step_gt.mu_gt_cap.detach().cpu().numpy()
            else:
                mu_gt_np = np.zeros(int(data_gt.num_nodes), dtype=np.float64)
            if track_vision and allowed is not None:
                region_np = allowed.detach().cpu().numpy().astype(bool)
            else:
                step_pr = build_clot_phi_step(data_pred, ti, phys_cfg, bio_cfg, device)
                region_np = step_pr.region.detach().cpu().numpy().astype(bool)
        else:
            mu_gt_np, mu_pr_np, region_np = _clot_phi_mu_at_frame(
                data_gt,
                data_pred,
                clot_model,
                phys_cfg,
                bio_cfg,
                device,
                ti,
                show_gt=show_gt,
                prev_mu_eff_si=prev_mu,
                allowed_commit_mask=allowed,
                pin_display_region_to_allowed=track_vision,
                macro_step_index=idx,
            )
        mu_gt_frames.append(mu_gt_np)
        mu_pr_frames.append(mu_pr_np)
        region_frames.append(region_np)
        if track_vision and allowed is not None and mlp_deploy_vision_grow_enabled():
            allowed = expand_seed_growth_allowed_mask(
                allowed,
                torch.from_numpy(mu_pr_np).to(device=device, dtype=torch.float32),
                data_gt,
                device,
                phys_cfg=phys_cfg,
            )
    return mu_gt_frames, mu_pr_frames, region_frames


def _build_clot_phi_figure(
    pos: np.ndarray,
    mu_gt_frames: List[np.ndarray],
    mu_pr_frames: List[np.ndarray],
    region_frames: List[np.ndarray],
    custom_times: List[float],
    *,
    case_label: str = "",
    show_gt: bool = True,
    t_final_si: float = 0.0,
) -> Optional[Callable[[int], None]]:
    """Clot map figure (MLP mu only); updates via shared Time idx slider on Flow window."""
    n_frames = len(custom_times)
    if n_frames < 1:
        return None

    ncols = 2 if show_gt else 1
    fig, axs = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    if ncols == 1:
        axs = np.asarray([axs])
    mu_vmin, mu_vmax = 0.0, 0.10

    def _region_scatter(fig, ax, values, region, title):
        idx = np.where(region)[0]
        if idx.size == 0:
            ax.set_title(title, fontsize=11)
            ax.axis("off")
            return None
        sc = ax.scatter(
            pos[idx, 0],
            pos[idx, 1],
            c=values[idx],
            s=14,
            cmap="RdBu_r",
            vmin=mu_vmin,
            vmax=mu_vmax,
        )
        ax.set_title(title, fontsize=11)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        return sc

    m0 = region_frames[0]
    scatters: Dict[str, Any] = {}
    if show_gt:
        scatters["gt"] = _region_scatter(
            fig, axs[0], mu_gt_frames[0], m0, "GT mu cap 0.10 Pa*s"
        )
        pr_title = (
            "Pred mu eff (rollout ch3)"
            if biochem_mlp_mu_map_enabled()
            else (
                "Pred mu eff (phi-gated v2)"
                if biochem_mlp_mu_map_enabled() and mlp_mu_map_phi_gate_enabled()
                else "Pred mu eff (MLP)"
            )
        )
        scatters["pr"] = _region_scatter(fig, axs[1], mu_pr_frames[0], m0, pr_title)
    else:
        pr_title = (
            "Pred mu eff (rollout ch3)"
            if biochem_mlp_mu_map_enabled()
            else (
                "Pred mu eff (phi-gated v2)"
                if biochem_mlp_mu_map_enabled() and mlp_mu_map_phi_gate_enabled()
                else "Pred mu eff (MLP)"
            )
        )
        scatters["pr"] = _region_scatter(fig, axs[0], mu_pr_frames[0], m0, pr_title)

    fig.subplots_adjust(bottom=0.10, top=0.84)
    fig.text(
        0.5,
        0.02,
        "Use Time idx slider on the Flow & rheology window",
        ha="center",
        fontsize=9,
        color="dimgray",
    )

    def update_clot_frame(frame_idx: int) -> None:
        frame_idx = int(max(0, min(frame_idx, n_frames - 1)))
        m = region_frames[frame_idx]
        idx = np.where(m)[0]
        t_si = float(custom_times[frame_idx])
        extrap = t_final_si > 0 and t_si > t_final_si + 1e-3
        t_lbl = f"t={t_si:.1f}s{' (extrap)' if extrap else ''}"
        if show_gt and scatters.get("gt") is not None:
            if idx.size > 0:
                scatters["gt"].set_offsets(pos[idx])
                scatters["gt"].set_array(mu_gt_frames[frame_idx][idx])
        if scatters.get("pr") is not None:
            if idx.size > 0:
                scatters["pr"].set_offsets(pos[idx])
                scatters["pr"].set_array(mu_pr_frames[frame_idx][idx])
        _set_figure_suptitle(
            fig,
            f"Clot map (MLP mu) -- {case_label} ({t_lbl})",
            fontsize=14,
            subplot_top=0.88,
            title_y=0.96,
        )
        fig.canvas.draw_idle()

    update_clot_frame(0)
    print(
        f"   ->  Clot MLP map: {n_frames} keyframes (use Time idx on Flow window)",
        flush=True,
    )
    return update_clot_frame


def _viz_commit_mask_enabled() -> bool:
    raw = (os.environ.get("VIZ_MLP_COMMIT_MASK") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


@torch.no_grad()
def _precompute_mlp_commit_mask_series(
    data_gt,
    clot_model: torch.nn.Module,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    pred_series_np: np.ndarray,
    custom_times: List[float],
    t_ref_times_si: torch.Tensor,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[float], str]:
    """Per-keyframe oracle gt_clot vs active-env MLP commit gates (full mesh)."""
    gt_frames: List[np.ndarray] = []
    active_frames: List[np.ndarray] = []
    gt_fracs: List[float] = []
    active_fracs: List[float] = []
    active_mode = mlp_mu_map_mask_mode()
    dtype = torch.float32
    allowed: torch.Tensor | None = None
    track_vision = active_mode == "seed_growth" or active_mode == "mlp_band" or (
        active_mode == "neighbor" and mlp_deploy_vision_restrict_enabled()
    )

    for idx, t_si in enumerate(custom_times):
        ti = int(_nearest_time_indices(t_ref_times_si, [float(t_si)])[0])
        frame = pred_series_np[idx]
        pred_t = torch.from_numpy(frame).to(device=device, dtype=dtype)
        prev_mu = None
        if idx > 0:
            prev_frame = pred_series_np[idx - 1]
            prev_mu = phys_cfg.viscosity_nd_to_si(
                torch.from_numpy(prev_frame[:, STATE_CHANNEL_MU_EFF_ND]).to(device=device, dtype=dtype)
            )
        if track_vision:
            if allowed is None:
                allowed = init_deploy_supervision_vision_mask(
                    data_gt,
                    device,
                    ti,
                    phys_cfg=phys_cfg,
                    bio_cfg=bio_cfg,
                )
            gate_gt, gate_active = compute_mlp_commit_gates_at_rollout_frame(
                clot_model,
                data_gt,
                ti,
                u_nd=pred_t[:, 0],
                v_nd=pred_t[:, 1],
                species_log=pred_t[:, species_channels.SPECIES_BLOCK],
                phys_cfg=phys_cfg,
                bio_cfg=bio_cfg,
                device=device,
                prev_mu_eff_si=prev_mu,
                allowed_commit_mask=allowed,
            )
            grow = active_mode == "seed_growth" or mlp_deploy_vision_grow_enabled()
            if grow:
                mu_out = phys_cfg.viscosity_nd_to_si(
                    torch.from_numpy(frame[:, STATE_CHANNEL_MU_EFF_ND]).to(device=device, dtype=dtype)
                )
                allowed = expand_seed_growth_allowed_mask(
                    allowed, mu_out, data_gt, device, phys_cfg=phys_cfg
                )
        else:
            gate_gt, gate_active = compute_mlp_commit_gates_at_rollout_frame(
                clot_model,
                data_gt,
                ti,
                u_nd=pred_t[:, 0],
                v_nd=pred_t[:, 1],
                species_log=pred_t[:, species_channels.SPECIES_BLOCK],
                phys_cfg=phys_cfg,
                bio_cfg=bio_cfg,
                device=device,
                prev_mu_eff_si=prev_mu,
            )
        gt_np = gate_gt.detach().cpu().numpy().astype(bool)
        active_np = gate_active.detach().cpu().numpy().astype(bool)
        gt_frames.append(gt_np)
        active_frames.append(active_np)
        gt_fracs.append(float(gt_np.mean()))
        active_fracs.append(float(active_np.mean()))

    return gt_frames, active_frames, gt_fracs, active_fracs, active_mode


def _build_commit_mask_figure(
    pos: np.ndarray,
    gt_frames: List[np.ndarray],
    active_frames: List[np.ndarray],
    gt_fracs: List[float],
    active_fracs: List[float],
    custom_times: List[float],
    *,
    case_label: str = "",
    active_mode: str = "neighbor",
    abc_leg: str = "",
    t_final_si: float = 0.0,
) -> Optional[Callable[[int], None]]:
    """Side-by-side MLP commit masks linked to Flow window Time idx slider."""
    n_frames = len(custom_times)
    if n_frames < 1:
        return None

    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
    _C_COMMIT = "#c0392b"
    _C_BULK = "#bdc3c7"

    def _mask_scatter(ax, mask: np.ndarray, title: str):
        m = mask.reshape(-1).astype(bool)
        colors = np.where(m, _C_COMMIT, _C_BULK)
        ax.scatter(pos[:, 0], pos[:, 1], c=colors, s=10, linewidths=0)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.axis("off")

    leg_tag = f"[{abc_leg}] " if abc_leg else ""
    _mask_scatter(
        axs[0],
        gt_frames[0],
        f"Oracle commit (gt_clot) n={int(gt_frames[0].sum())} ({100.0 * gt_fracs[0]:.2f}%)",
    )
    _mask_scatter(
        axs[1],
        active_frames[0],
        f"Active commit ({active_mode}) n={int(active_frames[0].sum())} ({100.0 * active_fracs[0]:.2f}%)",
    )
    fig.subplots_adjust(bottom=0.10, top=0.84)
    fig.text(
        0.5,
        0.02,
        "Red = MLP mu replaces Carreau. Use Time idx slider on Flow window.",
        ha="center",
        fontsize=9,
        color="dimgray",
    )

    def update_commit_frame(frame_idx: int) -> None:
        frame_idx = int(max(0, min(frame_idx, n_frames - 1)))
        t_si = float(custom_times[frame_idx])
        extrap = t_final_si > 0 and t_si > t_final_si + 1e-3
        t_lbl = f"t={t_si:.1f}s{' (extrap)' if extrap else ''}"
        gt_m = gt_frames[frame_idx]
        act_m = active_frames[frame_idx]
        overlap = int((gt_m & act_m).sum())
        gt_n = int(gt_m.sum())
        act_n = int(act_m.sum())
        dice = (2.0 * overlap / (gt_n + act_n)) if (gt_n + act_n) > 0 else 0.0
        _mask_scatter(
            axs[0],
            gt_m,
            f"Oracle commit (gt_clot) n={gt_n} ({100.0 * gt_fracs[frame_idx]:.2f}%)",
        )
        _mask_scatter(
            axs[1],
            act_m,
            f"Active commit ({active_mode}) n={act_n} ({100.0 * active_fracs[frame_idx]:.2f}%)",
        )
        _set_figure_suptitle(
            fig,
            f"{leg_tag}MLP commit masks -- {case_label} ({t_lbl})  overlap={overlap} dice={dice:.3f}",
            fontsize=13,
            subplot_top=0.88,
            title_y=0.96,
        )
        fig.canvas.draw_idle()

    update_commit_frame(0)
    print(
        f"   ->  MLP commit mask panel: oracle gt_clot vs active={active_mode} "
        f"({n_frames} keyframes, Time idx on Flow window)",
        flush=True,
    )
    return update_commit_frame


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
    t_final_si: float = 0.0,
    on_frame_callbacks: Optional[List[Callable[[int], None]]] = None,
    clot_model: Optional[torch.nn.Module] = None,
):
    device = data_biochem.x.device
    bio_cfg = BiochemConfig(phase="biochem")
    t_ref_si = bio_cfg.resolve_biochem_times(data_biochem, device)
    time_indices = _nearest_time_indices(t_ref_si, [float(t) for t in custom_times])
    series = _temporal_series_arrays(
        model_biochem,
        data_biochem,
        pred_biochem_series_np,
        comsol_series_np,
        time_indices=time_indices,
        clot_model=clot_model,
    )
    vel_all = series["vel_all"]
    fib_all = series["fib_all"]
    mat_all = series["mat_all"]
    mu_dyn_all = series["mu_dyn_all"]
    mb_mu1_all = series["mb_mu1_all"]
    mu2_all = series["mu2_all"]
    mu_labels = series["mu_labels"]
    mu2_cap = series["mu2_cap"]
    show_legacy_gel = series["show_legacy_gel"]
    show_comsol = series["show_comsol"]
    vel_vmin = series["vel_vmin"]
    vel_vmax = series["vel_vmax"]
    fib_vmin = series["fib_vmin"]
    fib_vmax = series["fib_vmax"]
    mat_vmin = series["mat_vmin"]
    mat_vmax = series["mat_vmax"]
    mu_vmin = series["mu_vmin"]
    mu_vmax = series["mu_vmax"]
    m1_vmin = series["m1_vmin"]
    m1_vmax = series["m1_vmax"]
    comsol_vel = series.get("comsol_vel")
    comsol_fib = series.get("comsol_fib")
    comsol_mat = series.get("comsol_mat")
    comsol_mu_dyn = series.get("comsol_mu_dyn")
    comsol_mb_mu1 = series.get("comsol_mb_mu1")
    comsol_mu2 = series.get("comsol_mu2")

    ncols = 2 if show_comsol else 1
    col_b, col_c = 0, (1 if show_comsol else None)

    t0_label = f"t={custom_times[0]:.1f}s"

    def _init_scatter(fig, ax, values, title, cmap, vmin, vmax):
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=values[0], cmap=cmap, s=3, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        return sc

    # --- Main: flow + effective viscosity + optional legacy mu1/mu2 ---
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
    if show_legacy_gel:
        scatters["m1_b"] = _init_scatter(
            fig, axs[2, col_b], mb_mu1_all, f"Biochem: {mu_labels['mu1_product']}", _viz_mu_cmap(), m1_vmin, m1_vmax
        )
        scatters["m2_b"] = _init_scatter(
            fig, axs[3, col_b], mu2_all, f"Biochem: {mu_labels['mu2_trigger']}", "Reds", 0.0, mu2_cap
        )
    else:
        scatters["m2_b"] = _init_scatter(
            fig,
            axs[2, col_b],
            mu2_all,
            f"Biochem: {mu_labels['mu2_trigger']}",
            "Reds",
            0.0,
            mu2_cap,
        )
        axs[3, col_b].axis("off")
        axs[3, col_b].set_title("")

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

    state: Dict[str, Any] = {
        "custom_times": list(custom_times),
        "n_frames": len(custom_times),
        "series": series,
        "show_comsol": show_comsol,
    }

    fig.subplots_adjust(bottom=0.14)

    ax_slider = fig.add_axes([0.12, 0.06, 0.50, 0.03])
    time_slider_main = Slider(
        ax=ax_slider,
        label="Time idx",
        valmin=0,
        valmax=max(0, state["n_frames"] - 1),
        valinit=0,
        valstep=1,
        color="teal",
    )

    if on_refresh is not None:
        ax_refresh = fig.add_axes([0.89, 0.04, 0.10, 0.04])
        refresh_button = Button(ax_refresh, refresh_label, color="lightgray", hovercolor="gainsboro")
        refresh_button.on_clicked(lambda _: on_refresh())

    frame_hooks = list(on_frame_callbacks or [])

    def _apply_frame(idx: int) -> None:
        idx = int(max(0, min(idx, state["n_frames"] - 1)))
        times = state["custom_times"]
        t_si = float(times[idx])
        extrap = t_final_si > 0 and t_si > t_final_si + 1e-3
        t_lbl = f"t={t_si:.1f}s{' (extrap)' if extrap else ''}"
        s = state["series"]
        scatters["vel_b"].set_array(s["vel_all"][idx])
        scatters["mu_b"].set_array(s["mu_dyn_all"][idx])
        if "m1_b" in scatters:
            scatters["m1_b"].set_array(s["mb_mu1_all"][idx])
        scatters["m2_b"].set_array(s["mu2_all"][idx])
        if state["show_comsol"]:
            scatters["vel_c"].set_array(s["comsol_vel"][idx])
            scatters["mu_c"].set_array(s["comsol_mu_dyn"][idx])
            scatters["m1_c"].set_array(s["comsol_mb_mu1"][idx])
            scatters["m2_c"].set_array(s["comsol_mu2"][idx])
        for hook in frame_hooks:
            hook(idx)
        _set_figure_suptitle(
            fig, f"{title_prefix} — Flow & rheology ({t_lbl})", fontsize=16, subplot_top=0.88, title_y=0.96
        )
        fig.canvas.draw_idle()

    def _on_slider_change(val) -> None:
        _apply_frame(int(round(float(val))))

    time_slider_main.on_changed(_on_slider_change)

    if state["n_frames"] <= 1:
        print(
            "   [WARN]  Temporal slider: only one keyframe in this rollout "
            f"(n={state['n_frames']}); use --full-viz or VIZ_BIOCHEM_MACRO_STEPS=full for more steps."
        )
    else:
        t_end = state["custom_times"][-1]
        print(
            f"   ->  Temporal slider: {state['n_frames']} keyframes (0..{state['n_frames'] - 1}), "
            f"t in [0, {t_end:.1f}] s (also drives Clot MLP map)",
            flush=True,
        )
    plt.show()


def _run_phase_comparison_graphsage_redirect(
    *, source: str, anchor_stem: Optional[str], time_index: int = -1
) -> bool:
    """Stage-A kinematics viz + GraphSAGE ``biochem_deploy`` deploy. Returns False.

    Replaces the retired GNODE temporal/rheology comparison figures (removed in the
    2026-06 GraphSAGE migration). Kinematics (GINO-DEQ) viz is unchanged; biochem
    visualization routes to ``predict_species_gnn_deploy``.
    """
    if source == "synthetic":
        print(
            "[WARN] Synthetic biochem comparison viz was retired with the GNODE removal. "
            "Use --steady-kin-only for kinematics, or pass a biochem anchor graph.",
            flush=True,
        )
        return False
    try:
        stem = _resolve_anchor_stem(anchor_stem)
    except FileNotFoundError as exc:
        print(f"[WARN] {exc}", flush=True)
        return False
    anchor_path = _anchor_graph_dir() / f"{stem}.pt"
    if not anchor_path.is_file():
        print(f"[WARN] anchor graph missing: {anchor_path}", flush=True)
        return False

    print(f"[i]  Stage-A kinematics viz for {stem} (GINO-DEQ)", flush=True)
    try:
        run_steady_kinematics_viz(cases=[("patient", stem, anchor_path)], time_index=time_index)
    except Exception as exc:  # viz is best-effort
        print(f"[WARN] kinematics viz failed: {exc}", flush=True)

    print(f"[i]  Biochem deploy via GraphSAGE biochem_deploy stack for {stem}", flush=True)
    try:
        from src.inference.predict_species_gnn_deploy import predict_species_gnn_deploy

        result = predict_species_gnn_deploy(anchor_path, flow_source="kinematics")
        print(
            f"[OK]  GraphSAGE deploy {stem}: clot_F1@t_last={result['clot_f1_t_last']:.3f} "
            f"health_pass={result['health_pass']} ckpt={Path(result['species_ckpt']).name}",
            flush=True,
        )
        print(
            "[i]  Full deploy metrics/JSON: "
            f"python -m src.inference.predict_species_gnn_deploy --graph {anchor_path}",
            flush=True,
        )
    except Exception as exc:  # deploy needs CUDA + a trained species ckpt
        print(
            f"[WARN] GraphSAGE biochem deploy unavailable ({exc}). "
            "Run on a CUDA host with a trained species ckpt; see docs/BIOCHEM_GNN.md.",
            flush=True,
        )
    return False


def run_phase_comparison(
    source: str = "anchor",
    regenerate: bool = True,
    seed: int = 42,
    biochem_checkpoint: Optional[str] = None,
    anchor_stem: Optional[str] = None,
    teacher_only: bool = False,
    fast_viz: Optional[bool] = None,
    sim_end_s: Optional[float] = None,
    sim_end_prompt: bool = True,
    clot_phi_checkpoint: Optional[str] = None,
    show_gelation_triggers: Optional[bool] = None,
    deploy_mu_map: Optional[bool] = None,
):
    """Stage-A kinematics viz + GraphSAGE biochem deploy (GNODE retired 2026-06).

    GNODE-specific args (``biochem_checkpoint``, ``teacher_only``,
    ``clot_phi_checkpoint``, ``show_gelation_triggers``, ``deploy_mu_map``,
    ``sim_end_s``) are accepted for CLI back-compat but no longer used.
    """
    return _run_phase_comparison_graphsage_redirect(
        source=source, anchor_stem=anchor_stem, time_index=-1
    )
    if use_anchor:
        print(f"[i]  Data source: COMSOL biochem anchor graph")
    else:
        print(f"[i]  Data source: synthetic vessel (seed={seed})")
    fast_mode = _viz_fast_enabled(fast_viz)
    print(
        f" Fast visualization mode: {'ON' if fast_mode else 'OFF'} "
        f"(dense COMSOL-spaced rollout; use --full-viz or VIZ_FAST=0 for high-fidelity)"
    )
    refresh_state = {"requested": False}

    data_kine_base = None
    anchor_path: Optional[Path] = None
    case_label = ""

    if use_anchor:
        stem = _resolve_anchor_stem(anchor_stem)
        case_label = stem
        print(f"   ->  Anchor stem: {stem} ({_anchor_graph_dir()})")
        try:
            data_biochem, anchor_path = _load_anchor_graph(stem, device)
        except FileNotFoundError as exc:
            print(f"[WARN]  {exc}")
            return False
        if not _graph_has_comsol_trajectory(data_biochem):
            print(
                f"[WARN]  Anchor graph '{stem}' has no usable COMSOL labels in data.y. "
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
                print("[WARN]  No existing cached synthetic data found. Regenerating now...")
                need_regen = True

        if need_regen:
            for d in [raw_dir, graph_kine_base_dir, graph_biochem_dir]:
                for f in d.glob("*"):
                    if f.is_file():
                        f.unlink()

            print("\n Generating 1 complex synthetic vessel for the comparison...")
            vg = VesselGeneratorPhase3(output_dir=raw_dir)
            vg.run_pipeline(n=1, level=1, num_workers=1, seed=seed)

            print("\n Converting mesh to graphs for each phase's specific channel requirements...")
            mg1 = MeshToGraphComplete(
                phase="kinematics", raw_dir=raw_dir, label_dir=raw_dir, proc_dir=graph_kine_base_dir
            )
            mg1.run(max_files=1)
            mg3 = MeshToGraphPhase3(raw_dir=raw_dir, label_dir=raw_dir, proc_dir=graph_biochem_dir)
            mg3.run(max_files=1)
        else:
            print("\n Reusing existing single-case synthetic data.")

        try:
            data_kine_base = _load_single_graph(graph_kine_base_dir, device, "kinematics")
            data_biochem = _load_single_graph(graph_biochem_dir, device, "biochem")
            case_label = "synthetic"
        except FileNotFoundError as exc:
            print(f"[WARN]  Failed to generate or load graph files: {exc}")
            return False

    pos = data_biochem.x[:, :2].cpu().numpy()
    n_nodes = int(data_biochem.x.shape[0])
    n_edges = int(data_biochem.edge_index.shape[1])
    print(f"   ->  Graph: {n_nodes} nodes, {n_edges} edges", flush=True)

    # ------------------------------------------------------------------
    # 4. Load Models
    # ------------------------------------------------------------------
    print("\n Loading trained models...")
    biochem_choice = _resolve_biochem_checkpoint(biochem_checkpoint, teacher_only=teacher_only)
    biochem_ckpt = biochem_choice.path
    biochem_meta, biochem_state = _checkpoint_state_dict(_load_torch_checkpoint(biochem_ckpt))
    _print_biochem_checkpoint_banner(biochem_choice, biochem_meta)
    inject_shell = snapshot_mlp_clot_inject_shell_env()
    fp = biochem_forward_policy_from_checkpoint_meta(biochem_meta)
    if fp is not None:
        apply_biochem_forward_policy_from_checkpoint_meta(biochem_meta, quiet=True)
    else:
        print(
            "   [WARN]  No forward_policy in checkpoint — μ rollout uses current shell env. "
            "Re-train/save teacher to embed policy, or set BIOCHEM_* manually.",
            flush=True,
        )
    restore_mlp_clot_inject_shell_env(inject_shell)

    model_kine_base = None
    kin_ckpt = _try_resolve_kinematics_checkpoint()
    if kin_ckpt is not None:
        print(f"   ->  Using kinematics checkpoint: {kin_ckpt.name}")
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
            print("   ->  kinematics GINO_DEQ from checkpoint model_config.")
        else:
            print(
                "   ->  kinematics GINO_DEQ from weight/reference inference "
                "(re-save kinematics_best.pth to embed model_config)."
            )
        model_kine_base = build_gino_deq_from_ctor(phys_cfg_kine, kin_ctor).to(device)
        model_kine_base.load_state_dict(kin_state, strict=False)
        model_kine_base.eval()
    elif not use_anchor:
        print("   ->  No kinematics checkpoint found; skipping GINO-DEQ column.", flush=True)

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
    print(f"   ->  bio_encoder prior dim: {bio_enc_prior}")
    print(f"   ->  latent_dim: {latent_dim}")
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
                f"   ->  biochem GNODE from checkpoint model_config + forward_policy ({fp_note})."
            )
        else:
            print("   ->  biochem GNODE from checkpoint model_config (saved by train_biochem_corrector).")
    else:
        print(
            f"   ->  biochem GNODE from weight inference (legacy ckpt): "
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
        print(f"   ->  Skipped {len(skipped_bio)} checkpoint key(s) (no target or shape mismatch).")
        siren_skipped = [k for k in skipped_bio if k.startswith("siren_decoder.")]
        if siren_skipped and ctor["use_siren_decoder"]:
            print(
                f"   [WARN]  siren_decoder not loaded ({len(siren_skipped)} keys) — biochem |u| will look wrong. "
                "Set BIOCHEM_USE_SIREN=1 and re-run, or use a checkpoint trained with SIREN."
            )
    model_biochem.eval()
    clot_ckpt_path = _resolve_clot_phi_checkpoint(clot_phi_checkpoint)
    wire_on = deploy_mu_map
    if wire_on is None:
        wire_on = (os.environ.get("BIOCHEM_WIRE_DEPLOY_MU_MAP") or "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
    if wire_on and clot_ckpt_path is not None:
        wire_deploy_mu_map(clot_ckpt=str(clot_ckpt_path), wired=True)
        print(
            f"   ->  Wired deploy MLP mu map ON (mlp_band + vision); ckpt={clot_ckpt_path.name}",
            flush=True,
        )
    inj = attach_clot_phi_injector_to_teacher(
        model_biochem,
        device,
        os.environ.get("BIOCHEM_MLP_CLOT_CKPT") or clot_phi_checkpoint,
    )
    if inj is not None:
        if biochem_mlp_mu_map_enabled():
            print(
                "   ->  MLP mu map v2 ON (Carreau bulk + phi-mask clot mu, closed-loop DEQ)",
                flush=True,
            )
        else:
            print("   ->  MLP clot mu inject v1 ON (closed-loop DEQ coupling)", flush=True)
    elif biochem_mlp_mu_map_enabled():
        print(
            "   [WARN]  BIOCHEM_MLP_MU_MAP=1 but clot-phi injector not attached; "
            "dynamic mu panel will recompute from MLP if ckpt loads for clot map only.",
            flush=True,
        )

    # ------------------------------------------------------------------
    # 5. Inference
    # ------------------------------------------------------------------
    print("\n Running inference...", flush=True)

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
            print("   ->  Kinematics (GINO-DEQ) on visualization mesh…", flush=True)
            pred_kine_on_biochem_mesh = _run_model_once(model_kine_base, kine_data)
            _cuda_sync()
            print(f"   ->  Kinematics done in {time.perf_counter() - t_k0:.1f}s", flush=True)
        pred_kine_base = pred_kine_on_biochem_mesh

        # ``GNODE_Phase3`` expects non-dimensional times (same as training: ``to_t_nd(..., bio_cfg.t_final)``).
        dense_times_si_full = bio_cfg.resolve_biochem_times(data_biochem, device)
        if dense_times_si_full.numel() < 2:
            raise ValueError("Biochem timeline must contain at least two timestamps for rollout visualization.")
        t_ref = float(bio_cfg.t_final)
        t_final_si = float(dense_times_si_full[-1].item())
        default_mode = "dense"
        time_mode = (os.environ.get("VIZ_BIOCHEM_TIME_MODE", default_mode) or default_mode).strip().lower()
        t_end_si = _resolve_viz_sim_end_si(
            sim_end_s,
            t_final_si=t_final_si,
            prompt=sim_end_prompt,
        )
        print(
            f"   ->  Simulation horizon: {t_end_si:.0f}s (COMSOL export t_final~{t_final_si:.0f}s)",
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
                y_oracle = data_biochem.y.to(device=device)
                print(
                    "   ->  K10g oracle clots: mu_eff in wall-adjacent band from GT (truncated to rollout)",
                    flush=True,
                )
            else:
                print("   [WARN]  BIOCHEM_K10G_ORACLE_CLOTS=1 but data.y missing; open-loop rollout.", flush=True)

        rollout_plan = _BiochemVizRolloutPlan(
            data_biochem=data_biochem,
            model_biochem=model_biochem,
            device=device,
            bio_cfg=bio_cfg,
            dense_times_si_full=dense_times_si_full,
            t_ref=t_ref,
            t_final_si=t_final_si,
            fast_mode=fast_mode,
            y_oracle=y_oracle,
            t_end_si=t_end_si,
            use_anchor=use_anchor,
            time_mode=time_mode,
        )
        pred_biochem_series_np, custom_times, comsol_series_np, pred_biochem = rollout_plan.run_keyframed()

        if use_anchor and comsol_series_np is not None:
            comsol_times_si = bio_cfg.resolve_biochem_times(data_biochem, device)
            idx_comsol_final = _nearest_time_indices(comsol_times_si, [t_final_si])[0]
            if data_biochem.y.dim() == 3:
                comsol_final_np = data_biochem.y[idx_comsol_final].detach().cpu().numpy()
            else:
                comsol_final_np = data_biochem.y.detach().cpu().numpy()
            print(
                f"   ->  COMSOL reference: {int(comsol_times_si.numel())} export steps; "
                f"slider aligned to model keyframes",
                flush=True,
            )
        else:
            comsol_final_np = None

    pred_biochem_np = pred_biochem.detach().cpu().numpy()
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
    time_label = f"t_final~{t_final_si:.0f}s (last COMSOL export step)"
    print(" Generating comparison plots...")
    if clot_ckpt_path is None:
        clot_ckpt_path = _resolve_clot_phi_checkpoint(clot_phi_checkpoint)
    show_gelation = show_gelation_triggers
    if show_gelation is None:
        show_gelation = (os.environ.get("VIZ_SHOW_GELATION") or "").strip().lower() in ("1", "true", "yes", "on")
    if clot_ckpt_path is not None and show_gelation_triggers is None and not show_gelation:
        show_gelation = False
    recompute_raw = (os.environ.get("VIZ_MU_DYNAMIC_RECOMPUTE") or "").strip().lower()
    if biochem_mlp_mu_map_enabled() and recompute_raw in ("1", "true", "yes", "on"):
        mu_dyn_note = "MLP recompute on pred u,v (not ch3)"
    else:
        mu_dyn_note = "rollout ch3 (matches abc probe)"
    abc_leg = (os.environ.get("VIZ_ABC_LEG") or "").strip()
    if abc_leg:
        mu_dyn_note = f"{abc_leg}; {mu_dyn_note}"
    guide_lines = [
        "   Figure guide (matplotlib order):",
        "     - Steady GINO-DEQ vs COMSOL GT @ t=0 (when anchor labels present)",
        f"     - Dynamic mu_eff -- closed-loop rollout at {time_label} ({mu_dyn_note})",
    ]
    if clot_ckpt_path is not None:
        guide_lines.append(
            f"     - Clot map (MLP mu) -- "
            + (
                "rollout ch3 (wired deploy; matches dynamic mu)"
                if biochem_mlp_mu_map_enabled()
                else f"offline MLP readout; ckpt {clot_ckpt_path.name}"
            )
            + "; updates with Time idx on Flow window"
        )
    if show_gelation:
        guide_lines.append(
            "     - Gelation triggers -- Mat/FI terms in forward (optional; use --show-gelation-triggers)"
        )
    guide_lines.append(
        "     - Temporal inspector -- Flow/rheology with Time slider (keyframes, not only t_final)"
    )
    print("\n".join(guide_lines))

    if pred_kine_base_np is not None and model_kine_base is not None:
        gt_t0 = _steady_kine_target_tensor(data_biochem, time_index=0) if use_anchor else None
        gt_t0_np = gt_t0.detach().cpu().numpy() if gt_t0 is not None else None
        rel_kine_t0 = _rel_l2_uvp(pred_kine_base_np, gt_t0_np) if gt_t0_np is not None else None
        if gt_t0_np is not None and float(np.linalg.norm(gt_t0_np[:, :3])) > 1e-6:
            if rel_kine_t0 is not None:
                print(
                    f"   ->  Steady GINO-DEQ vs COMSOL GT @ t=0: rel_L2(uvp)={rel_kine_t0:.4f}",
                    flush=True,
                )
            _show_steady_kinematics_pred_vs_gt(
                pos,
                pred_kine_base_np,
                gt_t0_np,
                case_label=case_label,
                cohort="anchor" if use_anchor else "synthetic",
                rel_l2=rel_kine_t0,
                gt_row_label="COMSOL GT (t=0)",
            )
        else:
            if use_anchor:
                print(
                    "   [WARN]  No COMSOL [u,v,p] at t=0 on anchor; steady kine figure is pred-only.",
                    flush=True,
                )
            _show_kinematics_static_figure(
                pos, vel_kine_base, p_kine_base, mu_kine_base, case_label=case_label
            )

    clot_model_viz: Optional[torch.nn.Module] = None
    if clot_ckpt_path is not None:
        try:
            clot_model_viz, _ = _load_clot_phi_model(clot_ckpt_path, device)
            print(f"   ->  Clot MLP loaded for v2 mu panels: {clot_ckpt_path.name}", flush=True)
        except Exception as exc:
            print(f"   [WARN]  Clot MLP load failed: {exc}", flush=True)
    elif biochem_mlp_mu_map_enabled():
        print(
            "   [WARN]  BIOCHEM_MLP_MU_MAP=1 but no clot-phi ckpt; dynamic mu uses Carreau-only.",
            flush=True,
        )

    # --- FIGURE 2: rollout mu_eff at final time ---
    mu_labels = _biochem_mu_viz_labels()
    idx_final = int(_nearest_time_indices(bio_cfg.resolve_biochem_times(data_biochem, device), [t_final_si])[0])
    mu_dyn_rollout_np = _rollout_mu_eff_si_numpy(model_biochem.phys_cfg, pred_biochem_np)
    mu_dyn_np = mu_dyn_rollout_np
    if biochem_mlp_mu_map_enabled() and recompute_raw in ("1", "true", "yes", "on"):
        mu_dyn_np = _committed_mu_single_frame_numpy(
            data_biochem,
            pred_biochem_np,
            idx_final,
            clot_model=clot_model_viz,
            phys_cfg=model_biochem.phys_cfg,
            bio_cfg=bio_cfg,
        )

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
    if abc_leg:
        rheo_title = f"[{abc_leg}] {rheo_title}"
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
        f"Biochem: {mu_labels['mu_dynamic']} ({mu_dyn_note})",
        _viz_mu_cmap(),
        vmin=mu_si_vmin,
        vmax=mu_si_vmax,
    )
    fig2.tight_layout(rect=(0, 0.03, 1, 0.90))
    _set_figure_suptitle(fig2, rheo_title, fontsize=14)

    clot_mu_series: Optional[Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]] = None
    if clot_model_viz is not None:
        try:
            t_ref_times_si = bio_cfg.resolve_biochem_times(data_biochem, device)
            clot_mu_series = _precompute_clot_phi_mu_series(
                data_biochem,
                clot_model_viz,
                model_biochem.phys_cfg,
                bio_cfg,
                device,
                pred_biochem_series_np,
                list(custom_times),
                show_gt=use_anchor and comsol_final_np is not None,
                t_ref_times_si=t_ref_times_si,
            )
        except Exception as exc:
            print(f"   [WARN]  Clot MLP precompute skipped: {exc}", flush=True)
            if show_gelation_triggers is None:
                show_gelation = True

    if show_gelation:
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
    clot_frame_fn: Optional[Callable[[int], None]] = None
    commit_mask_frame_fn: Optional[Callable[[int], None]] = None
    if clot_mu_series is not None:
        mu_gt_f, mu_pr_f, region_f = clot_mu_series
        print(" Opening Clot MLP map (linked to Time idx slider)...")
        clot_frame_fn = _build_clot_phi_figure(
            pos,
            mu_gt_f,
            mu_pr_f,
            region_f,
            list(custom_times),
            case_label=case_label,
            show_gt=use_anchor and comsol_final_np is not None,
            t_final_si=t_final_si,
        )
    if (
        clot_model_viz is not None
        and biochem_mlp_mu_map_enabled()
        and mlp_mu_map_phi_gate_enabled()
        and _viz_commit_mask_enabled()
        and use_anchor
    ):
        try:
            t_ref_times_si = bio_cfg.resolve_biochem_times(data_biochem, device)
            gt_m, act_m, gt_f, act_f, active_mode = _precompute_mlp_commit_mask_series(
                data_biochem,
                clot_model_viz,
                model_biochem.phys_cfg,
                bio_cfg,
                device,
                pred_biochem_series_np,
                list(custom_times),
                t_ref_times_si,
            )
            print(" Opening MLP commit mask compare (linked to Time idx slider)...")
            commit_mask_frame_fn = _build_commit_mask_figure(
                pos,
                gt_m,
                act_m,
                gt_f,
                act_f,
                list(custom_times),
                case_label=case_label,
                active_mode=active_mode,
                abc_leg=abc_leg or "",
                t_final_si=t_final_si,
            )
        except Exception as exc:
            print(f"   [WARN]  MLP commit mask panel skipped: {exc}", flush=True)
    print(" Opening interactive Biochem temporal slider...")
    frame_hooks: List[Callable[[int], None]] = []
    if clot_frame_fn is not None:
        frame_hooks.append(clot_frame_fn)
    if commit_mask_frame_fn is not None:
        frame_hooks.append(commit_mask_frame_fn)
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
        t_final_si=t_final_si,
        on_frame_callbacks=frame_hooks,
        clot_model=clot_model_viz,
    )
    return refresh_state["requested"]


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    default_anchor = _parser_default_anchor_stem()
    parser = argparse.ArgumentParser(
        description=(
            "Visualize biochem teacher/corrector vs COMSOL anchor labels. "
            f"Default: biochem anchor graph (--anchor {default_anchor}). "
            "Pass --synthetic for a generated vessel with no GT column."
        )
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help=(
            "Use a generated vessel under data/phase_comparison_test instead of a biochem anchor graph. "
            "Default (no flag) loads a patient anchor from graphs_biochem_anchors."
        ),
    )
    parser.add_argument(
        "--anchor",
        type=str,
        default=default_anchor,
        metavar="STEM",
        help=(
            f"Biochem anchor stem under graphs_biochem_anchors (default: {default_anchor}). "
            "Ignored when --synthetic is set. Override default with VIZ_ANCHOR_STEM."
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
            "Biochem weights file. Default: biochem_teacher_best_high_mu.pth -> "
            "biochem_teacher_last.pth (then legacy fallbacks). Override via path or VIZ_BIOCHEM_CHECKPOINT."
        ),
    )
    parser.add_argument(
        "--teacher-only",
        action="store_true",
        help=(
            "Teacher checkpoints only: biochem_teacher_best_high_mu.pth -> biochem_teacher_last.pth. "
            "Same as VIZ_BIOCHEM_REQUIRE_TEACHER=1."
        ),
    )
    parser.add_argument(
        "--full-viz",
        action="store_true",
        help="Disable fast viz (dense rollout up to 16+ macro steps, finer ODE; sets VIZ_FAST=0).",
    )
    parser.add_argument(
        "--steady-kin-only",
        action="store_true",
        help=(
            "Skip biochem rollout; only steady GINO-DEQ on the given graph(s). "
            "Shows prediction vs labels when y is present."
        ),
    )
    parser.add_argument(
        "--steady-kin-compare",
        action="store_true",
        help=(
            "With --steady-kin-only: one patient anchor (see --anchor) and one kinematics "
            "synthetic graph (--kine-vessel or --kine-graph)."
        ),
    )
    parser.add_argument(
        "--kine-graph",
        type=str,
        default=None,
        metavar="PATH",
        help="Processed kinematics or patient .pt for steady-kin-only mode.",
    )
    parser.add_argument(
        "--kine-vessel",
        type=int,
        default=0,
        help="With --steady-kin-compare: vessel index under graphs_kinematics/newtonian (default 0).",
    )
    parser.add_argument(
        "--time-index",
        type=int,
        default=-1,
        help="Biochem anchor label time index for GT row (default -1 = last export step).",
    )
    parser.add_argument(
        "--sim-end-s",
        type=float,
        default=None,
        metavar="SEC",
        help=(
            "Simulation end time [s] for GNODE rollout (default 30000, or prompt). "
            "Must be >= COMSOL t_final on anchors. Env: VIZ_SIM_END_S."
        ),
    )
    parser.add_argument(
        "--no-sim-end-prompt",
        action="store_true",
        help="Skip interactive simulation-length prompt (use --sim-end-s, VIZ_SIM_END_S, or default 30000).",
    )
    parser.add_argument(
        "--deploy-mu-map",
        action="store_true",
        help=(
            "Wire closed-loop Leg B MLP mu map (mlp_band + vision restrict) into rollout; "
            "clot map pred panel uses rollout ch3 (matches dynamic mu)."
        ),
    )
    parser.add_argument(
        "--clot-phi-checkpoint",
        type=str,
        default=None,
        help=(
            "Clot-phi MLP weights for spatial clot panel (default: clot_baseline/clot_phi_best.pth if present). "
            "Set VIZ_NO_CLOT_PHI=1 to disable."
        ),
    )
    parser.add_argument(
        "--show-gelation-triggers",
        action="store_true",
        help="Show Mat/FI gelation trigger figure (hidden by default when clot-phi ckpt is loaded).",
    )
    parser.add_argument(
        "--no-clot-phi",
        action="store_true",
        help="Skip clot-phi MLP panel (raw mu_eff + optional gelation only).",
    )
    args = parser.parse_args()

    if args.full_viz:
        os.environ["VIZ_FAST"] = "0"
    if args.no_clot_phi:
        os.environ["VIZ_NO_CLOT_PHI"] = "1"

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

    if args.steady_kin_only:
        cases: List[Tuple[str, str, Path]] = []
        if args.steady_kin_compare:
            anchor_stem = _resolve_anchor_stem(args.anchor)
            k_anchor = _kinematics_anchor_graph_path(anchor_stem, "newtonian")
            if k_anchor.is_file():
                cases.append(("patient", anchor_stem, k_anchor))
            else:
                cases.append(
                    ("patient", anchor_stem, _anchor_graph_dir() / f"{anchor_stem}.pt")
                )
            if args.kine_graph:
                kpath = Path(args.kine_graph)
                if not kpath.is_absolute():
                    kpath = get_project_root() / kpath
                cases.append(("kinematics", kpath.stem, kpath))
            else:
                kdir = _kinematics_graph_dir("newtonian")
                kpath = kdir / f"vessel_{int(args.kine_vessel)}.pt"
                cases.append(("kinematics", f"vessel_{int(args.kine_vessel)}", kpath))
        elif args.kine_graph:
            kpath = Path(args.kine_graph)
            if not kpath.is_absolute():
                kpath = get_project_root() / kpath
            cohort = "patient" if "biochem_anchors" in kpath.as_posix() else "kinematics"
            cases.append((cohort, kpath.stem, kpath))
        elif args.anchor or not args.synthetic:
            anchor_stem = _resolve_anchor_stem(args.anchor)
            cases.append(
                ("patient", anchor_stem, _anchor_graph_dir() / f"{anchor_stem}.pt")
            )
        else:
            raise ValueError(
                "Use --steady-kin-compare, --kine-graph PATH, or --anchor STEM with --steady-kin-only."
            )
        run_steady_kinematics_viz(cases=cases, time_index=int(args.time_index))
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
    if source == "anchor":
        if not anchor_stems:
            raise SystemExit(
                f"No biochem anchor graphs under {_anchor_graph_dir()}. "
                "Export anchors or pass --synthetic for a generated vessel."
            )
        args_anchor = Path(args.anchor).stem if str(args.anchor).endswith(".pt") else str(args.anchor).strip()
        try:
            resolved_anchor = _resolve_anchor_stem(args_anchor)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc
        anchor_idx = anchor_stems.index(resolved_anchor)
        print(f"[i]  Default anchor mode: {resolved_anchor} ({_anchor_graph_dir()})", flush=True)

    while True:
        current_anchor = anchor_stems[anchor_idx] if source == "anchor" else None
        refresh_requested = run_phase_comparison(
            source=source,
            regenerate=regenerate,
            seed=seed,
            biochem_checkpoint=args.biochem_checkpoint,
            anchor_stem=current_anchor,
            teacher_only=args.teacher_only,
            fast_viz=not args.full_viz,
            sim_end_s=args.sim_end_s,
            sim_end_prompt=not args.no_sim_end_prompt,
            clot_phi_checkpoint=args.clot_phi_checkpoint,
            show_gelation_triggers=True if args.show_gelation_triggers else None,
            deploy_mu_map=True if args.deploy_mu_map else None,
        )
        if not refresh_requested:
            break
        if source == "anchor":
            if not anchor_stems:
                break
            anchor_idx = (anchor_idx + 1) % len(anchor_stems)
            print(f" Next anchor: {anchor_stems[anchor_idx]}")
        else:
            regenerate = True
            seed = int(np.random.default_rng().integers(0, 2**31 - 1))
            print(f" Refresh requested. Regenerating synthetic vessel with new seed: {seed}")