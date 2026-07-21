"""Steady kinematics viz + GraphSAGE biochem deploy smoke.

GNODE teacher/corrector temporal inspector was removed in the 2026-06 GraphSAGE
migration. Biochem visualization routes to ``predict_species_gnn_deploy``;
for full species timelines use ``scripts/viz_species_gnn_deploy.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch

from src.architecture.ginodeq import GINO_DEQ
from src.architecture.kinematics_model_config import (
    build_gino_deq_from_ctor,
    kinematics_checkpoint_tensors,
    resolve_gino_deq_ctor_kwargs,
)
from src.config import PhysicsConfig, STATE_CHANNEL_MU_EFF_ND, VesselConfig
from src.utils.channel_schema import infer_missing_schema
from src.utils.kinematics_paths import kinematics_graph_rheology_dir
from src.utils.paths import get_project_root, resolve_checkpoint

# Standard channel indices across all models for kinematics
_CHANNEL = dict(u=0, v=1, p=2, mu_eff=STATE_CHANNEL_MU_EFF_ND)
_KIN_CKPT_CANDIDATES = ("kinematics_best.pth", "kinematics_ckpt_latest.pth", "kinematics_ckpt_100.pth")
_DEFAULT_VAL_ANCHOR_STEM = "patient007"


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
    """Plot a scalar field on an unstructured mesh using tripcolor."""
    triang = mtri.Triangulation(pos[:, 0], pos[:, 1])

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

    for row_i, (vel, press, mu, row_lbl) in enumerate(row_specs):
        for col_i, (values, col_lbl, cmap) in enumerate(zip((vel, press, mu), col_titles, cmaps)):
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


def _run_phase_comparison_graphsage_redirect(
    *, source: str, anchor_stem: Optional[str], time_index: int = -1
) -> None:
    """Stage-A kinematics viz + GraphSAGE ``biochem_gnn`` deploy smoke."""
    if source == "synthetic":
        print(
            "[WARN] Synthetic biochem comparison viz was retired with the GNODE removal. "
            "Use --steady-kin-only for kinematics, or pass a biochem anchor graph.",
            flush=True,
        )
        return
    try:
        stem = _resolve_anchor_stem(anchor_stem)
    except FileNotFoundError as exc:
        print(f"[WARN] {exc}", flush=True)
        return
    anchor_path = _anchor_graph_dir() / f"{stem}.pt"
    if not anchor_path.is_file():
        print(f"[WARN] anchor graph missing: {anchor_path}", flush=True)
        return

    print(f"[i]  Stage-A kinematics viz for {stem} (GINO-DEQ)", flush=True)
    try:
        run_steady_kinematics_viz(cases=[("patient", stem, anchor_path)], time_index=time_index)
    except Exception as exc:  # viz is best-effort
        print(f"[WARN] kinematics viz failed: {exc}", flush=True)

    print(f"[i]  Biochem deploy via GraphSAGE biochem_gnn stack for {stem}", flush=True)
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
        print(
            "[i]  Species timeline viz: "
            "python scripts/viz_species_gnn_deploy.py",
            flush=True,
        )
    except Exception as exc:  # deploy needs CUDA + a trained species ckpt
        print(
            f"[WARN] GraphSAGE biochem deploy unavailable ({exc}). "
            "Run on a CUDA host with a trained species ckpt; see docs/BIOCHEM_GNN.md.",
            flush=True,
        )


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
    **_ignored: Any,
) -> None:
    """Stage-A kinematics viz + GraphSAGE biochem deploy (GNODE retired 2026-06).

    Legacy GNODE kwargs are accepted for call-site back-compat but ignored.
    """
    del (
        regenerate,
        seed,
        biochem_checkpoint,
        teacher_only,
        fast_viz,
        sim_end_s,
        sim_end_prompt,
        clot_phi_checkpoint,
        show_gelation_triggers,
        deploy_mu_map,
        _ignored,
    )
    _run_phase_comparison_graphsage_redirect(
        source=source, anchor_stem=anchor_stem, time_index=-1
    )


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    default_anchor = _parser_default_anchor_stem()
    parser = argparse.ArgumentParser(
        description=(
            "Steady GINO-DEQ kinematics viz + GraphSAGE biochem deploy smoke. "
            f"Default: biochem anchor graph (--anchor {default_anchor}). "
            "For species timelines use scripts/viz_species_gnn_deploy.py."
        )
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Retired with GNODE; prints a warning (use --steady-kin-only instead).",
    )
    parser.add_argument(
        "--anchor",
        type=str,
        default=default_anchor,
        metavar="STEM",
        help=(
            f"Biochem anchor stem under graphs_biochem_anchors (default: {default_anchor}). "
            "Override default with VIZ_ANCHOR_STEM."
        ),
    )
    parser.add_argument(
        "--list-anchors",
        action="store_true",
        help="List available anchor graph stems and exit",
    )
    parser.add_argument(
        "--steady-kin-only",
        action="store_true",
        help=(
            "Skip biochem deploy; only steady GINO-DEQ on the given graph(s). "
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
    # Legacy no-op flags kept so old go_* wrappers do not crash on argparse.
    parser.add_argument("--biochem-checkpoint", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--teacher-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--full-viz", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--sim-end-s", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--no-sim-end-prompt", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--deploy-mu-map", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--clot-phi-checkpoint", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--show-gelation-triggers", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-clot-phi", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--regenerate", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--reuse", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=42, help=argparse.SUPPRESS)
    args = parser.parse_args()

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

    legacy_flags = [
        name
        for name, val in (
            ("--teacher-only", args.teacher_only),
            ("--biochem-checkpoint", args.biochem_checkpoint),
            ("--clot-phi-checkpoint", args.clot_phi_checkpoint),
            ("--deploy-mu-map", args.deploy_mu_map),
            ("--show-gelation-triggers", args.show_gelation_triggers),
            ("--full-viz", args.full_viz),
            ("--sim-end-s", args.sim_end_s is not None),
        )
        if val
    ]
    if legacy_flags:
        print(
            "[WARN] GNODE teacher viz flags are retired and ignored: "
            + ", ".join(legacy_flags)
            + ". Use scripts/viz_species_gnn_deploy.py for species timelines.",
            flush=True,
        )

    source = "synthetic" if args.synthetic else "anchor"
    if source == "anchor":
        try:
            resolved_anchor = _resolve_anchor_stem(args.anchor)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"[i]  Default anchor mode: {resolved_anchor} ({_anchor_graph_dir()})", flush=True)
        run_phase_comparison(source="anchor", anchor_stem=resolved_anchor)
    else:
        run_phase_comparison(source="synthetic")
