"""Deploy stack viz: GT clot | GNN clot map | hop-colored error row.

Uses ``species_gnn_deploy_env`` (species ckpt, beta, per-anchor overrides).
Deploy flow is pred kinematics by default.

Examples:
    python scripts/viz_species_gnn_deploy.py --anchor patient007
    python scripts/viz_species_gnn_deploy.py --all-anchors
    python scripts/viz_species_gnn_deploy.py --legacy-six --max-frames 4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.clot_continuous_time import macro_tau_at_index  # noqa: E402
from src.core_physics.clot_nucleation_mask import resolve_nucleation_eligibility  # noqa: E402
from src.core_physics.species_gnn_clot_rollout import (  # noqa: E402
    load_species_gnn_rollout_bundle,
    prepare_species_gnn_rollout_static,
    rollout_species_gnn_phi_trajectory,
    species_gnn_rollout_ckpt,
)
from src.core_physics.species_gnn_ladder_viz import ladder_viz_times, scatter_clot_error_panel  # noqa: E402
from src.core_physics.species_pushforward_continuous import (  # noqa: E402
    BIOCHEM_ANCHORS_6,
    discover_biochem_anchors,
    parse_biochem_train_anchors,
)
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.evaluation.viz_clot_phi_simple import _scatter_fullmesh_region  # noqa: E402
from src.evaluation.clot_relaxed_metrics import compute_clot_relaxed_metrics  # noqa: E402
from src.evaluation.viz_clot_trigger import clot_trigger_viz_f1  # noqa: E402
from src.inference.species_gnn_deploy_env import (  # noqa: E402
    DEFAULT_MANIFEST,
    load_deploy_manifest,
    species_ckpt_for_anchor,
    species_gnn_deploy_env,
)
from src.utils.paths import get_project_root  # noqa: E402


ROW_GT = "Ground truth (GT)"
ROW_PRED = "Model prediction"
ROW_ERR = "Error (FP=red, FN=blue)"


def _viz_out_dir() -> Path:
    p = get_project_root() / "outputs/biochem/viz/species_gnn_deploy"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _patient_label(anchor: str) -> str:
    m = re.match(r"^patient(\d+)$", anchor.strip(), flags=re.I)
    if m:
        return f"patient {m.group(1)}"
    return anchor.strip()


def _nucleation_masks_for_phi_trajectory(
    data,
    phi_traj: dict[int, torch.Tensor],
    *,
    phys: PhysicsConfig,
    bio: BiochemConfig,
    device: torch.device,
    nucleation_hops: int = 1,
) -> dict[int, torch.Tensor]:
    """Replay deploy nucleation eligibility E(t) from predicted phi commits."""
    masks: dict[int, torch.Tensor] = {}
    commits_prev: torch.Tensor | None = None
    for t in sorted(int(k) for k in phi_traj.keys()):
        elig = resolve_nucleation_eligibility(
            data,
            t,
            device,
            phys,
            bio,
            commits_prev=commits_prev,
            growth_seed="pred",
            nucleation_hops=nucleation_hops,
        )
        masks[t] = elig.float()
        phi = phi_traj[t].reshape(-1)
        commits_prev = (phi >= 0.5).bool()
    return masks


def _column_title(t: int, tau: float, *, guiding: float, f05: float) -> str:
    return f"t={t} tau={tau:.2f}\nF0.5={f05:.2f}  guiding={guiding:.2f}"


def _scatter_clot_panel(
    ax,
    pos: np.ndarray,
    vals: np.ndarray,
    title: str,
    *,
    s: float,
    layer_positive_on_top: bool,
    nuc_eligible: np.ndarray | None = None,
    clot_thresh: float = 0.5,
) -> None:
    """Clot phi panel: light-blue bulk + red commits; optional dark-blue nucleation hints."""
    full_region = np.ones(vals.reshape(-1).shape[0], dtype=bool)
    _scatter_fullmesh_region(
        ax,
        pos,
        vals,
        full_region,
        title,
        cmap="bwr",
        vmin=0.0,
        vmax=1.0,
        s=s,
        layer_positive_on_top=layer_positive_on_top,
        positive_thresh=clot_thresh,
    )
    if nuc_eligible is None:
        return
    v = vals.reshape(-1)
    elig = nuc_eligible.reshape(-1).astype(bool)
    show = elig & (v < float(clot_thresh))
    if not show.any():
        return
    ax.scatter(
        pos[show, 0],
        pos[show, 1],
        c="#08306b",
        s=max(s * 0.95, 2.0),
        linewidths=0,
        alpha=0.9,
        zorder=3,
    )

def _hop_error_legend(fig) -> None:
    """Create a legend for error hop distances with dynamic colors.
    Colors are defined in src.core_physics.species_gnn_ladder_viz as FP_COLORS and FN_COLORS.
    Positive values (FP) use FP_COLORS, negative values (FN) use FN_COLORS.
    """
    from matplotlib.patches import Patch
    # Import color dictionaries
    from src.core_physics.species_gnn_ladder_viz import FP_COLORS, FN_COLORS
    # Determine max hop indices from the dictionaries
    max_fp_hop = max(FP_COLORS.keys())
    max_fn_hop = max(FN_COLORS.keys())
    legend_elements = []
    # Positive (FP) entries
    for hop in range(0, max_fp_hop + 1):
        color = FP_COLORS[hop]
        label = f"+{hop}: FP Hop {hop}" if hop > 0 else f"+0: FP Wall (Hop 0)"
        legend_elements.append(Patch(facecolor=color, label=label))
    # Negative (FN) entries
    for hop in range(0, max_fn_hop + 1):
        color = FN_COLORS[hop]
        label = f"-{hop}: FN Hop {hop}" if hop > 0 else f"-0: FN Wall (Hop 0)"
        legend_elements.append(Patch(facecolor=color, label=label))
    fig.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        ncol=1,
        fontsize=8,
        title="Error Hop Distance\n(-x: FN, +x: FP)",
        title_fontsize=9,
    )

def viz_anchor_deploy(
    anchor: str,
    *,
    manifest: dict,
    device: torch.device,
    flow_source: str,
    manifest_name: str,
    max_frames: int,
    scatter_size: float,
    include_error_row: bool,
    out: Path | None = None,
    bundle_cache: dict | None = None,
) -> dict:
    """Run one anchor: GT | pred | hop-colored error PNG + sidecar JSON.

    ``bundle_cache`` is an optional ``{ckpt_path_str: SpeciesGnnRolloutBundle}`` shared across
    anchors so multi-anchor viz does not reload the same species weights from disk each time.
    """
    root = get_project_root()
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    data = torch.load(
        root / "data/processed/graphs_biochem_anchors" / f"{anchor}.pt",
        map_location=device,
        weights_only=False,
    )
    pos = data.x[:, :2].detach().cpu().numpy()
    n_nodes = int(data.num_nodes)

    from src.core_physics.clot_phi_simple import _wall_mask_from_data
    from src.core_physics.species_pushforward_continuous import compute_hop_distances

    wall_mask_full = _wall_mask_from_data(data, device, n_nodes)
    hop_distances_gpu = compute_hop_distances(data.edge_index, wall_mask_full, n_nodes)
    hop_distances_np = hop_distances_gpu.detach().cpu().numpy()

    times = ladder_viz_times(int(data.y.shape[0]), max_frames=int(max_frames))
    mask = torch.ones(n_nodes, device=device, dtype=torch.bool)

    ckpt_pick = species_ckpt_for_anchor(anchor, manifest, prefer_loao=True)
    beta_ov = (manifest.get("beta_overrides") or {}).get(anchor, "")

    print(f"[i] anchor={anchor} flow={flow_source} ckpt={ckpt_pick.name}", flush=True)
    if beta_ov:
        print(f"[i] beta_override={beta_ov}", flush=True)

    t0 = time.perf_counter()
    with species_gnn_deploy_env(
        manifest,
        overrides={"T0_R4_FLOW_SOURCE": flow_source},
        anchor=anchor,
        prefer_loao=True,
    ):
        ckpt = Path(species_gnn_rollout_ckpt())
        ckpt_key = str(ckpt.resolve())
        bundle = None
        if bundle_cache is not None and ckpt_key in bundle_cache:
            bundle = bundle_cache[ckpt_key]
        else:
            bundle = load_species_gnn_rollout_bundle(ckpt, device=device)
            if bundle is not None and bundle_cache is not None:
                bundle_cache[ckpt_key] = bundle
        if bundle is None:
            raise FileNotFoundError(f"missing ckpt: {ckpt}")
        static = prepare_species_gnn_rollout_static(data, device=device)
        phi_gnn = rollout_species_gnn_phi_trajectory(
            data, bundle, static, phys_cfg=phys, bio_cfg=bio, device=device,
            flow_source=flow_source,
        )
        beta_used = os.environ.get("SPECIES_GELATION_BETA_OVERRIDE", "")
    nuc_masks = _nucleation_masks_for_phi_trajectory(
        data, phi_gnn, phys=phys, bio=bio, device=device,
    )
    elapsed = time.perf_counter() - t0
    num_steps = len(phi_gnn)
    ms_per_step = (elapsed / num_steps) * 1000 if num_steps > 0 else 0.0
    speed_note = f"ML Rollout Speed: {num_steps} steps in {elapsed:.2f}s ({ms_per_step:.1f} ms/step)"
    print(f"[i] {speed_note}", flush=True)

    row_labels = [ROW_GT, ROW_PRED]
    if include_error_row:
        row_labels.append(ROW_ERR)
    fig, axes = plt.subplots(
        len(row_labels), len(times),
        figsize=(2.7 * len(times), 2.5 * len(row_labels)),
        squeeze=False,
    )
    fig.suptitle(
        f"biochem GNN deploy -- {_patient_label(anchor)} | {flow_source} flow | {manifest_name}\n{speed_note}",
        fontsize=11,
        y=1.02,
    )

    frames: list[dict] = []
    for j, t in enumerate(times):
        phi_gt = gt_clot_phi_at_time(data, int(t), phys, device)
        p_gnn = phi_gnn[int(t)]
        p_nuc = nuc_masks[int(t)]
        tau = float(macro_tau_at_index(data, int(t), bio_cfg=bio))
        ei = data.edge_index.to(device=device)
        m_gnn = clot_trigger_viz_f1(p_gnn, phi_gt, mask)
        g_gnn = compute_clot_relaxed_metrics(p_gnn.reshape(-1), phi_gt.reshape(-1), ei)
        col_title = _column_title(
            int(t), tau,
            guiding=float(g_gnn["clot_guiding"]),
            f05=float(g_gnn["clot_relaxed_f05"]),
        )
        phi_gt_np = phi_gt.detach().cpu().numpy()
        p_gnn_np = p_gnn.detach().cpu().numpy()
        p_nuc_np = p_nuc.detach().cpu().numpy()
        scatter_s = float(scatter_size)
        _scatter_clot_panel(
            axes[0, j], pos, phi_gt_np,
            row_labels[0] if j == 0 else "",
            s=scatter_s, layer_positive_on_top=True,
        )
        _scatter_clot_panel(
            axes[1, j], pos, p_gnn_np,
            row_labels[1] if j == 0 else "",
            s=scatter_s, layer_positive_on_top=True,
            nuc_eligible=p_nuc_np,
        )
        axes[1, j].set_title(col_title, fontsize=10, pad=6)
        axes[1, j].title.set_fontweight("bold")
        err_counts: dict[str, int] = {}
        if include_error_row:
            err_counts = scatter_clot_error_panel(
                axes[2, j],
                pos,
                phi_gt_np,
                p_gnn_np,
                row_labels[2] if j == 0 else "",
                s=scatter_s,
                hop_distances=hop_distances_np,
            )
            axes[2, j].set_title(
                f"FP={err_counts['fp']}  FN={err_counts['fn']}",
                fontsize=8,
                pad=4,
            )
        frames.append({
            "time": int(t),
            "tau": tau,
            "gnn_f1": float(m_gnn["clot_f1"]),
            "gnn_guiding": float(g_gnn["clot_guiding"]),
            "gnn_f05": float(g_gnn["clot_relaxed_f05"]),
            "gnn_dil_iou": float(g_gnn["clot_dilation_iou"]),
            "nucleation_frac": float(p_nuc.mean().item()),
            **{f"err_{k}": int(v) for k, v in err_counts.items()},
        })

    if include_error_row:
        _hop_error_legend(fig)

    fig.tight_layout(rect=[0, 0, 0.9, 0.98])
    if out is None:
        out = _viz_out_dir() / f"deploy_{anchor}_{flow_source}.png"
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out}", flush=True)

    payload = {
        "anchor": anchor,
        "patient_label": _patient_label(anchor),
        "flow_source": flow_source,
        "species_ckpt": str(ckpt_pick),
        "beta_override": str(beta_ov or beta_used),
        "ml_rollout_speed_note": speed_note,
        "ml_rollout_elapsed_s": elapsed,
        "ml_rollout_ms_per_step": ms_per_step,
        "rows": row_labels,
        "times": times,
        "frames": frames,
        "final_guiding": frames[-1]["gnn_guiding"] if frames else None,
        "final_f05": frames[-1]["gnn_f05"] if frames else None,
        "final_f1": frames[-1]["gnn_f1"] if frames else None,
        "png": str(out.relative_to(root)).replace("\\", "/"),
    }
    meta = out.with_suffix(".json")
    meta.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[save] {meta}", flush=True)
    if frames:
        last = frames[-1]
        print(
            f"[OK] {anchor} t{last['time']} guiding={last['gnn_guiding']:.3f} "
            f"F0.5={last['gnn_f05']:.3f} f1={last['gnn_f1']:.3f}",
            flush=True,
        )
    return payload


def _write_summary_md(rows: list[dict], out_path: Path) -> None:
    lines = [
        "# Anchor clot deploy visualizations",
        "",
        "Three rows per anchor: **GT clot** | **model prediction** | **error** (FP red / FN blue by BFS hop from wall).",
        "",
        "| Anchor | PNG | Final F0.5 | Final guiding | Rollout (s) |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['anchor']} | [{row['anchor']}]({row['png']}) | "
            f"{row.get('final_f05', 0.0):.3f} | {row.get('final_guiding', 0.0):.3f} | "
            f"{row.get('ml_rollout_elapsed_s', 0.0):.1f} |"
        )
    lines.extend([
        "",
        "## Run locally",
        "",
        "```powershell",
        "python scripts/viz_species_gnn_deploy.py --all-anchors",
        "```",
    ])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[save] {out_path}", flush=True)


def _resolve_anchors(args, root: Path) -> list[str]:
    if args.legacy_six:
        return list(BIOCHEM_ANCHORS_6)
    if args.anchor.strip() and not args.all_anchors and not args.anchors.strip():
        return [args.anchor.strip()]
    if args.anchors.strip():
        return parse_biochem_train_anchors(args.anchors, all_anchors=False, root=root)
    if args.all_anchors:
        return discover_biochem_anchors(root)
    return [args.anchor.strip() or "patient007"]


def main() -> int:
    ap = argparse.ArgumentParser(description="GNN clot map deploy viz")
    ap.add_argument("--anchor", default="patient007", help="Single anchor (default when no batch flag)")
    ap.add_argument("--anchors", default="", help="Comma-separated anchor stems")
    ap.add_argument("--all-anchors", action="store_true", help="All patient*.pt on disk")
    ap.add_argument("--legacy-six", action="store_true", help="Legacy triangle6 anchors only")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--flow", default="kinematics", choices=("gt", "kinematics"))
    ap.add_argument("--max-frames", type=int, default=6)
    ap.add_argument("--scatter-size", type=float, default=3.0)
    ap.add_argument("--no-error-row", action="store_true")
    ap.add_argument("--out", default="", help="Output PNG (single anchor only)")
    args = ap.parse_args()

    device = require_cuda_device()
    root = get_project_root()
    manifest = load_deploy_manifest(args.manifest.strip() or None)
    manifest_name = Path(args.manifest).stem if args.manifest.strip() else "deploy"
    anchors = _resolve_anchors(args, root)

    print(f"[i] CUDA: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"[i] anchors={len(anchors)} flow={args.flow} manifest={manifest_name}", flush=True)

    summary_rows: list[dict] = []
    failures: list[str] = []
    bundle_cache: dict = {}
    for anchor in anchors:
        print(f"\n[i] --- {anchor} ---", flush=True)
        out_path = None
        if args.out.strip() and len(anchors) == 1:
            out_path = Path(args.out)
        try:
            payload = viz_anchor_deploy(
                anchor,
                manifest=manifest,
                device=device,
                flow_source=args.flow,
                manifest_name=manifest_name,
                max_frames=int(args.max_frames),
                scatter_size=float(args.scatter_size),
                include_error_row=not args.no_error_row,
                out=out_path,
                bundle_cache=bundle_cache,
            )
            summary_rows.append(payload)
        except Exception as exc:
            print(f"[ERR] {anchor}: {exc}", flush=True)
            failures.append(f"{anchor}: {exc}")

    if len(anchors) > 1:
        out_dir = _viz_out_dir()
        summary_json = out_dir / "deploy_all_anchors_summary.json"
        summary_json.write_text(
            json.dumps({"anchors_ok": summary_rows, "failures": failures}, indent=2),
            encoding="utf-8",
        )
        _write_summary_md(summary_rows, out_dir / "clot_visualizations.md")
        print(f"\n[OK] saved {len(summary_rows)}/{len(anchors)} -> {out_dir}", flush=True)
        if failures:
            print(f"[WARN] {len(failures)} failures", flush=True)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
