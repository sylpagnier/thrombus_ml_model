"""Mat-growth deploy clot ladder: GT | model pred | error (FP/FN).

Uses deploy-faithful env restoration (same as ``eval_mat_growth_simple``).
Default ckpt: promoted ``WC_mat_flow_dynamic`` canonical winner (legacy: ``W_mat_flow_stagnation``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.eval_mat_growth_simple import _apply_ckpt_recipe  # noqa: E402
from src.biochem_gnn.config import apply_deploy_env  # noqa: E402
from src.biochem_gnn.mat_growth_simple import apply_mat_growth_leg_env, leg_out_ckpt  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.clot_continuous_time import macro_tau_at_index  # noqa: E402
from src.core_physics.species_gnn_clot_rollout import (  # noqa: E402
    load_species_gnn_rollout_bundle,
    prepare_species_gnn_rollout_static,
    rollout_species_gnn_phi_trajectory,
)
from src.core_physics.species_gnn_ladder_viz import (  # noqa: E402
    ladder_viz_times,
    scatter_clot_error_panel,
)
from src.core_physics.species_pushforward_continuous import (  # noqa: E402
    clear_offwall_model_cache,
    compute_hop_distances,
    train_deploy_eval_flow_source,
)
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.evaluation.clot_relaxed_metrics import compute_clot_relaxed_metrics  # noqa: E402
from src.evaluation.viz_clot_phi_simple import _scatter_fullmesh_region  # noqa: E402
from src.evaluation.clot_timeline_metrics import clot_frame_metrics, summarize_clot_timeline  # noqa: E402
from src.evaluation.viz_clot_trigger import clot_trigger_viz_f1  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402

DEFAULT_LEG = "WC_mat_flow_dynamic"
ROW_GT = "Ground truth (GT)"
ROW_PRED = "Model prediction"
ROW_ERR = "Error (FP=red, FN=blue)"


def _viz_out_dir() -> Path:
    p = get_project_root() / "outputs/biochem/viz/mat_growth"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _hop_error_legend(fig) -> None:
    from matplotlib.patches import Patch

    from src.core_physics.species_gnn_ladder_viz import FN_COLORS, FP_COLORS

    legend_elements = []
    for hop in range(0, max(FP_COLORS.keys()) + 1):
        label = f"+{hop}: FP Hop {hop}" if hop > 0 else "+0: FP Wall (Hop 0)"
        legend_elements.append(Patch(facecolor=FP_COLORS[hop], label=label))
    for hop in range(0, max(FN_COLORS.keys()) + 1):
        label = f"-{hop}: FN Hop {hop}" if hop > 0 else "-0: FN Wall (Hop 0)"
        legend_elements.append(Patch(facecolor=FN_COLORS[hop], label=label))
    fig.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        ncol=1,
        fontsize=8,
        title="Error Hop Distance\n(-x: FN, +x: FP)",
        title_fontsize=9,
    )


def _scatter_clot_panel(
    ax,
    pos: np.ndarray,
    vals: np.ndarray,
    row_label: str,
    *,
    s: float,
) -> None:
    full_region = np.ones(vals.reshape(-1).shape[0], dtype=bool)
    _scatter_fullmesh_region(
        ax,
        pos,
        vals,
        full_region,
        row_label,
        cmap="bwr",
        vmin=0.0,
        vmax=1.0,
        s=s,
        layer_positive_on_top=True,
        positive_thresh=0.5,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Mat-growth clot ladder (GT | pred | error)")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--leg", default=DEFAULT_LEG, help="mat_growth_ladder leg code")
    ap.add_argument("--mat-leg", default="", help="Apply mat_growth_simple leg env (e.g. WC_v7_clot_phi_mse)")
    ap.add_argument("--ckpt", default="", help="override ckpt path")
    ap.add_argument("--offwall-ckpt", default="", help="Growth specialist for two-model compound rollout")
    ap.add_argument(
        "--two-model-route",
        default="",
        choices=("", "wall", "frontier", "growth"),
        help="Two-model routing when --offwall-ckpt is set (default: frontier)",
    )
    ap.add_argument("--two-model-frontier-hops", type=int, default=2)
    ap.add_argument("--arm-label", default="", help="Title tag (e.g. Arm_A_canonical)")
    ap.add_argument("--flow", default="kinematics", choices=("gt", "kinematics"))
    ap.add_argument("--max-frames", type=int, default=10)
    ap.add_argument("--scatter-size", type=float, default=3.0)
    ap.add_argument("--no-error-row", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = require_cuda_device()
    root = get_project_root()
    ckpt = Path(args.ckpt.strip()) if args.ckpt.strip() else root / leg_out_ckpt(args.leg.strip())
    if not ckpt.is_file():
        raise SystemExit(f"[ERR] missing ckpt: {ckpt}")

    clear_offwall_model_cache()
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    meta = dict(payload.get("meta") or {})
    _apply_ckpt_recipe(meta, label="mat_growth_simple")
    if args.mat_leg.strip():
        apply_mat_growth_leg_env(args.mat_leg.strip(), force=True)
    offwall_raw = args.offwall_ckpt.strip()
    if offwall_raw:
        offwall_path = Path(offwall_raw)
        if not offwall_path.is_absolute():
            offwall_path = root / offwall_path
        if not offwall_path.is_file():
            raise SystemExit(f"[ERR] missing --offwall-ckpt: {offwall_path}")
        route = args.two_model_route.strip() or "frontier"
        os.environ["SPECIES_TWO_MODEL_MODE"] = "1"
        os.environ["SPECIES_OFFWALL_MODEL_CKPT"] = str(offwall_path).replace("\\", "/")
        os.environ["SPECIES_TWO_MODEL_ROUTE"] = route
        os.environ["SPECIES_TWO_MODEL_FRONTIER_HOPS"] = str(int(args.two_model_frontier_hops))
        two_model_note = f"two-model route={route}"
    else:
        os.environ["SPECIES_TWO_MODEL_MODE"] = "0"
        os.environ.pop("SPECIES_OFFWALL_MODEL_CKPT", None)
        two_model_note = "single-model"
    flow_eval = train_deploy_eval_flow_source()
    apply_deploy_env(overrides={"T0_R4_FLOW_SOURCE": args.flow if args.flow != "kinematics" else flow_eval})

    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    data = torch.load(
        root / "data/processed/graphs_biochem_anchors" / f"{args.anchor}.pt",
        map_location=device,
        weights_only=False,
    )
    pos = data.x[:, :2].detach().cpu().numpy()
    n_nodes = int(data.num_nodes)
    from src.core_physics.clot_phi_simple import _wall_mask_from_data

    wall_mask_full = _wall_mask_from_data(data, device, n_nodes)
    hop_distances_np = compute_hop_distances(data.edge_index, wall_mask_full, n_nodes).detach().cpu().numpy()
    times = ladder_viz_times(int(data.y.shape[0]), max_frames=int(args.max_frames))
    mask = torch.ones(n_nodes, device=device, dtype=torch.bool)

    print(f"[i] CUDA: {torch.cuda.get_device_name(0)}", flush=True)
    arm_tag = args.arm_label.strip() or args.leg.strip() or DEFAULT_LEG
    print(f"[i] arm={arm_tag} ckpt={ckpt} flow={args.flow} {two_model_note}", flush=True)

    t0 = time.perf_counter()
    bundle = load_species_gnn_rollout_bundle(ckpt, device=device)
    if bundle is None:
        raise SystemExit(f"[ERR] could not load bundle: {ckpt}")
    static = prepare_species_gnn_rollout_static(data, device=device)
    phi_pred_traj = rollout_species_gnn_phi_trajectory(
        data,
        bundle,
        static,
        phys_cfg=phys,
        bio_cfg=bio,
        device=device,
        flow_source=args.flow,
    )
    print(f"[i] rollout {time.perf_counter() - t0:.1f}s", flush=True)

    row_labels = [ROW_GT, ROW_PRED]
    if not args.no_error_row:
        row_labels.append(ROW_ERR)

    fig, axes = plt.subplots(
        len(row_labels),
        len(times),
        figsize=(2.7 * len(times), 2.5 * len(row_labels)),
        squeeze=False,
    )
    leg_tag = arm_tag
    fig.suptitle(
        f"Mat growth clot -- {args.anchor} | {leg_tag} | {args.flow} flow | {two_model_note}",
        fontsize=11,
        y=1.01,
    )

    frames: list[dict] = []
    scatter_s = float(args.scatter_size)
    ei = data.edge_index.to(device=device)
    for j, t in enumerate(times):
        phi_gt = gt_clot_phi_at_time(data, int(t), phys, device)
        phi_pred = phi_pred_traj[int(t)]
        tau = float(macro_tau_at_index(data, int(t), bio_cfg=bio))
        fm = clot_frame_metrics(phi_pred, phi_gt, n_band=n_nodes)
        m = clot_trigger_viz_f1(phi_pred, phi_gt, mask)
        g = compute_clot_relaxed_metrics(phi_pred.reshape(-1), phi_gt.reshape(-1), ei)
        col_title = (
            f"t={t}  tau={tau:.2f}\nF1={m['clot_f1']:.2f}  "
            f"FP={int(fm['clot_fp'])} FN={int(fm['clot_fn'])}"
        )

        phi_gt_np = phi_gt.detach().cpu().numpy()
        phi_pred_np = phi_pred.detach().cpu().numpy()
        _scatter_clot_panel(
            axes[0, j], pos, phi_gt_np, row_labels[0] if j == 0 else "", s=scatter_s,
        )
        _scatter_clot_panel(
            axes[1, j], pos, phi_pred_np, row_labels[1] if j == 0 else "", s=scatter_s,
        )
        axes[1, j].set_title(col_title, fontsize=9, pad=5)
        err_counts: dict[str, int] = {}
        if not args.no_error_row:
            err_counts = scatter_clot_error_panel(
                axes[2, j],
                pos,
                phi_gt_np,
                phi_pred_np,
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
            "clot_f1": float(m["clot_f1"]),
            "clot_guiding": float(g["clot_guiding"]),
            "clot_f05": float(g["clot_relaxed_f05"]),
            **{k: float(fm[k]) for k in fm if k.startswith("clot_")},
            **{f"err_{k}": int(v) for k, v in err_counts.items()},
        })

    timeline_summary = summarize_clot_timeline(frames)
    if not args.no_error_row:
        _hop_error_legend(fig)
    fig.tight_layout(rect=[0, 0, 0.9, 0.98])
    if args.out.strip():
        out = Path(args.out)
    else:
        out = _viz_out_dir() / f"clot_ladder_{leg_tag}_{args.anchor}.png"
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out}", flush=True)

    meta_out = out.with_suffix(".json")
    meta_out.write_text(
        json.dumps({
            "anchor": args.anchor,
            "leg": leg_tag,
            "arm_label": arm_tag,
            "mat_leg": args.mat_leg.strip(),
            "two_model": two_model_note,
            "offwall_ckpt": offwall_raw,
            "ckpt": str(ckpt),
            "flow_source": args.flow,
            "rows": row_labels,
            "times": times,
            "frames": frames,
            "timeline_summary": timeline_summary,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"[save] {meta_out}", flush=True)
    if timeline_summary:
        print(
            f"[i] timeline medFP={timeline_summary.get('clot_fp_median', 0):.0f} "
            f"p90FP={timeline_summary.get('clot_fp_p90', 0):.0f} "
            f"medFN={timeline_summary.get('clot_fn_median', 0):.0f} "
            f"earlyFP={timeline_summary.get('clot_fp_early_mean', 0):.0f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
