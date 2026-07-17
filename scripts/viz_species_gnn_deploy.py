"""Deploy stack viz: GT clot | GNN clot map (+ nucleation E(t) overlay).

Uses ``species_gnn_deploy_env`` (species ckpt, beta, per-anchor overrides).
Deploy flow is pred kinematics by default.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

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


def main() -> int:
    ap = argparse.ArgumentParser(description="GNN clot map deploy viz")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--flow", default="kinematics", choices=("gt", "kinematics"))
    ap.add_argument("--max-frames", type=int, default=10)
    ap.add_argument("--scatter-size", type=float, default=3.0)
    ap.add_argument("--no-error-row", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = require_cuda_device()
    root = get_project_root()
    manifest = load_deploy_manifest(args.manifest.strip() or None)
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
    from src.core_physics.species_pushforward_continuous import compute_hop_distances
    wall_mask_full = _wall_mask_from_data(data, device, n_nodes)
    hop_distances_gpu = compute_hop_distances(data.edge_index, wall_mask_full, n_nodes)
    hop_distances_np = hop_distances_gpu.detach().cpu().numpy()
    
    times = ladder_viz_times(int(data.y.shape[0]), max_frames=int(args.max_frames))
    mask = torch.ones(n_nodes, device=device, dtype=torch.bool)

    ckpt_pick = species_ckpt_for_anchor(args.anchor, manifest, prefer_loao=True)
    beta_ov = (manifest.get("beta_overrides") or {}).get(args.anchor, "")

    print(f"[i] CUDA: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"[i] anchor={args.anchor} flow={args.flow} ckpt={ckpt_pick}", flush=True)
    if beta_ov:
        print(f"[i] beta_override={beta_ov}", flush=True)

    t0 = time.perf_counter()
    with species_gnn_deploy_env(
        manifest,
        overrides={"T0_R4_FLOW_SOURCE": args.flow},
        anchor=args.anchor,
        prefer_loao=True,
    ):
        ckpt = Path(species_gnn_rollout_ckpt())
        bundle = load_species_gnn_rollout_bundle(ckpt, device=device)
        if bundle is None:
            raise SystemExit(f"[ERR] missing ckpt: {ckpt}")
        static = prepare_species_gnn_rollout_static(data, device=device)
        phi_gnn = rollout_species_gnn_phi_trajectory(
            data, bundle, static, phys_cfg=phys, bio_cfg=bio, device=device,
            flow_source=args.flow,
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

    manifest_name = Path(args.manifest).stem if args.manifest.strip() else "deploy"
    row_labels = [ROW_GT, ROW_PRED]
    if not args.no_error_row:
        row_labels.append(ROW_ERR)
    fig, axes = plt.subplots(
        len(row_labels), len(times),
        figsize=(2.7 * len(times), 2.5 * len(row_labels)),
        squeeze=False,
    )
    fig.suptitle(
        f"biochem GNN deploy -- {_patient_label(args.anchor)} | {args.flow} flow | {manifest_name}\n{speed_note}",
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
        scatter_s = float(args.scatter_size)
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
        if not args.no_error_row:
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

    fig.tight_layout()
    if args.out.strip():
        out = Path(args.out)
    else:
        out = _viz_out_dir() / f"deploy_{args.anchor}_{args.flow}.png"
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out}", flush=True)

    meta = out.with_suffix(".json")
    meta.write_text(
        json.dumps({
            "anchor": args.anchor,
            "patient_label": _patient_label(args.anchor),
            "flow_source": args.flow,
            "species_ckpt": str(ckpt_pick),
            "beta_override": str(beta_ov or beta_used),
            "manifest": str(args.manifest),
            "rows": row_labels,
            "times": times,
            "frames": frames,
            "final_guiding": frames[-1]["gnn_guiding"] if frames else None,
            "final_f05": frames[-1]["gnn_f05"] if frames else None,
            "final_f1": frames[-1]["gnn_f1"] if frames else None,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"[save] {meta}", flush=True)
    if frames:
        last = frames[-1]
        print(
            f"[OK] t{last['time']} guiding={last['gnn_guiding']:.3f} "
            f"F0.5={last['gnn_f05']:.3f} f1={last['gnn_f1']:.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
