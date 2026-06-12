"""Per-anchor val metrics for species continuous GNN (growth / state / clot ladder F1)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.species_gnn_clot_rollout import (
    load_species_gnn_rollout_bundle,
    prepare_species_gnn_rollout_static,
    rollout_species_gnn_phi_trajectory,
)
from src.evaluation.viz_clot_trigger import clot_trigger_viz_f1
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
from src.core_physics.species_pushforward_continuous import (
    BIOCHEM_ANCHORS_6,
    eval_continuous_window,
    filter_continuous_windows,
    iter_pushforward_windows,
    load_continuous_bundle,
    log_series_on_band,
    pushforward_unroll_steps,
    pushforward_step_stride,
)
from src.utils.paths import get_project_root
from src.config import VesselConfig


def _eval_anchor(
    anchor: str,
    *,
    ckpt: Path,
    device: torch.device,
    unroll: int,
    stride: int,
) -> dict:
    root = get_project_root()
    graph_path = root / VesselConfig(phase="biochem_anchors").graph_output_dir / f"{anchor}.pt"
    data = torch.load(graph_path, map_location="cpu", weights_only=False)
    bundle = load_continuous_bundle(ckpt, device=device, quiet=True)
    if bundle is None:
        raise FileNotFoundError(ckpt)
    static = prepare_species_gnn_rollout_static(data, device=device)
    node_idx = static.node_idx
    n_band = int(node_idx.numel())
    val_m = torch.ones(n_band, dtype=torch.bool, device=device)
    windows = filter_continuous_windows(
        iter_pushforward_windows(int(data.y.shape[0]), unroll=unroll, stride=stride),
        data,
        node_idx,
        device,
        min_delta_mag=1e-8,
    )
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    growth_f1: list[float] = []
    state_f1: list[float] = []
    for win in windows[:40]:
        series = log_series_on_band(data, win, device, node_idx)
        m = eval_continuous_window(
            bundle.model,
            base_feats=static.base_feats,
            edge_index=static.edge_index,
            log_series=series,
            mask=val_m,
            physics_ctx=None,
        )
        growth_f1.append(m["mean_growth_f1"])
        state_f1.append(m["final_state_f1"])

    clot_frames: list[dict] = []
    gnn_bundle = load_species_gnn_rollout_bundle(ckpt, device=device, quiet=True)
    if gnn_bundle is not None:
        phi_traj = rollout_species_gnn_phi_trajectory(
            data, gnn_bundle, static=static, phys_cfg=phys, bio_cfg=bio, device=device,
        )
        n_t = int(data.y.shape[0])
        mask = torch.ones(int(data.num_nodes), device=device, dtype=torch.bool)
        times = [0, 5, 11, 17, 23, 29, 35, 41, 47, 53]
        for t in times:
            if t >= n_t:
                continue
            phi_gt = gt_clot_phi_at_time(data, int(t), phys, device)
            m = clot_trigger_viz_f1(phi_traj[int(t)], phi_gt, mask)
            clot_frames.append({"time": t, "clot_f1": float(m["clot_f1"])})

    return {
        "anchor": anchor,
        "n_windows": len(windows),
        "mean_growth_f1": sum(growth_f1) / max(len(growth_f1), 1),
        "mean_state_f1": sum(state_f1) / max(len(state_f1), 1),
        "clot_frames": clot_frames,
        "clot_f1_t53": next((f["clot_f1"] for f in clot_frames if f["time"] == 53), 0.0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/biochem/species_snapshot_s26/best.pth")
    ap.add_argument("--anchors", default=",".join(BIOCHEM_ANCHORS_6))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(args.ckpt)
    if not ckpt.is_absolute():
        ckpt = get_project_root() / ckpt
    unroll = pushforward_unroll_steps()
    stride = pushforward_step_stride()
    rows: list[dict] = []
    for anc in [a.strip() for a in args.anchors.split(",") if a.strip()]:
        print(f"[i] eval {anc} ...", flush=True)
        rows.append(_eval_anchor(anc, ckpt=ckpt, device=device, unroll=unroll, stride=stride))
        r = rows[-1]
        print(
            f"  growth_f1={r['mean_growth_f1']:.3f} state_f1={r['mean_state_f1']:.3f} "
            f"clot_t53={r['clot_f1_t53']:.3f}",
            flush=True,
        )

    out_path = Path(args.out) if args.out else ckpt.parent / "multi_anchor_eval.json"
    if not out_path.is_absolute():
        out_path = get_project_root() / out_path
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
