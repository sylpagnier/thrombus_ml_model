"""Build the JSON payload for the Phase-6 wall+lumen time-lapse visualization.

Extends ``gen_temporal_viz_data.py`` (the Phase-3 standard) with the lumen arm. Two
things are genuinely new here, both worth stating plainly:

  1. WALL timing now comes from ``predict_wall_onset`` -- the shipped AP-closure-fixed
     ODE, not the old frozen-``ap`` flash. Nodes in the mask that never cross the ODE
     threshold (they arrived via graph growth, not ignition) get the vessel's median
     crossing time, exactly the convention ``scripts/eval_growth_count.py::ode_onset``
     uses for the growth-count metric.

  2. LUMEN (off-wall) timing DOES NOT EXIST in the shipped model -- ``grow_into_lumen``
     is a static geometric propagation rule with no time axis at all. What is plotted
     here is a diagnostic extension, not a shipped quantity: each admitted lumen node
     inherits the EARLIEST onset time among the wall/lumen neighbours that admitted it
     (a min-time BFS, same hop structure and admissibility as the shipped rule). Treat
     the lumen curve as illustrative of *when a node becomes reachable*, not a claim
     about physical thrombus-extension timing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.mls_gradient import node_positions  # noqa: E402
from src.core_physics.physics_lumen_model import (  # noqa: E402
    median_edge_length, speed_nd, speed_nd_pred, wall_normal_projection,
)
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.evaluation.clot_relaxed_metrics import (  # noqa: E402
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
)
from scripts.predict_wall_clot import LUMEN_HOPS, LUMEN_SPEED, predict_wall_clot, predict_wall_onset  # noqa: E402

DIR = Path("data/processed/graphs_biochem_anchors")
VESSELS = [
    ("patient012", "gt"),
    ("patient044", "gt"),
    ("patient042", "gt"),
    ("patient007", "pred"),
    ("patient032", "pred"),
]
N_FRAMES = 13
MAX_BG_POINTS = 1800


def adjacency(edge_index: np.ndarray, n: int) -> sp.csr_matrix:
    A = sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
                      shape=(n, n)).tocsr()
    return ((A + A.T) > 0).astype(np.int8)


def lumen_onset_bfs(seed_time: np.ndarray, committed: np.ndarray, admissible: np.ndarray,
                    edge_index: np.ndarray, n: int, hops: int) -> np.ndarray:
    """Min-time BFS: each admitted off-wall node gets its earliest-committing neighbour's time.

    ``committed`` is the PREDICTED wall-clot mask (``grow_into_lumen``'s seed), not the
    geometric wall mask -- only nodes actually predicted to commit can seed a neighbour.
    """
    src = np.concatenate([edge_index[0], edge_index[1]])
    dst = np.concatenate([edge_index[1], edge_index[0]])
    time = np.where(committed, seed_time, np.inf)
    known = committed.copy()
    for _ in range(max(int(hops), 0)):
        cand = admissible & ~known
        src_known = known[src]
        dst_cand = cand[dst]
        valid = src_known & dst_cand
        if not valid.any():
            break
        np.minimum.at(time, dst[valid], time[src[valid]])
        newly = cand & np.isfinite(time)
        if not newly.any():
            break
        known = known | newly
    return time  # inf where never reached


def wall_onset_filled(data, bio, flow) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(wall_mask, onset_filled, t) -- median-fill convention from eval_growth_count.py."""
    mask, onset, t = predict_wall_onset(data, bio, flow=flow)
    mask = np.asarray(mask).astype(bool)
    onset = np.asarray(onset)
    crossed = onset >= 0
    med = int(np.median(onset[crossed & mask])) if (crossed & mask).any() else 0
    filled = np.where(mask, np.where(onset >= 0, onset, med), -1)
    return mask, filled, np.asarray(t)


def main() -> None:
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    out = {}
    for anchor, flow in VESSELS:
        d = torch.load(DIR / f"{anchor}.pt", map_location="cpu", weights_only=False)
        pos = node_positions(d)
        wall = d.mask_wall.reshape(-1).bool().numpy()
        n = len(wall)
        ei = d.edge_index.numpy()
        interior = np.where(~wall)[0]
        stride = max(1, len(interior) // MAX_BG_POINTS)
        bg = interior[::stride]

        wall_mask, onset_idx, t = wall_onset_filled(d, bio, flow)
        onset_t = np.where(onset_idx >= 0, t[np.clip(onset_idx, 0, len(t) - 1)], -1.0)

        spd = speed_nd_pred(d) if flow == "pred" else speed_nd(d)
        admissible = (~wall) & (spd < LUMEN_SPEED)
        lumen_time = lumen_onset_bfs(onset_t, wall_mask, admissible, ei, n, LUMEN_HOPS)
        lumen_mask = np.isfinite(lumen_time) & ~wall

        full_mask, _ = predict_wall_clot(d, bio, flow=flow, lumen=True)
        full_mask = np.asarray(full_mask).astype(bool)
        assert np.array_equal(wall_mask | lumen_mask, full_mask), \
            "diagnostic lumen reachability must match the shipped grow_into_lumen mask"

        gt_hot = np.zeros((len(t), n), dtype=bool)
        for i in range(len(t)):
            gt_hot[i] = gt_clot_phi_at_time(d, i, phys, device=torch.device("cpu")).numpy() > 0.5

        model_wall_hot = (onset_t[None, :] >= 0) & (onset_t[None, :] <= t[:, None]) & wall_mask[None, :]
        model_lumen_hot = (lumen_time[None, :] <= t[:, None]) & lumen_mask[None, :]
        gt_wall_hot = gt_hot & wall[None, :]
        gt_lumen_hot = gt_hot & ~wall[None, :]

        lumen_render_set = np.where(lumen_mask | gt_lumen_hot.any(axis=0))[0]
        wall_idx = np.where(wall)[0]
        frame_idx = np.linspace(0, len(t) - 1, N_FRAMES).round().astype(int)

        # Wall-normal distance, for a colour gradient by depth into the lumen.  Normalised
        # by 1.5 median edge lengths -- checked against the actual render-set spread (most
        # off-wall clot sits within 2 edge lengths of the wall, physics_lumen_model.py's own
        # characterisation), so the gradient uses the visible range instead of clipping flat.
        dist_raw, _ = wall_normal_projection(pos, wall)
        h_edge = median_edge_length(pos, ei)
        dist_norm = np.clip(dist_raw / (1.5 * max(h_edge, 1e-9)), 0.0, 1.0)

        def pts(idx):
            return [[round(float(pos[i, 0]), 4), round(float(pos[i, 1]), 4)] for i in idx]

        # THE DEPLOY SCORE, ON WALL AND OFF WALL, AT EVERY TIMESTEP -- not just the final
        # mask.  Same canonical scoring function both times (compute_clot_relaxed_metrics ->
        # clot_score_from_deploy_dict); only which domain counts as pred/gt changes.
        ei_t = d.edge_index
        wall_f = torch.tensor(wall.astype(np.float32))
        off_f = torch.tensor((~wall).astype(np.float32))

        def domain_score(pred_hot, gt_hot_t, domain_f):
            pred_d = torch.tensor(pred_hot.astype(np.float32)) * domain_f
            gt_d = torch.tensor(gt_hot_t.astype(np.float32)) * domain_f
            m = compute_clot_relaxed_metrics(pred_d, gt_d, ei_t, wall_mask=torch.tensor(wall))
            return clot_score_from_deploy_dict(metrics_to_deploy_prefix(m))

        score_wall, score_offwall = [], []
        for i in range(len(t)):
            model_hot_i = model_wall_hot[i] | model_lumen_hot[i]
            score_wall.append(domain_score(model_hot_i, gt_hot[i], wall_f))
            score_offwall.append(domain_score(model_hot_i, gt_hot[i], off_f))

        out[anchor] = {
            "flow": flow,
            "t_final": float(t[-1]),
            "n_wall": int(wall.sum()),
            "bg": pts(bg),
            "wall_pos": pts(wall_idx),
            "lumen_pos": pts(lumen_render_set),
            "lumen_dist": [round(float(x), 3) for x in dist_norm[lumen_render_set]],
            "frame_t": [round(float(t[i]), 1) for i in frame_idx],
            "frame_gt_wall": [[bool(x) for x in gt_wall_hot[i][wall_idx]] for i in frame_idx],
            "frame_model_wall": [[bool(x) for x in model_wall_hot[i][wall_idx]] for i in frame_idx],
            "frame_gt_lumen": [[bool(x) for x in gt_lumen_hot[i][lumen_render_set]] for i in frame_idx],
            "frame_model_lumen": [[bool(x) for x in model_lumen_hot[i][lumen_render_set]] for i in frame_idx],
            "frame_n_gt_wall": [int(gt_wall_hot[i].sum()) for i in frame_idx],
            "frame_n_model_wall": [int(model_wall_hot[i].sum()) for i in frame_idx],
            "frame_n_gt_lumen": [int(gt_lumen_hot[i].sum()) for i in frame_idx],
            "frame_n_model_lumen": [int(model_lumen_hot[i].sum()) for i in frame_idx],
            "score_t": [round(float(x), 1) for x in t],
            "score_wall": [round(float(x), 4) for x in score_wall],
            "score_offwall": [round(float(x), 4) for x in score_offwall],
        }
        print(f"{anchor} [{flow}]: wall={wall.sum()} lumen_render={len(lumen_render_set)}  "
              f"final wall score={score_wall[-1]:.3f}  final offwall score={score_offwall[-1]:.3f}")

    out_path = Path("outputs/offwall_temporal_data.json")
    out_path.write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote {out_path}  ({out_path.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
