"""Build the JSON payload for the clot_gnn_v1 time-lapse visualization (VIZ_STANDARD).

clot_gnn_v1 predicts ONE static per-node score from t=0 GT flow + geometry -- there is no
per-node onset time, unlike the physics AP-closure arm. So the "Model" panel is the SAME
mask at every frame; only the "Ground truth" panel actually grows. That is stated in-page,
per docs/VIZ_STANDARD.md point 6 (never let a diagnostic/static quantity read as dynamic).

Only FIT/DEV vessels are used -- clot_gnn_v1's SEALED set (patient042/043 and, by exclusion
from fit_anchors/dev_anchors, everything else) is never opened here (docs/PHASE9_ML.md 10.1).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.clot_ml.data import attach_physics, load_cache
from src.clot_ml.locked import load_ensemble, predict_scores
from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.mls_gradient import node_positions
from src.core_physics.physics_lumen_model import median_edge_length, wall_normal_projection
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
from src.evaluation.clot_relaxed_metrics import (
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
)

DIR = Path("data/processed/graphs_biochem_anchors")
# clot_gnn_v1's own split (data/reference/clot_gnn_locked.json manifest). DEV first and
# complete (all 3) -- generalization is the number that matters; FIT vessels are trained
# on and shown for contrast only. Never add a SEALED vessel here (042/043 + everything
# outside fit_anchors/dev_anchors).
DEV_ANCHORS = ["patient044", "patient041", "patient040"]
FIT_ANCHORS = ["patient012", "patient032"]
VESSELS = DEV_ANCHORS + FIT_ANCHORS
SPLIT_OF = {a: "dev" for a in DEV_ANCHORS} | {a: "fit" for a in FIT_ANCHORS}
SEALED_GUARD = {"patient042", "patient043"}
N_FRAMES = 13
MAX_BG_POINTS = 1800
# FIT-tuned thresholds from scripts/compare_gnn_vs_physics.py (grid sweep on FIT only).
T_WALL, T_OFF = 0.740, 0.940


def main() -> None:
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    cache = attach_physics(load_cache("gt"))
    ens = load_ensemble()

    out = {}
    for anchor in VESSELS:
        S = cache[anchor]
        assert anchor not in SEALED_GUARD, "clot_gnn_v1 SEALED, do not open"
        assert anchor in SPLIT_OF, f"{anchor} not declared FIT or DEV -- classify before viz'ing"
        d = torch.load(DIR / f"{anchor}.pt", map_location="cpu", weights_only=False)
        pos = node_positions(d)
        wall = d.mask_wall.reshape(-1).bool().numpy()
        assert np.array_equal(wall, S["wall"].astype(bool)), "cache/pack node order mismatch"
        n = len(wall)
        ei = d.edge_index.numpy()
        interior = np.where(~wall)[0]
        stride = max(1, len(interior) // MAX_BG_POINTS)
        bg = interior[::stride]

        score = predict_scores(ens, S)
        gnn_wall = (score >= T_WALL) & wall
        gnn_off = (score >= T_OFF) & ~wall
        gnn_mask = gnn_wall | gnn_off

        t = d.t.reshape(-1).detach().cpu().numpy().astype(np.float64)
        gt_hot = np.zeros((len(t), n), dtype=bool)
        for i in range(len(t)):
            gt_hot[i] = gt_clot_phi_at_time(d, i, phys, device=torch.device("cpu")).numpy() > 0.5
        gt_wall_hot = gt_hot & wall[None, :]
        gt_lumen_hot = gt_hot & ~wall[None, :]

        dist_raw, _ = wall_normal_projection(pos, wall)
        h_edge = median_edge_length(pos, ei)
        dist_norm = np.clip(dist_raw / (1.5 * max(h_edge, 1e-9)), 0.0, 1.0)

        lumen_render_set = np.where(gnn_off | gt_lumen_hot.any(axis=0))[0]
        wall_idx = np.where(wall)[0]
        frame_idx = np.linspace(0, len(t) - 1, N_FRAMES).round().astype(int)

        def pts(idx):
            return [[round(float(pos[i, 0]), 4), round(float(pos[i, 1]), 4)] for i in idx]

        ei_t = d.edge_index
        wall_f = torch.tensor(wall.astype(np.float32))
        off_f = torch.tensor((~wall).astype(np.float32))
        gnn_mask_f = torch.tensor(gnn_mask.astype(np.float32))

        def domain_score(gt_hot_t, domain_f):
            gt_d = torch.tensor(gt_hot_t.astype(np.float32)) * domain_f
            pred_d = gnn_mask_f * domain_f
            m = compute_clot_relaxed_metrics(pred_d, gt_d, ei_t, wall_mask=torch.tensor(wall))
            return clot_score_from_deploy_dict(metrics_to_deploy_prefix(m))

        score_wall = [domain_score(gt_hot[i], wall_f) for i in range(len(t))]
        score_offwall = [domain_score(gt_hot[i], off_f) for i in range(len(t))]

        gnn_wall_frame = [bool(x) for x in gnn_wall[wall_idx]]
        gnn_lumen_frame = [bool(x) for x in gnn_off[lumen_render_set]]

        out[anchor] = {
            "flow": "gt",
            "split": SPLIT_OF[anchor],
            "t_final": float(t[-1]),
            "n_wall": int(wall.sum()),
            "bg": pts(bg),
            "wall_pos": pts(wall_idx),
            "lumen_pos": pts(lumen_render_set),
            "lumen_dist": [round(float(x), 3) for x in dist_norm[lumen_render_set]],
            "frame_t": [round(float(t[i]), 1) for i in frame_idx],
            "frame_gt_wall": [[bool(x) for x in gt_wall_hot[i][wall_idx]] for i in frame_idx],
            "frame_model_wall": [gnn_wall_frame for _ in frame_idx],
            "frame_gt_lumen": [[bool(x) for x in gt_lumen_hot[i][lumen_render_set]] for i in frame_idx],
            "frame_model_lumen": [gnn_lumen_frame for _ in frame_idx],
            "score_t": [round(float(x), 1) for x in t],
            "score_wall": [round(float(x), 4) for x in score_wall],
            "score_offwall": [round(float(x), 4) for x in score_offwall],
        }
        print(f"{anchor}: wall={wall.sum()} lumen_render={len(lumen_render_set)}  "
              f"GNN mask wall={int(gnn_wall.sum())} off={int(gnn_off.sum())}  "
              f"final wall score={score_wall[-1]:.3f}  final offwall score={score_offwall[-1]:.3f}")

    out_path = Path("outputs/gnn_temporal_data.json")
    out_path.write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote {out_path}  ({out_path.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
