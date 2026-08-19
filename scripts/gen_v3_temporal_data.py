"""Build the JSON payload for the clot_gnn_v3 time-lapse visualization (VIZ_STANDARD).

Unlike v1/v2, v3 predicts P(node is clot at time t) DIRECTLY -- time is a model input,
not a static mask held constant or a diagnostic onset extrapolation. Both windows in this
viz genuinely animate from the model's own per-timestep output.

HONESTY NOTE, read before trusting any number here. Per docs/PHASE9_ML.md 14: "There is no
vessel left to score v3 against out-of-fold -- like v2, it trains on the whole pool by
design." Every vessel this script can reach (the 19-vessel training_pool) is IN-SAMPLE for
the shipped model. The only genuine generalization estimate is the geometry-stratified
5-fold CV that SELECTED this design (13.9 / manifest.json scores_out_of_fold_cv) --
reproduced in the artifact's comparison table, not re-derived here. SEALED
(patient042/043 + everything outside training_pool) is never opened.
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
from src.clot_ml.locked import load_default, predict_default_series
from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.mls_gradient import node_positions
from src.core_physics.physics_lumen_model import median_edge_length, wall_normal_projection
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
from src.evaluation.clot_relaxed_metrics import (
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
)

DIR = Path("data/processed/graphs_biochem_anchors")
# All in training_pool (data/reference/clot_gnn_locked.json) -- SEALED never opened.
PRIORITY_ANCHORS = ["patient040", "patient041", "patient044"]   # aneurysm, stenosis, stenosis
BASELINE_ANCHORS = ["patient012", "patient032"]
VESSELS = PRIORITY_ANCHORS + BASELINE_ANCHORS
CLASS_OF = {"patient040": "aneurysm", "patient041": "stenosis", "patient044": "stenosis",
           "patient012": "baseline", "patient032": "baseline"}
SEALED_GUARD = {"patient042", "patient043", "patient001"}
N_FRAMES = 13
MAX_BG_POINTS = 1800


def main() -> None:
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    cache = attach_physics(load_cache("gt"))
    bundle, kind = load_default()
    assert kind == "temporal_v3", f"expected clot_gnn_v3 shipped, got {kind}"

    out = {}
    for anchor in VESSELS:
        assert anchor not in SEALED_GUARD, "SEALED, do not open"
        S = cache[anchor]
        d = torch.load(DIR / f"{anchor}.pt", map_location="cpu", weights_only=False)
        pos = node_positions(d)
        wall = d.mask_wall.reshape(-1).bool().numpy()
        n = len(wall)
        ei = d.edge_index.numpy()
        interior = np.where(~wall)[0]
        stride = max(1, len(interior) // MAX_BG_POINTS)
        bg = interior[::stride]

        t = d.t.reshape(-1).detach().cpu().numpy().astype(np.float64)
        times = list(range(len(t)))
        res = predict_default_series(bundle, kind, d, times, flow="gt", sample=S)
        series = res["series"]                                   # {ti: bool mask [N]}
        model_hot = np.stack([series[i] for i in range(len(t))], axis=0)  # [T, N]

        gt_hot = np.zeros((len(t), n), dtype=bool)
        for i in range(len(t)):
            gt_hot[i] = gt_clot_phi_at_time(d, i, phys, device=torch.device("cpu")).numpy() > 0.5

        dist_raw, _ = wall_normal_projection(pos, wall)
        h_edge = median_edge_length(pos, ei)
        dist_norm = np.clip(dist_raw / (1.5 * max(h_edge, 1e-9)), 0.0, 1.0)

        model_final = model_hot[-1]
        gt_ever = gt_hot.any(axis=0)
        lumen_render_set = np.where((model_final | gt_ever) & ~wall)[0]
        wall_idx = np.where(wall)[0]
        frame_idx = np.linspace(0, len(t) - 1, N_FRAMES).round().astype(int)

        def pts(idx):
            return [[round(float(pos[i, 0]), 4), round(float(pos[i, 1]), 4)] for i in idx]

        ei_t = d.edge_index
        wall_f = torch.tensor(wall.astype(np.float32))
        off_f = torch.tensor((~wall).astype(np.float32))

        def domain_score(pred_hot, gt_hot_t, domain_f):
            pred_d = torch.tensor(pred_hot.astype(np.float32)) * domain_f
            gt_d = torch.tensor(gt_hot_t.astype(np.float32)) * domain_f
            m = compute_clot_relaxed_metrics(pred_d, gt_d, ei_t, wall_mask=torch.tensor(wall))
            return clot_score_from_deploy_dict(metrics_to_deploy_prefix(m))

        score_wall = [domain_score(model_hot[i], gt_hot[i], wall_f) for i in range(len(t))]
        score_offwall = [domain_score(model_hot[i], gt_hot[i], off_f) for i in range(len(t))]

        out[anchor] = {
            "flow": "gt",
            "geom_class": CLASS_OF[anchor],
            "t_final": float(t[-1]),
            "n_wall": int(wall.sum()),
            "bg": pts(bg),
            "wall_pos": pts(wall_idx),
            "lumen_pos": pts(lumen_render_set),
            "lumen_dist": [round(float(x), 3) for x in dist_norm[lumen_render_set]],
            "frame_t": [round(float(t[i]), 1) for i in frame_idx],
            "frame_gt_wall": [[bool(x) for x in (gt_hot[i] & wall)[wall_idx]] for i in frame_idx],
            "frame_model_wall": [[bool(x) for x in (model_hot[i] & wall)[wall_idx]] for i in frame_idx],
            "frame_gt_lumen": [[bool(x) for x in (gt_hot[i] & ~wall)[lumen_render_set]] for i in frame_idx],
            "frame_model_lumen": [[bool(x) for x in (model_hot[i] & ~wall)[lumen_render_set]] for i in frame_idx],
            "score_t": [round(float(x), 1) for x in t],
            "score_wall": [round(float(x), 4) for x in score_wall],
            "score_offwall": [round(float(x), 4) for x in score_offwall],
        }
        print(f"{anchor} [{CLASS_OF[anchor]}]: wall={wall.sum()} lumen_render={len(lumen_render_set)}  "
              f"final wall score={score_wall[-1]:.3f}  final offwall score={score_offwall[-1]:.3f}")

    out_path = Path("outputs/v3_temporal_data.json")
    out_path.write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote {out_path}  ({out_path.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
