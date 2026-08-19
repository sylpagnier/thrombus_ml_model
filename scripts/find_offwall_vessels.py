"""Rank cohort vessels by off-wall (lumen) GT clot fraction, to pick viz candidates."""
from __future__ import annotations

import sys
import glob
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.species_pushforward_continuous import resolve_deploy_eval_time_index  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.biochem_gnn.mat_growth_simple import (  # noqa: E402
    WALL_COHORT_V2_GENERALIZATION, WALL_COHORT_V2_TRAIN,
)

DIR = Path("data/processed/graphs_biochem_anchors")


def main() -> int:
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    names = sorted(set(WALL_COHORT_V2_TRAIN) | set(WALL_COHORT_V2_GENERALIZATION))
    rows = []
    for a in names:
        p = DIR / f"{a}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        if int(d.y.shape[0]) < 150:
            continue
        wall = d.mask_wall.reshape(-1).bool().numpy()
        t_eval = resolve_deploy_eval_time_index(int(d.y.shape[0]))
        gt = (gt_clot_phi_at_time(d, t_eval, phys, device=torch.device("cpu"))
              .reshape(-1).numpy() > 0.5)
        n_gt = int(gt.sum())
        n_off = int((gt & ~wall).sum())
        has_pred = getattr(d, "u0_pred", None) is not None
        split = "sealed" if a in WALL_COHORT_V2_GENERALIZATION else "train"
        rows.append((a, split, n_gt, n_off, n_off / max(n_gt, 1), has_pred))

    rows.sort(key=lambda r: -r[4])
    print("%12s %7s %6s %6s %8s %6s" % ("vessel", "split", "nGT", "nOff", "offFrac", "u0pred"))
    for a, split, n_gt, n_off, frac, has_pred in rows:
        print("%12s %7s %6d %6d %7.1f%% %6s" % (a, split, n_gt, n_off, frac * 100, has_pred))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
