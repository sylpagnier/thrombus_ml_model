"""Sanity baselines for the physics wall model, and a diagnosis of where it fails.

Two jobs:

  A. BASELINES.  The physics gate scores 0.72/0.74/0.86 (train/dev/sealed) on the
     canonical wall-masked metric.  Before believing that, check what trivial predictors
     score on the same metric: all-wall, none, random-at-matched-rate, and the repo's
     own ``is_low_shear`` feature (which is what the previous stack actually saw).

  B. RESIDUAL.  Per-vessel, which gate fires, and does over/under-prediction track the
     slow/fast regime split (PHASE3_HANDOFF 1.4 / 10.4)?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.biochem_gnn.mat_growth_simple import (  # noqa: E402
    WALL_COHORT_V2_DEV, WALL_COHORT_V2_GENERALIZATION, WALL_COHORT_V2_TRAIN,
)
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.physics_wall_model import t0_flow_fields  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.core_physics.species_pushforward_continuous import resolve_deploy_eval_time_index  # noqa: E402
from src.evaluation.clot_relaxed_metrics import (  # noqa: E402
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
)

DIR = Path("data/processed/graphs_biochem_anchors")


def score(data, phi_pred, phys):
    t_eval = resolve_deploy_eval_time_index(int(data.y.shape[0]))
    gt = gt_clot_phi_at_time(data, t_eval, phys, device=torch.device("cpu")).reshape(-1)
    wall = data.mask_wall.reshape(-1).bool()
    m = compute_clot_relaxed_metrics(phi_pred.reshape(-1) * wall.float(),
                                     gt * wall.float(), data.edge_index, wall_mask=wall)
    o = metrics_to_deploy_prefix(m)
    return clot_score_from_deploy_dict(o), o, (gt * wall.float())


def main() -> int:
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    names = sorted(set(WALL_COHORT_V2_TRAIN) | set(WALL_COHORT_V2_DEV)
                   | set(WALL_COHORT_V2_GENERALIZATION))
    rng = np.random.default_rng(0)
    res = {k: [] for k in ("gate", "all_wall", "random", "repo_lowshear", "sep_only", "low_only")}
    rows = []
    for a in names:
        p = DIR / f"{a}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        wall = d.mask_wall.reshape(-1).bool().numpy()
        f = t0_flow_fields(d, bio, hops=3)
        preds = {
            "gate": (f.gate > 0) & wall,
            "all_wall": wall.copy(),
            "sep_only": (f.gate_sep > 0) & wall,
            "low_only": (f.gate_low > 0) & wall,
        }
        s_gate, m_gate, gt = score(d, torch.tensor(preds["gate"].astype(np.float32)), phys)
        rate = preds["gate"].sum() / max(wall.sum(), 1)
        r = rng.random(len(wall)) < rate
        preds["random"] = r & wall
        # repo's own is_low_shear feature, the input the previous stack actually saw
        from src.core_physics.clot_t0_extended_probe import build_feature_table_at_time
        tb = build_feature_table_at_time(d, 0, device=torch.device("cpu"), phys_cfg=phys, bio_cfg=bio)
        gam_repo = tb["gamma_si"][0].numpy()
        preds["repo_lowshear"] = (gam_repo < float(bio.lss)) & wall

        row = {"anchor": a}
        for k, pr in preds.items():
            s, m, _ = score(d, torch.tensor(pr.astype(np.float32)), phys)
            res[k].append((a, s))
            row[k] = s
        # regime + residual context
        u = d.y[0, :, 0].numpy(); v = d.y[0, :, 1].numpy()
        band_speed_q25 = float(np.percentile(np.hypot(u, v)[wall], 25))
        row.update(n_gt=int(gt.sum()), n_pred=int(preds["gate"].sum()),
                   ratio=preds["gate"].sum() / max(int(gt.sum()), 1),
                   frac_low=float(f.gate_low[wall].mean()),
                   frac_sep=float(f.gate_sep[wall].mean()),
                   q25=band_speed_q25,
                   relP=m_gate["deploy_clot_relaxed_prec"], relR=m_gate["deploy_clot_relaxed_rec"])
        rows.append(row)

    print("[A] canonical deploy_clot_score, mean over %d vessels" % len(rows))
    for k, v in res.items():
        by = {a: s for a, s in v}
        f = lambda names_: float(np.mean([by[a] for a in names_ if a in by]))
        print("   %-14s all %.4f | train %.4f  dev %.4f  sealed %.4f"
              % (k, float(np.mean([s for _, s in v])), f(WALL_COHORT_V2_TRAIN),
                 f(WALL_COHORT_V2_DEV), f(WALL_COHORT_V2_GENERALIZATION)))

    print("\n[B] residual structure (sorted by pred/gt ratio)")
    print("%12s %7s %6s %6s %7s %7s %7s %7s"
          % ("vessel", "score", "nPred", "nGT", "ratio", "fracLow", "fracSep", "q25speed"))
    for r in sorted(rows, key=lambda z: -z["ratio"]):
        print("%12s %7.4f %6d %6d %7.2f %7.3f %7.3f %8.4f"
              % (r["anchor"], r["gate"], r["n_pred"], r["n_gt"], r["ratio"],
                 r["frac_low"], r["frac_sep"], r["q25"]))
    rr = np.array([r["ratio"] for r in rows])
    for k in ("frac_low", "frac_sep", "q25"):
        x = np.array([r[k] for r in rows])
        print("   spearman(log ratio, %s) = %.3f"
              % (k, np.corrcoef(np.argsort(np.argsort(np.log(rr + 1e-9))).astype(float),
                                np.argsort(np.argsort(x)).astype(float))[0, 1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
