"""Train and lock clot_gnn_v2 -- the geometry-stratified-validated ensemble.

v1 (`promote_clot_gnn.py`) was fitted on the OLD FIT split: 16 baseline vessels, zero
aneurysms, zero stenoses (docs/PHASE9_ML.md 11.1 -- that split was confounded with geometry
class). v2 uses the three configurations validated by geometry-stratified 5-fold CV
(docs/PHASE9_ML.md 11.2: out-of-fold ALL wall 0.9198 / off 0.7270, meeting both targets
including on the priority class), and fits the SHIPPED weights on the full 19-vessel
eligible pool -- FIT + DEV together -- so the deployed model has actually seen an aneurysm
and two stenoses, not zero.

    outputs/clot_ml/locked/clot_gnn_v2/member_<cfg>_s<seed>.pth
    outputs/clot_ml/locked/clot_gnn_v2/manifest.json
    data/reference/clot_gnn_locked.json          <- repointed to v2; v1 files untouched

    python scripts/promote_clot_gnn_v2.py --name clot_gnn_v2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.clot_ml.data import attach_physics, load_cache  # noqa: E402
from src.clot_ml.geometry_splits import classes_for, eligible_pool, is_priority  # noqa: E402
from src.clot_ml.gnn import ClotGNN  # noqa: E402
from train_clot_gnn import train_one  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"

# The three configs behind cv5a / cv5b / cv5c (docs/PHASE9_ML.md 11), 3 seeds each.
BASE = dict(epochs=80, dim=64, layers=4, drop=0.1, lr=3e-3, wd=1e-4, pos_weight=30.0,
            reg_w=1.0, metric_w=2.0, metric_start=0.3, off_mult=1.0)
MEMBERS = {
    "rec3": dict(rounds=3, seeds=3),
    "rec5": dict(rounds=5, seeds=3),
    "rec3o": dict(rounds=3, seeds=3, off_mult=2.5),
}

# Out-of-fold numbers from the CV run that selected these configs (measured, not refit here
# -- refitting on the full pool has no held-out vessels left to score against).
CV_SCORES_OUT_OF_FOLD = dict(
    legacy=dict(
        all=dict(wall=0.9111, off=0.7039),
        baseline=dict(wall=0.9108, off=0.6670),
        priority=dict(wall=0.9129, off=0.8270),
        aneurysm=dict(wall=0.9749, off=0.8417, n=1),
        stenosis=dict(wall=0.8820, off=0.8196, n=2)),
    severity=dict(
        all=dict(wall=0.9198, off=0.7270),
        baseline=dict(wall=0.9202, off=0.6935),
        priority=dict(wall=0.9177, off=0.8388),
        aneurysm=dict(wall=0.9762, off=0.8417, n=1),
        stenosis=dict(wall=0.8885, off=0.8373, n=2)),
    physics_baseline=dict(
        legacy=dict(all=dict(wall=0.8683, off=0.4061),
                    baseline=dict(wall=0.8617, off=0.3743),
                    priority=dict(wall=0.9036, off=0.5122)),
        severity=dict(all=dict(wall=0.8766, off=0.4141),
                      baseline=dict(wall=0.8709, off=0.3819),
                      priority=dict(wall=0.9067, off=0.5214))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="clot_gnn_v2")
    ap.add_argument("--flow", default="gt")
    args = ap.parse_args()

    dev_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = attach_physics(load_cache(args.flow))
    pool = [a for a in eligible_pool() if a in cache]
    classes = classes_for(pool, PACKS)
    pool = [a for a in pool if a in classes]
    prio = [a for a in pool if is_priority(classes[a])]
    print("[i] training pool n=%d, priority=%d (%s)" % (len(pool), len(prio), ", ".join(prio)),
          flush=True)

    out = REPO / "outputs/clot_ml/locked" / args.name
    out.mkdir(parents=True, exist_ok=True)

    Xall = np.concatenate([cache[a]["X"] for a in pool])
    mu, sd = Xall.mean(0), Xall.std(0)
    sd[sd < 1e-6] = 1.0
    cols = [str(c) for c in cache[pool[0]]["cols"]]
    members, t0 = [], time.time()

    for cname, over in MEMBERS.items():
        cfg = dict(BASE)
        cfg.update({k: v for k, v in over.items() if k != "seeds"})
        n_seeds = int(over.get("seeds", 3))
        for s in range(n_seeds):
            fn = "member_%s_s%d.pth" % (cname, s)
            if (out / fn).exists():
                members.append(dict(file=fn, config=cname, seed=s, **cfg))
                print("   kept  %-24s" % fn, flush=True)
                continue
            ns = SimpleNamespace(**cfg, seeds=1)
            predict = train_one(pool, cache, ns, dev_t, seed=s)
            model = predict.model
            assert isinstance(model, ClotGNN), type(model)
            torch.save(dict(state_dict=model.state_dict(), cfg=cfg, seed=s,
                            in_dim=model.enc[0].in_features - model.extra_dim,
                            extra_dim=model.extra_dim), out / fn)
            members.append(dict(file=fn, config=cname, seed=s, **cfg))
            print("   saved %-24s (%.0fs)" % (fn, time.time() - t0), flush=True)

    np.savez_compressed(out / "feature_norm.npz", mu=mu, sd=sd, cols=np.array(cols))
    manifest = dict(
        name=args.name,
        promoted_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        description=(
            "Physics-informed recurrent clot GNN, PHASE9/10. Ensemble of 9 members over 3 "
            "configurations (docs/PHASE9_ML.md 11), fitted on the FULL 19-vessel eligible "
            "pool -- FIT+DEV together, geometry-stratified: includes 1 aneurysm and 2 "
            "stenoses in training, unlike v1 which trained on baseline vessels only. "
            "DEV and SEALED naming is now retired for this pool; SEALED (8 vessels) never "
            "seen. Scores below are OUT-OF-FOLD from the 5-fold CV that selected these "
            "configs, not from these exact weights (there is no held-out vessel left once "
            "the shipped model trains on the whole pool)."),
        docs="docs/PHASE9_ML.md",
        supersedes="clot_gnn_v1",
        flow=args.flow,
        training_pool=list(pool), priority_anchors=list(prio),
        geometry_classes={a: classes[a] for a in pool},
        readout="thresh, thresholds tuned per domain, geometry-stratified 5-fold",
        scores_out_of_fold_cv=CV_SCORES_OUT_OF_FOLD,
        n_members=len(members), members=members,
        feature_norm="feature_norm.npz", n_features=len(cols))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    ptr = REPO / "data/reference/clot_gnn_locked.json"
    ptr.write_text(json.dumps(dict(
        name=args.name, path=str(out.relative_to(REPO)).replace("\\", "/"),
        manifest=str((out / "manifest.json").relative_to(REPO)).replace("\\", "/"),
        promoted_at=manifest["promoted_at"], docs="docs/PHASE9_ML.md",
        supersedes="clot_gnn_v1",
        scores_out_of_fold_cv=CV_SCORES_OUT_OF_FOLD), indent=2))
    print("locked %d members -> %s\npointer -> %s (now clot_gnn_v2)  (%.0fs)"
          % (len(members), out, ptr, time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
