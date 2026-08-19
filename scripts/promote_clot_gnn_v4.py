"""Train and lock `clot_gnn_v4` -- the advective-transport ensemble, strict-protocol validated.

v4 differs from `clot_gnn_v2`/`v3` in exactly one thing: the feature block.  It adds the 13
channels of `src/clot_ml/features_v4.py` -- COMSOL's own advection operator solved on the
mesh (`src/clot_ml/transport.py`), plus the indicator-gate physics variant.  Architecture,
configurations, seeds and readout family are unchanged, so the comparison in
`docs/PHASE10_V4.md` is a clean feature ablation.

Validated strictly-nested (every readout scalar selected on out-of-fold scores of vessels
outside the held-out fold -- `scripts/eval_strict.py`, `scripts/eval_strict_temporal.py`):

                  mean wall   mean off   FIN wall   FIN off
    v3 (cv5a,b,c)    0.8687     0.6389     0.9014     0.7011
    v4 (v5a,b,c)     0.8750     0.6833     0.9176     0.7359

Read `docs/PHASE10_V4.md` 2 before quoting any of it: the cohort noise floor is +-0.024 wall
and +-0.091 off-wall, so what supports v4 is the direction being consistent on all four
metrics and both domains, not the size of any one of them.

As with v2/v3 the SHIPPED weights train on the whole 19-vessel eligible pool, so there is no
held-out vessel left to score them against; the manifest carries the out-of-fold CV numbers
that SELECTED this design, which is the honest generalisation estimate.  SEALED never seen.

    python scripts/promote_clot_gnn_v4.py --name clot_gnn_v4
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
from src.clot_ml.features_v4 import V4_CHANNELS  # noqa: E402
from src.clot_ml.geometry_splits import classes_for, eligible_pool, is_priority  # noqa: E402
from src.clot_ml.gnn import ClotGNN  # noqa: E402
from train_clot_gnn import train_one  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"

# The three configurations behind the v5a / v5b / v5c CV tags, 3 seeds each.
BASE = dict(epochs=80, dim=64, layers=4, drop=0.1, lr=3e-3, wd=1e-4, pos_weight=30.0,
            reg_w=1.0, metric_w=2.0, metric_start=0.3, off_mult=1.0, metric="legacy")
MEMBERS = {
    "v5a": dict(rounds=3, seeds=3),
    "v5b": dict(rounds=5, seeds=3),
    "v5c": dict(rounds=3, seeds=3, off_mult=2.5),
}

#: Strictly-nested, out-of-fold, severity metric (docs/PHASE10_V4.md).  NOT from these
#: weights -- these train on the whole pool.
STRICT_CV = dict(
    protocol=("geometry-stratified 5-fold; every readout scalar selected on the OUT-OF-FOLD "
              "scores of vessels outside the held-out fold (scripts/eval_strict.py)"),
    v4=dict(final=dict(all=dict(wall=0.9176, off=0.7359),
                       baseline=dict(wall=0.9186, off=0.6973),
                       priority=dict(wall=0.9123, off=0.8644)),
            mean_over_time=dict(all=dict(wall=0.8750, off=0.6833),
                                baseline=dict(wall=0.8741, off=0.6532),
                                priority=dict(wall=0.8798, off=0.7836)),
            oracle_timing_same_set=dict(all=dict(wall=0.9662, off=0.8910))),
    v3_same_protocol=dict(final=dict(all=dict(wall=0.9014, off=0.7011)),
                          mean_over_time=dict(all=dict(wall=0.8687, off=0.6389))),
    physics_backbone=dict(final=dict(all=dict(wall=0.8766, off=0.4141))),
    noise_floor=dict(wall=0.024, off=0.091,
                     note=("config spread of one arm; docs/PHASE10_V4.md 2. A 0.01-0.03 "
                           "cohort-mean difference on these 19 vessels is not a result.")),
    paired_bootstrap_final_v3_to_v4=dict(wall=[0.0124, -0.0169, 0.0528],
                                         off=[-0.0015, -0.0283, 0.0209],
                                         note="feature effect only, before the readout"))

#: The readout these scores are produced with.  It is NOT a plain threshold, and the choice
#: between the two arms is made per domain inside each fold
#: (scripts/eval_expected_score_readout.py).  In every fold it selects the same pair.
READOUT = dict(
    selection="per-domain, in-fold, over {cohort_cut, resid, resid_adapt, expected_tuned}",
    wall=("resid_adapt -- the physics-conditioned keep/add readout, with its four cuts "
          "PERTURBED by a fitted slope on the vessel's mean score.  b=0 reproduces the "
          "cohort readout exactly, so it can only move if the statistic pays."),
    off=("expected_tuned -- rank the nodes and commit the prefix that maximises the "
         "EXPECTED severity score, using the model's own p in place of GT, with two "
         "in-fold scalars (gamma sharpening, prefix scale) correcting for miscalibration. "
         "This is the fix for the low-burden precision problem: the budget adapts per "
         "vessel with no label.  Off-wall 0.7136 -> 0.7359 against a cohort cut."),
    temporal=("time-conditioned head, 4 seeds averaged, monotone in time, and every node in "
              "the committed set is clot at the last timestep (commit_final)"),
    scripts=["scripts/eval_expected_score_readout.py", "scripts/eval_strict_temporal.py"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="clot_gnn_v4")
    ap.add_argument("--cache", default="v5")
    ap.add_argument("--repoint", action="store_true",
                    help="move data/reference/clot_gnn_locked.json to this artifact")
    args = ap.parse_args()

    dev_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = attach_physics(load_cache(args.cache))
    pool = [a for a in eligible_pool() if a in cache]
    classes = classes_for(pool, PACKS)
    pool = [a for a in pool if a in classes]
    prio = [a for a in pool if is_priority(classes[a])]
    cols = [str(c) for c in cache[pool[0]]["cols"]]
    missing = [c for c in V4_CHANNELS if c not in cols]
    if missing:
        raise SystemExit("cache %r is not a v4 cache; missing %s" % (args.cache, missing))
    print("[i] pool n=%d priority=%d (%s), %d features"
          % (len(pool), len(prio), ", ".join(prio), len(cols)), flush=True)

    out = REPO / "outputs/clot_ml/locked" / args.name
    out.mkdir(parents=True, exist_ok=True)
    Xall = np.concatenate([cache[a]["X"] for a in pool])
    mu, sd = Xall.mean(0), Xall.std(0)
    sd[sd < 1e-6] = 1.0

    members, t0 = [], time.time()
    for cname, over in MEMBERS.items():
        cfg = dict(BASE)
        cfg.update({k: v for k, v in over.items() if k != "seeds"})
        for s in range(int(over.get("seeds", 3))):
            fn = "member_%s_s%d.pth" % (cname, s)
            if (out / fn).exists():                      # resumable, as v1's had to be
                members.append(dict(file=fn, config=cname, seed=s, **cfg))
                print("   kept  %-22s" % fn, flush=True)
                continue
            predict = train_one(pool, cache, SimpleNamespace(**cfg, seeds=1), dev_t, seed=s)
            model = predict.model
            assert isinstance(model, ClotGNN), type(model)
            torch.save(dict(state_dict=model.state_dict(), cfg=cfg, seed=s,
                            in_dim=model.enc[0].in_features - model.extra_dim,
                            extra_dim=model.extra_dim), out / fn)
            members.append(dict(file=fn, config=cname, seed=s, **cfg))
            print("   saved %-22s (%.0fs)" % (fn, time.time() - t0), flush=True)

    np.savez_compressed(out / "feature_norm.npz", mu=mu, sd=sd, cols=np.array(cols))
    manifest = dict(
        name=args.name, kind="gnn_ensemble",
        promoted_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        description=(
            "Physics-informed recurrent clot GNN, PHASE10. Same architecture and "
            "configurations as clot_gnn_v2, plus the 13 advective-transport / indicator-gate "
            "channels of src/clot_ml/features_v4.py -- COMSOL's own operator "
            "(dMat/dt + u.grad(Mat) = 0, D=0, wall flux BC) solved on the mesh, which is the "
            "first off-wall feature family that transports along the flow rather than along "
            "the mesh normal. Fitted on the full 19-vessel eligible pool; SEALED never seen. "
            "Scores are STRICTLY NESTED out-of-fold from the CV that selected this design, "
            "not from these weights."),
        docs="docs/PHASE10_V4.md", supersedes="clot_gnn_v3",
        feature_cache=args.cache, v4_channels=list(V4_CHANNELS),
        requires=("src.clot_ml.features_v4.augment_sample must be applied to a v3 sample "
                  "before predict_scores -- these members expect %d features" % len(cols)),
        training_pool=list(pool), priority_anchors=list(prio),
        geometry_classes={a: classes[a] for a in pool},
        scores_strict_cv=STRICT_CV, readout=READOUT,
        n_members=len(members), members=members,
        feature_norm="feature_norm.npz", n_features=len(cols))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("locked %d members -> %s  (%.0fs)" % (len(members), out, time.time() - t0))

    if args.repoint:
        ptr = REPO / "data/reference/clot_gnn_locked.json"
        prev = json.loads(ptr.read_text()) if ptr.exists() else {}
        ptr.write_text(json.dumps(dict(
            name=args.name, kind="gnn_ensemble",
            path=str(out.relative_to(REPO)).replace("\\", "/"),
            manifest=str((out / "manifest.json").relative_to(REPO)).replace("\\", "/"),
            promoted_at=manifest["promoted_at"], docs="docs/PHASE10_V4.md",
            supersedes=prev.get("name", "clot_gnn_v3"),
            scores_strict_cv=STRICT_CV, readout=READOUT), indent=2))
        print("pointer -> %s (now %s)" % (ptr, args.name))
    else:
        print("[i] pointer NOT moved; rerun with --repoint to ship this generation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
