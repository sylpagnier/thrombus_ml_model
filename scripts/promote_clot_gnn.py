"""Train and lock the shipped clot-GNN ensemble.

Writes every ensemble member's weights (fitted on ALL of FIT, never on DEV or SEALED) plus
a manifest, under a single referenceable name:

    outputs/clot_ml/locked/<name>/member_<cfg>_s<seed>.pth
    outputs/clot_ml/locked/<name>/manifest.json
    data/reference/clot_gnn_locked.json          <- the pointer other code reads

    python scripts/promote_clot_gnn.py --name clot_gnn_v1
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

from src.clot_ml.data import attach_physics, load_cache, splits  # noqa: E402
from src.clot_ml.gnn import ClotGNN  # noqa: E402
from train_clot_gnn import train_one  # noqa: E402

# The four configurations whose average is the headline ensemble (docs/PHASE9_ML.md 2).
BASE = dict(epochs=80, dim=64, layers=4, drop=0.1, lr=3e-3, wd=1e-4, pos_weight=30.0,
            reg_w=1.0, metric_w=2.0, metric_start=0.3, rounds=3, off_mult=1.0)
MEMBERS = {
    "rec3s": dict(rounds=3, seeds=3),
    "rec5s": dict(rounds=5, seeds=3),
    "rec3s6": dict(rounds=3, seeds=6),
    "rec3o": dict(rounds=3, seeds=3, off_mult=2.5),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="clot_gnn_v1")
    ap.add_argument("--flow", default="gt")
    args = ap.parse_args()

    dev_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = attach_physics(load_cache(args.flow))
    fit, dev = splits(cache)
    out = REPO / "outputs/clot_ml/locked" / args.name
    out.mkdir(parents=True, exist_ok=True)

    Xall = np.concatenate([cache[a]["X"] for a in fit])
    mu, sd = Xall.mean(0), Xall.std(0)
    sd[sd < 1e-6] = 1.0
    cols = [str(c) for c in cache[fit[0]]["cols"]]
    members, t0 = [], time.time()

    for cname, over in MEMBERS.items():
        cfg = dict(BASE)
        cfg.update({k: v for k, v in over.items() if k != "seeds"})
        n_seeds = int(over.get("seeds", 3))
        for s in range(n_seeds):
            fn = "member_%s_s%d.pth" % (cname, s)
            if (out / fn).exists():          # resumable: a CUDA hiccup must not lose work
                members.append(dict(file=fn, config=cname, seed=s, **cfg))
                print("   kept  %-24s" % fn, flush=True)
                continue
            ns = SimpleNamespace(**cfg, seeds=1)
            # train_one returns a closure; grab the module it captured for serialisation
            predict = train_one(fit, cache, ns, dev_t, seed=s)
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
            "Physics-informed recurrent clot GNN, PHASE9.  Ensemble of 15 members over 4 "
            "configurations, each fitted on ALL of FIT (DEV and SEALED never seen).  "
            "Input is the t=0 GT flow field plus mesh geometry; output is one per-node score "
            "read out per domain."),
        docs="docs/PHASE9_ML.md",
        flow=args.flow,
        fit_anchors=list(fit), dev_anchors=list(dev),
        readout="thresh, thresholds tuned per domain on FIT",
        scores_out_of_fold=dict(
            fit=dict(wall=0.8998, off=0.6145, full=0.8458),
            dev=dict(wall=0.8918, off=0.8058, full=0.8604),
            physics_baseline=dict(fit_wall=0.8584, fit_off=0.3651,
                                  dev_wall=0.8901, dev_off=0.5051)),
        n_members=len(members), members=members,
        feature_norm="feature_norm.npz", n_features=len(cols))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    ptr = REPO / "data/reference/clot_gnn_locked.json"
    ptr.write_text(json.dumps(dict(
        name=args.name, path=str(out.relative_to(REPO)).replace("\\", "/"),
        manifest=str((out / "manifest.json").relative_to(REPO)).replace("\\", "/"),
        promoted_at=manifest["promoted_at"], docs="docs/PHASE9_ML.md",
        scores=manifest["scores_out_of_fold"]), indent=2))
    print("locked %d members -> %s\npointer -> %s  (%.0fs)"
          % (len(members), out, ptr, time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
