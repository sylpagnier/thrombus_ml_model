"""PHASE9: flow-aware residual GNN for the full-mesh clot map.  Targets wall>0.9, off>0.7.

Three things distinguish this from the four ML attempts the repo has already buried:

  * **the loss is the metric** (`src/clot_ml/softmetric.py`) -- a differentiable copy of
    `0.5*dilation_IoU + 0.5*relaxed_F0.5`, per domain.  BCE optimises none of that;
    PHASE6_RESULTS 15.3 showed the score is a cliff that plain per-node losses cannot see.
  * **the physics is the base, not the competitor** -- the backbone's `log(Mat/crit)` is an
    additive base for the regression head (zero-init, so an untrained net *is* the physics),
    and its mask drives a residual readout: separate thresholds for keeping a
    physics-positive node and for adding a physics-negative one.  Wall error is two opposite
    failure modes (weak-sep FP on 018/019/025, ungated FN on 012/028) and one threshold
    cannot fix both.
  * **anisotropic message passing** -- upstream/downstream aggregation weighted by the t=0
    velocity projected on each edge.  Isotropic smoothing of the source is measurably wrong
    (PHASE6_RESULTS 3.4).

    python scripts/train_clot_gnn.py --folds 4 --epochs 300
    python scripts/train_clot_gnn.py --lovo --epochs 300 --seeds 3 --tag final
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.clot_ml.data import attach_physics, load_cache, splits  # noqa: E402
from src.clot_ml.evaluate import banner  # noqa: E402
from src.clot_ml.gnn import ClotGNN, to_device  # noqa: E402
from src.clot_ml.protocol import Bench  # noqa: E402
from src.clot_ml.recurrent import (  # noqa: E402
    N_FEEDBACK, N_FEEDBACK_ADV, advective_operators, feedback_channels,
    feedback_channels_advective, neighbour_operator,
)
from src.clot_ml.severity_metric import DEFAULT as SEVERITY_CFG, soft_severity  # noqa: E402
from src.clot_ml.softmetric import (  # noqa: E402
    dilation_operator, soft_dilate, soft_score, to_torch_sparse,
)

LOG = REPO / "outputs/phase9_log.jsonl"
GRID = np.linspace(0.02, 0.995, 24)


def log_result(tag, summ, extra=None):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    rec = dict(tag=tag, t=time.strftime("%m-%d %H:%M:%S"), fit=summ["fit"], dev=summ["dev"])
    if extra:
        rec.update(extra)
    with LOG.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# residual readout: keep / add thresholds per domain, against the physics mask
# ---------------------------------------------------------------------------
def apply_readout(S, score, th):
    w, ph = S["wall"], S["phys_mask"]
    keep_w, add_w, keep_o, add_o = th
    wall_pred = (w & ph & (score >= keep_w)) | (w & ~ph & (score >= add_w))
    off_pred = (~w & ph & (score >= keep_o)) | (~w & ~ph & (score >= add_o))
    return wall_pred | off_pred


def pick_readout(bench, scores, anchors, grid):
    """Four scalars, chosen per domain on a coarse grid.  Domains are scored separately."""
    def best_pair(domain_of, dom_key):
        best, pair = -1e9, (float(grid[0]), float(grid[0]))
        for tk in grid:
            for ta in grid:
                vals = []
                for a in anchors:
                    S = bench.cache[a]
                    d = domain_of(S)
                    ph = S["phys_mask"]
                    pr = (d & ph & (scores[a] >= tk)) | (d & ~ph & (scores[a] >= ta))
                    v = bench.vs[a].score(pr, d)
                    if v == v:
                        vals.append(v)
                if vals and np.mean(vals) > best:
                    best, pair = float(np.mean(vals)), (float(tk), float(ta))
        return pair, best

    (kw, aw), _ = best_pair(lambda S: S["wall"], "wall")
    (ko, ao), _ = best_pair(lambda S: ~S["wall"], "off")
    return (kw, aw, ko, ao)


def to_weighted_sparse(M, dev_t):
    """Like ``to_torch_sparse`` but KEEPS the values -- upwind weights are not indicators."""
    C = M.tocoo()
    idx = torch.tensor(np.stack([C.row, C.col]), dtype=torch.long, device=dev_t)
    val = torch.tensor(C.data, dtype=torch.float32, device=dev_t)
    return torch.sparse_coo_tensor(idx, val, M.shape).coalesce()


# ---------------------------------------------------------------------------
def build_graph(S, mu, sd, dev_t, *, need_soft=False, need_fb=False, adv_fb=False):
    g = to_device(S, mu, sd, dev_t)
    g["phys"] = torch.tensor(S["phys_mask"].astype(np.float32), device=dev_t)
    if need_fb:
        g["At"] = to_torch_sparse(neighbour_operator(S["edge_index"], len(S["wall"])), dev_t)
        g["owner"] = torch.tensor(S["owner"].astype(np.int64), device=dev_t)
        if adv_fb:
            Wu, Wd = advective_operators(S["pos"], S["edge_index"], S["u"], S["v"])
            g["Wup"] = to_weighted_sparse(Wu, dev_t)
            g["Wdn"] = to_weighted_sparse(Wd, dev_t)
    if need_soft:
        D = dilation_operator(S["edge_index"], len(S["wall"]), 2)
        g["D"] = to_torch_sparse(D, dev_t)
        g["gt_dil"] = soft_dilate(g["y"], g["D"]).detach()
    g["off"] = 1.0 - g["wall"]
    return g


def rollout(model, g, rounds, adv_fb=False):
    """K shared-weight refinement rounds; round 0 occlusion is the physics mask."""
    if rounds <= 1:
        return model(g["x"], g["ei"], g["ea"], g["w_up"], g["w_dn"], g["mat_phys"])
    p = g["phys"].clone()
    R = int(rounds)
    for k in range(R):
        extra = (feedback_channels_advective(p, g["At"], g["Wup"], g["Wdn"],
                                            g["owner"])
                 if adv_fb else feedback_channels(p, g["At"], g["owner"]))
        logit, reg = model(g["x"], g["ei"], g["ea"], g["w_up"], g["w_dn"], g["mat_phys"],
                           extra=extra)
        # truncated BPTT: only the last round carries gradient, so peak memory is one
        # round's activations rather than R (a 4 GB card cannot hold R=3 at dim 96).
        if k < R - 1:
            p = torch.sigmoid(logit).detach()
        else:
            p = torch.sigmoid(logit)
    return logit, reg


def prepare(cache, anchors, mu, sd, dev_t, need_soft=True, need_fb=False, adv_fb=False):
    G = {}
    for a in anchors:
        g = to_device(cache[a], mu, sd, dev_t)
        S = cache[a]
        g["phys"] = torch.tensor(S["phys_mask"].astype(np.float32), device=dev_t)
        if need_fb:
            g["At"] = to_torch_sparse(neighbour_operator(S["edge_index"], len(S["wall"])), dev_t)
            g["owner"] = torch.tensor(S["owner"].astype(np.int64), device=dev_t)
            if adv_fb:
                Wu, Wd = advective_operators(S["pos"], S["edge_index"], S["u"], S["v"])
                g["Wup"] = to_weighted_sparse(Wu, dev_t)
                g["Wdn"] = to_weighted_sparse(Wd, dev_t)
        if need_soft:
            D = dilation_operator(S["edge_index"], len(S["wall"]), 2)
            g["D"] = to_torch_sparse(D, dev_t)
            gt = g["y"]
            g["gt_dil"] = soft_dilate(gt, g["D"]).detach()
        g["off"] = 1.0 - g["wall"]
        G[a] = g
    return G


def train_one(train_anchors, cache, args, dev_t, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xall = np.concatenate([cache[a]["X"] for a in train_anchors])
    mu, sd = Xall.mean(0), Xall.std(0)
    sd[sd < 1e-6] = 1.0
    rounds = int(getattr(args, "rounds", 1))
    adv_fb = bool(getattr(args, "adv_fb", False))
    off_only = bool(getattr(args, "off_only", False))
    G = prepare(cache, train_anchors, mu, sd, dev_t, need_soft=args.metric_w > 0,
                need_fb=rounds > 1, adv_fb=adv_fb)
    in_dim = G[train_anchors[0]]["x"].shape[1]
    edim = G[train_anchors[0]]["ea"].shape[1]
    n_fb = N_FEEDBACK_ADV if adv_fb else N_FEEDBACK
    model = ClotGNN(in_dim, edim, dim=args.dim, layers=args.layers, drop=args.drop,
                    extra_dim=(n_fb if rounds > 1 else 0)).to(dev_t)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(args.epochs * len(train_anchors), 1),
        pct_start=0.25)
    pw = torch.tensor(args.pos_weight, device=dev_t)
    warm = max(int(args.epochs * args.metric_start), 1)
    for ep in range(args.epochs):
        model.train()
        use_metric = args.metric_w > 0 and ep >= warm
        for i in np.random.permutation(len(train_anchors)):
            g = G[train_anchors[i]]
            opt.zero_grad(set_to_none=True)
            logit, reg = rollout(model, g, rounds, adv_fb)
            # OFF-WALL SPECIALIST.  The metric is domain-restricted, so a model whose whole
            # loss is the off-wall domain is a legitimate arm (docs/PHASE9_ML.md 0 already
            # reports a wall-specialised ensemble).  `off_mult` only reweights the metric
            # term on a shared trunk that the wall's ~5x larger BCE still dominates; this
            # masks BCE and the regression too, so nothing in the objective is wall.
            if off_only:
                sel_n = g["off"] > 0.5
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logit[sel_n], g["y"][sel_n], pos_weight=pw)
                loss = loss + args.reg_w * torch.nn.functional.smooth_l1_loss(
                    reg[sel_n], g["mat_gt"][sel_n])
            else:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logit, g["y"], pos_weight=pw)
                loss = loss + args.reg_w * torch.nn.functional.smooth_l1_loss(
                    reg, g["mat_gt"])
            if use_metric:
                p = torch.sigmoid(logit)
                parts = []
                # off-wall is the domain furthest from target (0.63 vs 0.70) and it holds a
                # fifth of the nodes; weight its metric term explicitly.
                use_sev = str(getattr(args, "metric", "legacy")) == "severity"
                doms_ = ((("off", 1.0),) if off_only
                         else (("wall", 1.0),
                               ("off", float(getattr(args, "off_mult", 1.0)))))
                for dom, mult in doms_:
                    sc_ = (soft_severity(p, g["y"], g["D"], g[dom], g["gt_dil"], SEVERITY_CFG)
                           if use_sev else
                           soft_score(p, g["y"], g["D"], g[dom], g["gt_dil"],
                                      float(getattr(args, "loss_shape_w", 0.5))))
                    if sc_ is not None:
                        parts.append(mult * (1.0 - sc_))
                if parts:
                    loss = loss + args.metric_w * torch.stack(parts).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
    model.eval()

    @torch.no_grad()
    def predict(anchor):
        g = build_graph(cache[anchor], mu, sd, dev_t, need_fb=rounds > 1, adv_fb=adv_fb)
        logit, _ = rollout(model, g, rounds, adv_fb)
        return torch.sigmoid(logit).cpu().numpy()

    @torch.no_grad()
    def predict_reg(anchor):
        """The REGRESSION head's ``log1p(Mat/crit)``, which no readout has ever used.

        GT clot *is* ``{Mat >= crit}`` (PHASE7 10.1), so this head predicts the physical
        quantity the label is a threshold on, while the classifier predicts the label.
        `docs/PHASE9_ML.md` 13.1 measured the two and the regression head ranks GT `Mat`
        slightly *better* (0.619 against the classifier's 0.601) -- and then only the
        classifier was ever read out.  Saved so the readout can use both.
        """
        g = build_graph(cache[anchor], mu, sd, dev_t, need_fb=rounds > 1, adv_fb=adv_fb)
        _, reg = rollout(model, g, rounds, adv_fb)
        return reg.cpu().numpy()

    predict.model = model          # for checkpointing (scripts/promote_clot_gnn.py)
    predict.norm = (mu, sd)
    predict.reg = predict_reg
    return predict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--pos-weight", type=float, default=30.0)
    ap.add_argument("--reg-w", type=float, default=1.0)
    ap.add_argument("--metric-w", type=float, default=2.0)
    ap.add_argument("--metric-start", type=float, default=0.3)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--metric", default="legacy", choices=["legacy", "severity"],
                    help="which score the soft loss imitates")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--lovo", action="store_true")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--flow", default="gt")
    ap.add_argument("--save-scores", default="")
    args = ap.parse_args()

    dev_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = attach_physics(load_cache(args.flow))
    fit, dev = splits(cache)
    bench = Bench(cache, fit, dev)
    print("[i] dev=%s FIT=%d DEV=%d ep=%d dim=%d L=%d mw=%.1f pw=%.0f seeds=%d"
          % (dev_t, len(fit), len(dev), args.epochs, args.dim, args.layers,
             args.metric_w, args.pos_weight, args.seeds), flush=True)

    folds = ([[a] for a in fit] if args.lovo
             else [list(fit[i::args.folds]) for i in range(args.folds)])
    rows, t0, all_scores = {}, time.time(), {}
    for k, held in enumerate(folds):
        tr = [a for a in fit if a not in held]
        sc = {}
        for s in range(args.seeds):
            predict = train_one(tr, cache, args, dev_t, seed=s)
            for a in tr + held:
                sc[a] = sc.get(a, 0.0) + predict(a) / args.seeds
        th = pick_readout(bench, sc, tr, GRID)
        for a in held:
            rows[a] = bench.row(a, apply_readout(cache[a], sc[a], th))
            all_scores[a] = sc[a]
        print("   fold %d/%d th=(%.2f,%.2f,%.2f,%.2f) %s (%.0fs)"
              % (k + 1, len(folds), *th,
                 " ".join("%s w%.3f o%s" % (a[-3:], rows[a]["wall"],
                          ("%.3f" % rows[a]["off"]) if rows[a]["off"] == rows[a]["off"] else "-")
                          for a in held), time.time() - t0), flush=True)

    sc = {}
    for s in range(args.seeds):
        predict = train_one(fit, cache, args, dev_t, seed=s)
        for a in fit + dev:
            sc[a] = sc.get(a, 0.0) + predict(a) / args.seeds
    th = pick_readout(bench, sc, fit, GRID)
    for a in dev:
        rows[a] = bench.row(a, apply_readout(cache[a], sc[a], th))
        all_scores[a] = sc[a]
    summ = bench.summarise(rows)
    tag = "gnn" + (("/" + args.tag) if args.tag else "")
    print(banner(tag, summ), " (%.0fs)" % (time.time() - t0), flush=True)
    for a in sorted(rows):
        r = rows[a]
        print("      %-12s wall %.4f off %6s" %
              (a, r["wall"], ("%.4f" % r["off"]) if r["off"] == r["off"] else " n/a"), flush=True)
    log_result(tag, summ, dict(epochs=args.epochs, dim=args.dim, layers=args.layers,
                               drop=args.drop, lr=args.lr, pos_weight=args.pos_weight,
                               reg_w=args.reg_w, metric_w=args.metric_w, seeds=args.seeds,
                               rounds=args.rounds,
                               folds=("lovo" if args.lovo else args.folds), th=list(th)))
    if args.save_scores:
        np.savez_compressed(args.save_scores, **{a: v for a, v in all_scores.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
