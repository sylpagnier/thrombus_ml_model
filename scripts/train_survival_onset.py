"""Train the survival onset head and score it on the time-resolved deploy metric.

The physics model supplies the committed SET (frozen -- it is already at the flow-oracle
ceiling); the head supplies WHEN each of those nodes ignites.  Because the set is frozen,
the final-time score is identical to the physics model's by construction and the head can
only change the trajectory.

Four arms are reported on the same set, so any difference is attributable to timing alone:

    flash      onset from the physics ODE                     (what ships today)
    linear     onset from ridge regression on the same feats  (the analytical control)
    survival   onset from the discrete-time hazard head       (the ML arm)
    oracle     onset = GT onset                               (the ceiling, +0.084/+0.036)

Clean protocol throughout: FIT / DEV / SEALED disjoint, truncated and empty-GT vessels
excluded, selection on DEV only, sealed opened once at the end.

    python scripts/train_survival_onset.py --flow gt --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.biochem_gnn.mat_growth_simple import (  # noqa: E402
    WALL_COHORT_V2_GENERALIZATION, WALL_COHORT_V2_TRAIN,
)
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.physics_wall_model import t0_flow_fields  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.core_physics.temporal_metrics import curve_l1, gt_onset_index, spearman  # noqa: E402
from src.differentiable_wall_model.survival_head import (  # noqa: E402
    FEATURE_NAMES, SurvivalOnsetHead, build_features, onset_targets,
    predicted_onset_frac, survival_nll,
)
from src.evaluation.clot_relaxed_metrics import (  # noqa: E402
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
)

DIR = Path("data/processed/graphs_biochem_anchors")
OUT = Path("outputs/survival_onset")
MIN_T, RELAX, GROW = 150, 2.0, 6
N_BINS, N_EVAL = 20, 12
DEV_STRIDE = 4


def physics_set(data, w, bio, flow):
    f = t0_flow_fields(data, bio, hops=3 if flow == "gt" else 4, flow_source=flow)
    ei = data.edge_index.numpy()
    n = len(w)
    A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.int8)
    cur = (f.gate > 0) & w
    adm = (f.sr < float(bio.lss) * RELAX) & w
    for _ in range(GROW):
        cur = cur | (((A @ cur.astype(np.int8)) > 0) & adm)
    return cur


def load(name, bio, phys, flow):
    p = DIR / f"{name}.pt"
    if not p.exists():
        return None
    d = torch.load(p, map_location="cpu", weights_only=False)
    T = int(d.y.shape[0])
    if T < MIN_T or (flow == "pred" and getattr(d, "u0_pred", None) is None):
        return None
    w = d.mask_wall.reshape(-1).bool().numpy()
    gt_on = gt_onset_index(d, phys, w)
    if not ((gt_on >= 0) & w).any():
        return None
    x, wall, phys_on, phys_idx = build_features(d, bio, phys, flow=flow)
    b, e = onset_targets(gt_on, T, N_BINS)
    S = physics_set(d, w, bio, flow)
    eval_ts = np.unique(np.linspace(T // N_EVAL, T - 1, N_EVAL).astype(int))
    gt_at = {int(ti): (gt_clot_phi_at_time(d, int(ti), phys,
                                           device=torch.device("cpu")).reshape(-1) > 0.5).numpy() & w
             for ti in eval_ts}
    return dict(name=name, d=d, w=w, x=x, bin=b, event=e, gt_on=gt_on, T=T,
                phys_on=phys_on, phys_idx=phys_idx, S=S, eval_ts=eval_ts, gt_at=gt_at,
                t=d.t.reshape(-1).numpy().astype(np.float64))


def build_splits(bio, phys, flow):
    cache, pool, sealed = {}, [], []
    for n in sorted(set(WALL_COHORT_V2_TRAIN) | set(WALL_COHORT_V2_GENERALIZATION)):
        c = load(n, bio, phys, flow)
        if c is None:
            continue
        cache[n] = c
        (sealed if n in WALL_COHORT_V2_GENERALIZATION else pool).append(n)
    dev = [n for i, n in enumerate(pool) if i % DEV_STRIDE == 0]
    fit = [n for n in pool if n not in dev]
    assert not (set(fit) & set(dev)) and not (set(fit) & set(sealed)) and not (set(dev) & set(sealed))
    return fit, dev, sealed, cache


# ------------------------------------------------------------------------ scoring

def time_resolved(c, onset_frac):
    """Median-over-time deploy score for a given per-node onset fraction in [0,1].

    ``onset_frac`` is only consulted on the physics committed set, so the final mask is
    the physics model's regardless of what the head predicts.
    """
    scores = []
    w = c["w"]
    wt = torch.tensor(w.astype(np.float32))
    for ti in c["eval_ts"]:
        frac = ti / max(c["T"] - 1, 1)
        pred = c["S"] & (onset_frac <= frac)
        gt = torch.tensor(c["gt_at"][int(ti)].astype(np.float32))
        m = compute_clot_relaxed_metrics(torch.tensor(pred.astype(np.float32)) * wt,
                                         gt * wt, c["d"].edge_index,
                                         wall_mask=torch.tensor(w))
        scores.append(clot_score_from_deploy_dict(metrics_to_deploy_prefix(m)))
    return float(np.median(scores)), scores


def onset_quality(c, onset_frac):
    """Rank correlation and curve L1 against GT, on nodes both sides commit."""
    w, gt_on, T = c["w"], c["gt_on"], c["T"]
    both = w & (gt_on >= 0) & c["S"]
    rho = spearman(onset_frac[both], gt_on[both] / max(T - 1, 1)) if both.sum() > 3 else float("nan")
    idx = np.where(c["S"], np.clip((onset_frac * (T - 1)).astype(int), 0, T - 1), -1)
    return rho, curve_l1(idx, gt_on, c["t"], w)


def arm_scores(cache, names, onset_fn):
    out = {}
    for n in names:
        c = cache[n]
        of = onset_fn(c)
        s, _ = time_resolved(c, of)
        rho, cl = onset_quality(c, of)
        out[n] = dict(score=s, rho=rho, curve_l1=cl)
    return out


def agg(rows, key):
    v = [r[key] for r in rows.values() if r[key] == r[key]]
    return float(np.mean(v)) if v else float("nan")


def agg_med(rows, key="score"):
    return float(np.median([r[key] for r in rows.values() if r[key] == r[key]]))


# ----------------------------------------------------------------------- training

def train(cache, fit, dev, device, cfg, seed, log=print):
    torch.manual_seed(seed)
    model = SurvivalOnsetHead(len(FEATURE_NAMES), hidden=cfg["hidden"], n_bins=N_BINS,
                              layers=cfg["layers"], dropout=cfg["dropout"]).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    n_ev = sum(int((cache[n]["event"].numpy() & cache[n]["w"]).sum()) for n in fit)
    n_cen = sum(int((~cache[n]["event"].numpy() & cache[n]["w"]).sum()) for n in fit)
    log("[train] %d params | %d events + %d censored = %d node-subjects across %d vessels"
        % (n_par, n_ev, n_cen, n_ev + n_cen, len(fit)))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    rng = random.Random(seed)
    best, stale, hist = -1.0, 0, []
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def head_onset(c):
        model.eval()
        with torch.no_grad():
            lg = model(c["x"].to(device), c["d"].edge_index.to(device))
        return predicted_onset_frac(lg).cpu().numpy()

    for ep in range(cfg["epochs"]):
        model.train()
        order = list(fit)
        rng.shuffle(order)
        tot, t0 = 0.0, time.time()
        for n in order:
            c = cache[n]
            lg = model(c["x"].to(device), c["d"].edge_index.to(device))
            mask = torch.tensor(c["w"], dtype=torch.bool, device=device)
            loss = survival_nll(lg, c["bin"].to(device), c["event"].to(device), mask)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach())
        sched.step()
        r = arm_scores(cache, dev, head_onset)
        m = agg_med(r)
        hist.append(dict(epoch=ep + 1, loss=tot / len(order), dev_median=m,
                         dev_rho=agg(r, "rho"), dev_curve=agg(r, "curve_l1")))
        flag = ""
        if m > best:
            best, stale = m, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            flag = "  [*]"
        else:
            stale += 1
        log("    ep %02d/%d nll %.4f | DEV median %.4f rho %.3f curveL1 %.4f  %.0fs%s"
            % (ep + 1, cfg["epochs"], tot / len(order), m, agg(r, "rho"),
               agg(r, "curve_l1"), time.time() - t0, flag))
        if stale >= cfg["patience"]:
            log("    early stop")
            break
    model.load_state_dict(best_state)
    return model, best, hist, n_par


def ridge_baseline(cache, fit, alpha=1.0):
    """Analytical control: ridge regression onto normalised GT onset, same features.

    The project's standing lesson (19.2: 187k parameters bought +0.024 over a logistic
    regression) is that a learned model must be checked against a linear one on identical
    inputs before any capacity claim is made.
    """
    X, Y = [], []
    for n in fit:
        c = cache[n]
        m = c["w"] & (c["gt_on"] >= 0)
        X.append(c["x"].numpy()[m])
        Y.append(c["gt_on"][m] / max(c["T"] - 1, 1))
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
    W = np.linalg.solve(Xb.T @ Xb + alpha * np.eye(Xb.shape[1]), Xb.T @ Y)

    def fn(c):
        xb = np.concatenate([c["x"].numpy(), np.ones((len(c["x"]), 1))], 1)
        return np.clip(xb @ W, 0.0, 1.0)
    return fn


DEFAULT_CFG = dict(hidden=64, layers=2, dropout=0.1, lr=3e-3, wd=1e-4,
                   epochs=60, patience=12)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", default="gt", choices=["gt", "pred"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=DEFAULT_CFG["epochs"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = dict(DEFAULT_CFG, epochs=args.epochs)
    tag = args.tag or f"{args.flow}_seed{args.seed}"

    fit, dev, sealed, cache = build_splits(bio, phys, args.flow)
    print("device %s | flow=%s | seed=%d | bins=%d" % (device, args.flow, args.seed, N_BINS))
    print("FIT n=%d  DEV n=%d %s  SEALED n=%d %s"
          % (len(fit), len(dev), [a[-3:] for a in dev], len(sealed), [a[-3:] for a in sealed]))

    def flash(c):
        return c["phys_on"]

    def oracle(c):
        return np.where(c["gt_on"] >= 0, c["gt_on"] / max(c["T"] - 1, 1), 1.0)

    model, best_dev, hist, n_par = train(cache, fit, dev, device, cfg, args.seed)
    ridge = ridge_baseline(cache, fit)

    def head(c):
        model.eval()
        with torch.no_grad():
            lg = model(c["x"].to(device), c["d"].edge_index.to(device))
        return predicted_onset_frac(lg).cpu().numpy()

    arms = {"flash (physics)": flash, "linear (ridge)": ridge,
            "survival (ML)": head, "oracle onset": oracle}
    res = {}
    for split, names in (("DEV", dev), ("SEALED", sealed)):
        print("\n%s\n%s -- %s\n%s" % ("=" * 78, split,
                                      "selection" if split == "DEV" else "opened once",
                                      "=" * 78))
        print("%-18s %9s %9s %9s" % ("arm", "median", "rho", "curveL1"))
        res[split] = {}
        for nm, fn in arms.items():
            r = arm_scores(cache, names, fn)
            res[split][nm] = dict(median=agg_med(r), mean=agg(r, "score"),
                                  rho=agg(r, "rho"), curve_l1=agg(r, "curve_l1"),
                                  per_vessel={k: v["score"] for k, v in r.items()})
            print("%-18s %9.4f %9.3f %9.4f"
                  % (nm, agg_med(r), agg(r, "rho"), agg(r, "curve_l1")))
        b = res[split]["flash (physics)"]["median"]
        o = res[split]["oracle onset"]["median"]
        for nm in ("linear (ridge)", "survival (ML)"):
            v = res[split][nm]["median"]
            frac = (v - b) / (o - b) if abs(o - b) > 1e-9 else float("nan")
            print("   %-16s vs physics %+.4f   = %.0f%% of the available prize"
                  % (nm, v - b, 100 * frac))

    (OUT / f"{tag}.json").write_text(json.dumps(dict(
        flow=args.flow, seed=args.seed, cfg=cfg, n_params=n_par, fit=fit, dev=dev,
        sealed=sealed, best_dev=best_dev, hist=hist, results=res), indent=2, default=float),
        encoding="utf-8")
    torch.save(model.state_dict(), OUT / f"{tag}.pth")
    print("\nwrote %s" % (OUT / f"{tag}.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
