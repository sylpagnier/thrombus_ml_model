"""ML corrector v2 -- trajectory supervision, split objectives, bounded + gated residuals.

Same clean protocol as ``sweep_ml_clean_protocol.py`` (FIT / DEV / SEALED, disjoint,
truncated and empty-GT vessels excluded, sealed opened once).  Four changes to the model
and the loss, each answering a measured defect -- see
``src/differentiable_wall_model/improved_heads.py`` for the reasoning:

  1. the temporal head is supervised on the TRAJECTORY, not just the endpoint;
  2. the two heads get DIFFERENT objectives instead of competing on one scalar;
  3. both corrections are BOUNDED so they stay residual to the physics;
  4. the spatial correction is GATED BY BASE UNCERTAINTY.

Also fixes the deployable arm: ``data.x`` ships GT t=0 priors, so with ``--flow pred`` the
prior channels are rebuilt from ``u0_pred`` (``deploy_features.rebuild_prior_channels``).
Without that, "deployable" still reads GT flow through the feature vector.

    python scripts/sweep_ml_v2.py --flow gt   --seed 0
    python scripts/sweep_ml_v2.py --flow pred --seed 0
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
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.biochem_gnn.mat_growth_simple import (  # noqa: E402
    WALL_COHORT_V2_GENERALIZATION, WALL_COHORT_V2_TRAIN,
)
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.physics_wall_model import t0_flow_fields  # noqa: E402
from src.core_physics.species_pushforward_continuous import resolve_deploy_eval_time_index  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.core_physics.temporal_metrics import curve_l1, gt_onset_index  # noqa: E402
from src.differentiable_wall_model.deploy_features import rebuild_prior_channels  # noqa: E402
from src.differentiable_wall_model.improved_heads import (  # noqa: E402
    BoundedMeshGraphNet, RateMultiplierCorrector, trajectory_probs,
)
from src.differentiable_wall_model.temporal_models import TemporalDifferentiableWallModel  # noqa: E402
from src.evaluation.clot_relaxed_metrics import (  # noqa: E402
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
)

DIR = Path("data/processed/graphs_biochem_anchors")
OUT = Path("outputs/ml_v2")
DEV_CANDIDATES = ("patient039", "patient040", "patient041", "patient044")
MIN_T = 150
RELAX, GROW, STENCIL = 2.0, 6, {"gt": 3, "pred": 4}
N_TRAJ = 8          # supervised timesteps per vessel


def eligible(name, phys, flow):
    p = DIR / f"{name}.pt"
    if not p.exists():
        return None
    d = torch.load(p, map_location="cpu", weights_only=False)
    if int(d.y.shape[0]) < MIN_T:
        return None
    if flow == "pred" and getattr(d, "u0_pred", None) is None:
        return None
    w = d.mask_wall.reshape(-1).bool()
    te = resolve_deploy_eval_time_index(int(d.y.shape[0]))
    gt = gt_clot_phi_at_time(d, te, phys, device=torch.device("cpu")).reshape(-1) * w.float()
    if float(gt.sum()) <= 0:
        return None
    T = int(d.y.shape[0])
    idx = np.unique(np.linspace(T // N_TRAJ, T - 1, N_TRAJ).astype(int))
    traj_gt = torch.stack([
        gt_clot_phi_at_time(d, int(k), phys, device=torch.device("cpu")).reshape(-1) * w.float()
        for k in idx])
    return dict(d=d, w=w, gt=gt, traj_idx=torch.tensor(idx, dtype=torch.long),
                traj_gt=traj_gt, gt_onset=gt_onset_index(d, phys, w.numpy()),
                t=d.t.reshape(-1).numpy().astype(np.float64))


def build_splits(phys, flow):
    fit, dev, sealed, cache = [], [], [], {}
    for n in sorted(set(WALL_COHORT_V2_TRAIN) | set(WALL_COHORT_V2_GENERALIZATION)):
        e = eligible(n, phys, flow)
        if e is None:
            continue
        cache[n] = e
        (sealed if n in WALL_COHORT_V2_GENERALIZATION
         else dev if n in DEV_CANDIDATES else fit).append(n)
    assert not (set(fit) & set(dev)) and not (set(fit) & set(sealed)) and not (set(dev) & set(sealed))
    return fit, dev, sealed, cache


def score_pred(c, pred):
    m = compute_clot_relaxed_metrics(pred.reshape(-1).cpu() * c["w"].float(), c["gt"],
                                     c["d"].edge_index, wall_mask=c["w"])
    return clot_score_from_deploy_dict(metrics_to_deploy_prefix(m))


def physics_pred(c, bio, flow):
    d, w = c["d"], c["w"]
    f = t0_flow_fields(d, bio, hops=STENCIL[flow], flow_source=flow)
    wn = w.numpy()
    ei = d.edge_index.numpy()
    n = len(wn)
    A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.int8)
    cur = (f.gate > 0) & wn
    adm = (f.sr < float(bio.lss) * RELAX) & wn
    for _ in range(GROW):
        cur = cur | (((A @ cur.astype(np.int8)) > 0) & adm)
    return torch.tensor(cur.astype(np.float32))


@torch.no_grad()
def eval_split(model, names, cache, device, flow, bio):
    model.eval()
    out = {}
    for n in names:
        c = cache[n]
        o = model(c["d"], flow_source=flow, device=device)
        pred = (o["prob_clot"] >= 0.5).float()
        traj = o["mat_traj"].detach().cpu().numpy()
        hot = traj >= float(bio.viscosity_mat_crit)
        idx = np.where(hot.any(0), hot.argmax(0), -1)
        out[n] = dict(score=score_pred(c, pred),
                      curve_l1=curve_l1(idx, c["gt_onset"], c["t"], c["w"].numpy()))
    return out


def make_model(cfg, cache, device, flow, bio, phys):
    xcache = {}

    def x_provider(data, fs):
        k = id(data)
        if k not in xcache:
            xcache[k] = (data.x.detach() if fs == "gt"
                         else rebuild_prior_channels(data, bio, phys, flow_source="pred",
                                                     hops=STENCIL["pred"]))
        return xcache[k]

    base = TemporalDifferentiableWallModel(
        temporal_corrector=RateMultiplierCorrector(hidden_channels=cfg["temporal_hidden"],
                                                   cap=cfg["rate_cap"])).to(device)
    d0 = cache[next(iter(cache))]["d"]
    m = BoundedMeshGraphNet(base_model=base, in_channels=d0.x.size(1),
                            hidden_channels=cfg["hidden_channels"],
                            num_layers=cfg["num_layers"],
                            delta_cap=cfg["delta_cap"],
                            uncertainty_gate=True, x_provider=x_provider).to(device)
    nn.init.zeros_(m.node_decoder.weight)
    nn.init.zeros_(m.node_decoder.bias)
    for n_, p in m.named_parameters():
        if "temporal_corrector" in n_:
            p.requires_grad = True
    spatial = [p for n_, p in m.named_parameters()
               if "base_model" not in n_ and p.requires_grad]
    temporal = [p for n_, p in m.named_parameters() if "temporal_corrector" in n_]
    return m, spatial, temporal


def mask_loss(p_c, t_c, cfg):
    bce = F.binary_cross_entropy(p_c, t_c)
    inter = (p_c * t_c).sum()
    dice = 1.0 - (2.0 * inter + 1e-6) / (p_c.sum() + t_c.sum() + 1e-6)
    return bce + cfg["dice_weight"] * dice


def traj_loss(out, c, bio, device):
    """FIX 1: supervise the whole growth curve, not only the endpoint."""
    wall = c["w"].to(device).float()
    phi_temp = out["params"].phi_temp
    p = trajectory_probs(out["mat_traj"], float(bio.viscosity_mat_crit), phi_temp,
                         wall, c["traj_idx"].to(device))
    t = c["traj_gt"].to(device)
    m = wall > 0
    return F.binary_cross_entropy(p[:, m].clamp(1e-6, 1 - 1e-6), t[:, m].clamp(0, 1))


def train(cfg, fit, dev, cache, device, flow, bio, phys, *, epochs, patience, seed, log=print):
    model, spatial, temporal = make_model(cfg, cache, device, flow, bio, phys)
    opt_s = torch.optim.Adam(spatial, lr=cfg["spatial_lr"])
    opt_t = torch.optim.Adam(temporal, lr=cfg["temporal_lr"])
    rng = random.Random(seed)
    best, stale = -1.0, 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    hist = []
    for ep in range(epochs):
        model.train()
        order = list(fit)
        rng.shuffle(order)
        ls, lt, t0 = 0.0, 0.0, time.time()
        for n in order:
            c = cache[n]
            wm = c["w"].to(device)
            # --- FIX 2a: temporal stage owns the TRAJECTORY loss ---
            out = model(c["d"], flow_source=flow, device=device)
            l_t = traj_loss(out, c, bio, device)
            opt_t.zero_grad()
            l_t.backward()
            torch.nn.utils.clip_grad_norm_(temporal, 1.0)
            opt_t.step()
            lt += float(l_t.detach())
            # --- FIX 2b: spatial stage owns the FINAL MASK ---
            out = model(c["d"], flow_source=flow, device=device)
            p_c = out["prob_clot"][wm].clamp(1e-6, 1 - 1e-6)
            t_c = c["gt"].to(device)[wm].clamp(0, 1)
            l_s = mask_loss(p_c, t_c, cfg)
            opt_s.zero_grad()
            l_s.backward()
            torch.nn.utils.clip_grad_norm_(spatial, 1.0)
            opt_s.step()
            ls += float(l_s.detach())
        ds = eval_split(model, dev, cache, device, flow, bio)
        dev_score = float(np.mean([v["score"] for v in ds.values()]))
        dev_curve = float(np.mean([v["curve_l1"] for v in ds.values()]))
        hist.append(dict(score=dev_score, curve_l1=dev_curve))
        flag = ""
        if dev_score > best:
            best, stale = dev_score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            flag = "  [*]"
        else:
            stale += 1
        log("    ep %02d/%d  mask %.4f traj %.4f | DEV score %.4f curveL1 %.4f  %.0fs%s"
            % (ep + 1, epochs, ls / len(order), lt / len(order), dev_score, dev_curve,
               time.time() - t0, flag))
        if stale >= patience:
            log("    early stop")
            break
    model.load_state_dict(best_state)
    return model, best, hist


DEFAULT_CFG = dict(hidden_channels=64, num_layers=2, dice_weight=8.5,
                   spatial_lr=5e-4, temporal_lr=2e-4,
                   delta_cap=2.0, rate_cap=1.0, temporal_hidden=16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", default="gt", choices=["gt", "pred"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-fit", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.flow}_seed{args.seed}"

    fit, dev, sealed, cache = build_splits(phys, args.flow)
    if args.max_fit:
        fit = fit[:args.max_fit]
    print("device %s | flow=%s | seed=%d" % (device, args.flow, args.seed))
    print("FIT n=%d  DEV n=%d %s  SEALED n=%d %s"
          % (len(fit), len(dev), [a[-3:] for a in dev], len(sealed), [a[-3:] for a in sealed]))

    base_sc = {n: score_pred(cache[n], physics_pred(cache[n], bio, args.flow))
               for n in dev + sealed}
    dev_bar = float(np.mean([base_sc[n] for n in dev]))
    seal_bar = float(np.mean([base_sc[n] for n in sealed]))
    print("physics bar: DEV %.4f | SEALED %.4f" % (dev_bar, seal_bar))

    model, best_dev, hist = train(DEFAULT_CFG, fit, dev, cache, device, args.flow, bio, phys,
                                  epochs=args.epochs, patience=args.patience, seed=args.seed)
    print("best DEV %.4f (bar %.4f, %+.4f)" % (best_dev, dev_bar, best_dev - dev_bar))

    print("\n=== SEALED (opened once) ===")
    ml = eval_split(model, sealed, cache, device, args.flow, bio)
    for n in sealed:
        print("  %-12s ML %.4f  physics %.4f  %+.4f   curveL1 %.4f"
              % (n, ml[n]["score"], base_sc[n], ml[n]["score"] - base_sc[n], ml[n]["curve_l1"]))
    ms = float(np.mean([ml[n]["score"] for n in sealed]))
    print("\nSEALED  ML mean %.4f | physics %.4f | delta %+.4f" % (ms, seal_bar, ms - seal_bar))
    res = dict(flow=args.flow, seed=args.seed, cfg=DEFAULT_CFG, fit=fit, dev=dev,
               sealed=sealed, physics=base_sc, ml={n: ml[n] for n in sealed},
               best_dev=best_dev, dev_bar=dev_bar, sealed_bar=seal_bar,
               sealed_ml_mean=ms, hist=hist)
    (OUT / f"{tag}.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    torch.save(model.state_dict(), OUT / f"{tag}.pth")
    print("wrote %s" % (OUT / f"{tag}.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
