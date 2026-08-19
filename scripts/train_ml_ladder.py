"""CONTROLLED ABLATION: MeshGraphNet + cGNODE vs the zero-parameter physics model.

Every previous comparison of these two heads was invalidated by a defect found afterwards,
so none of them tested the architecture.  This run fixes all of them at once and reports a
five-rung ladder on identical splits and seeds, so each component's marginal contribution
is separable.

THE BUGS, AND WHAT IS DONE ABOUT EACH
  1. Base did not reproduce the hard physics (-0.11 gt / -0.18 pred before any learning),
     because ``wake=8`` damps ``|dsrx|`` and the separation branch is proportional to it.
     -> base scalars selected on DEV, then a PARITY GATE refuses to train if the base is
        more than 0.02 below the hard model on FIT+DEV.
  2. Uncertainty gate was inert: median ``4p(1-p)`` = 0.0000, only 3.9% of nodes above 0.1,
     three seeds with different weights giving identical DEV.  -> switched OFF.
  3. Zero-init final layers gave the graph layers EXACTLY zero gradient (measured
     conv.lin_l/lin_r norms 0.000e+00).  -> small-random init on both heads.
  4. The readout is a near-step function, so almost no node contributes gradient.  And
     parity and trainability turn out to be mutually exclusive at a single temperature:

            phi_temp   parity vs hard   gradient coverage
              0.05         +0.026             1.6%
              0.20         +0.015             7.2%
              0.50         -0.212            95.1%

     -> forward keeps the sharp readout (accuracy); backward uses a soft surrogate.
        For the cGNODE the trajectory loss is computed at ``loss_temp`` directly.  For the
        MeshGraphNet the base logit is CLAMPED to +/-``logit_clamp``: at a 0.5 decision
        threshold this cannot change the evaluated mask at all, but it restores a usable
        d(sigmoid) where the base had saturated to +/-11.5.
  5. Too few / too uniform training vessels.  -> FIT is augmented with truncated runs
     (the repo's own rule makes T<150 a HOLDOUT restriction, not a training one -- per-step
     GT deltas are valid at any run length) and with mirrored packs.  Mirrors of SEALED
     vessels are excluded, which patient010_mirror_y would otherwise leak.

RUNGS   physics | base | base+MGN | base+cGNODE | base+both
DEV selects, SEALED is opened once at the end.

    python scripts/train_ml_ladder.py --flow gt --seed 0
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
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
from src.differentiable_wall_model.improved_heads import (  # noqa: E402
    BoundedMeshGraphNet, RateMultiplierCorrector,
)
from src.differentiable_wall_model.parameters import GlobalPhysicsParameters  # noqa: E402
from src.differentiable_wall_model.temporal_models import TemporalDifferentiableWallModel  # noqa: E402
from src.evaluation.clot_relaxed_metrics import (  # noqa: E402
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
)

DIR = Path("data/processed/graphs_biochem_anchors")
OUT = Path("outputs/ml_ladder")
MIN_T, RELAX, GROW = 150, 2.0, 6
STENCIL = {"gt": 3, "pred": 4}
DEV_STRIDE, N_TRAJ = 4, 8
PARITY_TOL = 0.02


# ------------------------------------------------------------------ clamped MGN

class ClampedMeshGraphNet(BoundedMeshGraphNet):
    """MeshGraphNet whose base logit is clamped so gradient survives a saturated base.

    ``sigmoid'(x)`` is ~1e-5 at |x|=11.5, which is where the sharp physics readout puts
    almost every node -- so the residual head received essentially no gradient.  Clamping
    the base logit to +/-4 leaves every decision at the 0.5 threshold untouched (the sign
    is preserved) while restoring a workable derivative.
    """

    def __init__(self, *args, logit_clamp: float = 4.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.logit_clamp = float(logit_clamp)

    def forward(self, data, *, flow_source="pred", device=None):
        from torch_geometric.utils import scatter

        device = device or torch.device("cpu")
        wall = data.mask_wall.reshape(-1).float().to(device)
        base_out = self.base_model(data, flow_source=flow_source, device=device)
        bp, bg = base_out["prob_clot"], base_out["gate_init"]
        x = torch.nan_to_num(data.x.to(device), nan=0.0, posinf=0.0, neginf=0.0)
        ea = torch.nan_to_num(data.edge_attr.to(device), nan=0.0, posinf=0.0, neginf=0.0)
        v = self.node_encoder(torch.cat([x, bp.unsqueeze(-1), bg.unsqueeze(-1)], -1))
        e = self.edge_encoder(ea)
        row, col = data.edge_index.to(device)
        for i in range(self.processor_steps):
            e = e + self.edge_mlps[i](torch.cat([e, v[row], v[col]], -1))
            agg = scatter(e, row, dim=0, dim_size=v.size(0), reduce="sum")
            v = v + self.node_mlps[i](torch.cat([v, agg], -1))
        delta = self.delta_cap * torch.tanh(self.node_decoder(v).squeeze(-1))
        p = torch.clamp(bp, 1e-5, 1 - 1e-5)
        base_logit = torch.clamp(torch.log(p / (1 - p)), -self.logit_clamp, self.logit_clamp)
        base_out["prob_clot"] = torch.sigmoid(base_logit + delta) * wall
        return base_out


# --------------------------------------------------------------------- data

def prep(d, name, phys, bio, flow, train_only):
    T = int(d.y.shape[0])
    if flow == "pred" and getattr(d, "u0_pred", None) is None:
        return None
    if not train_only and T < MIN_T:
        return None
    w = d.mask_wall.reshape(-1).bool()
    te = resolve_deploy_eval_time_index(T)
    gt = gt_clot_phi_at_time(d, te, phys, device=torch.device("cpu")).reshape(-1) * w.float()
    if float(gt.sum()) <= 0:
        return None
    idx = np.unique(np.linspace(max(T // N_TRAJ, 1), T - 1, N_TRAJ).astype(int))
    traj_gt = torch.stack([
        gt_clot_phi_at_time(d, int(k), phys, device=torch.device("cpu")).reshape(-1) * w.float()
        for k in idx])
    return dict(name=name, d=d, w=w, gt=gt, T=T, traj_idx=torch.tensor(idx, dtype=torch.long),
                traj_gt=traj_gt, gt_onset=gt_onset_index(d, phys, w.numpy()),
                t=d.t.reshape(-1).numpy().astype(np.float64))


def build_splits(bio, phys, flow, augment=True):
    cache, pool, sealed = {}, [], []
    for n in sorted(set(WALL_COHORT_V2_TRAIN) | set(WALL_COHORT_V2_GENERALIZATION)):
        p = DIR / f"{n}.pt"
        if not p.exists():
            continue
        c = prep(torch.load(p, map_location="cpu", weights_only=False), n, phys, bio,
                 flow, train_only=False)
        if c is None:
            continue
        cache[n] = c
        (sealed if n in WALL_COHORT_V2_GENERALIZATION else pool).append(n)
    dev = [n for i, n in enumerate(pool) if i % DEV_STRIDE == 0]
    fit = [n for n in pool if n not in dev]

    extra = []
    if augment:
        # truncated runs: legal for FIT (T>=150 is a HOLDOUT rule), never for DEV/SEALED
        for n in sorted(set(WALL_COHORT_V2_TRAIN)):
            if n in cache:
                continue
            p = DIR / f"{n}.pt"
            if not p.exists():
                continue
            c = prep(torch.load(p, map_location="cpu", weights_only=False), n, phys, bio,
                     flow, train_only=True)
            if c is not None:
                cache[n] = c
                extra.append(n)
        # mirrored packs -- but NEVER a mirror of a sealed vessel (patient010_mirror_y)
        for p in sorted(glob.glob(str(DIR / "*mirror*.pt"))):
            stem = Path(p).stem
            base = stem.split("_")[0]
            if base in WALL_COHORT_V2_GENERALIZATION:
                continue
            c = prep(torch.load(p, map_location="cpu", weights_only=False), stem, phys, bio,
                     flow, train_only=True)
            if c is not None:
                cache[stem] = c
                extra.append(stem)
    fit = fit + extra
    assert not (set(fit) & set(dev)) and not (set(fit) & set(sealed)) and not (set(dev) & set(sealed))
    for n in fit:
        assert n.split("_")[0] not in WALL_COHORT_V2_GENERALIZATION, "sealed leak via %s" % n
    return fit, dev, sealed, cache, extra


# ------------------------------------------------------------------- scoring

def score_pred(c, pred):
    m = compute_clot_relaxed_metrics(pred.reshape(-1).cpu() * c["w"].float(), c["gt"],
                                     c["d"].edge_index, wall_mask=c["w"])
    return clot_score_from_deploy_dict(metrics_to_deploy_prefix(m))


def physics_pred(c, bio, flow):
    f = t0_flow_fields(c["d"], bio, hops=STENCIL[flow], flow_source=flow)
    wn = c["w"].numpy()
    ei = c["d"].edge_index.numpy()
    n = len(wn)
    A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.int8)
    cur = (f.gate > 0) & wn
    adm = (f.sr < float(bio.lss) * RELAX) & wn
    for _ in range(GROW):
        cur = cur | (((A @ cur.astype(np.int8)) > 0) & adm)
    return torch.tensor(cur.astype(np.float32))


@torch.no_grad()
def eval_model(model, names, cache, device, flow, bio):
    model.eval()
    out = {}
    for n in names:
        c = cache[n]
        o = model(c["d"], flow_source=flow, device=device)
        traj = o["mat_traj"].detach().cpu().numpy()
        hot = traj >= float(bio.viscosity_mat_crit)
        idx = np.where(hot.any(0), hot.argmax(0), -1)
        out[n] = dict(score=score_pred(c, (o["prob_clot"] >= 0.5).float()),
                      curve_l1=curve_l1(idx, c["gt_onset"], c["t"], c["w"].numpy()))
    return out


def med(r, k="score"):
    return float(np.median([v[k] for v in r.values()]))


def mean(r, k="score"):
    return float(np.mean([v[k] for v in r.values()]))


# --------------------------------------------------------------------- model

def make_base(cfg, corrector=None, device="cpu"):
    pp = GlobalPhysicsParameters(init_wake=cfg["wake"], init_tau_low=cfg["tau"],
                                 init_tau_sep=cfg["tau"], init_phi_temp=cfg["phi_temp"])
    for p in pp.parameters():
        p.requires_grad = False
    return TemporalDifferentiableWallModel(parameter_provider=pp,
                                           temporal_corrector=corrector).to(device)


def build(rung, base_cfg, cfg, in_ch, device):
    corr = None
    if rung in ("cgnode", "both"):
        corr = RateMultiplierCorrector(hidden_channels=cfg["t_hidden"], cap=cfg["rate_cap"])
    base = make_base(base_cfg, corrector=corr, device=device)
    if rung in ("mgn", "both"):
        m = ClampedMeshGraphNet(base_model=base, in_channels=in_ch,
                                hidden_channels=cfg["s_hidden"], num_layers=cfg["s_layers"],
                                delta_cap=cfg["delta_cap"], uncertainty_gate=False,
                                logit_clamp=cfg["logit_clamp"]).to(device)
        nn.init.normal_(m.node_decoder.weight, std=1e-3)   # NOT zero: see module docstring
        nn.init.zeros_(m.node_decoder.bias)
    else:
        m = base
    for n_, p in m.named_parameters():
        p.requires_grad = ("temporal_corrector" in n_) or ("base_model" not in n_ and
                                                           "param_provider" not in n_)
    spatial = [p for n_, p in m.named_parameters()
               if p.requires_grad and "temporal_corrector" not in n_]
    temporal = [p for n_, p in m.named_parameters()
                if p.requires_grad and "temporal_corrector" in n_]
    return m, spatial, temporal


def traj_loss(out, c, bio, device, loss_temp):
    """cGNODE objective -- the growth CURVE, at a temperature that actually has gradient."""
    wall = c["w"].to(device)
    sel = out["mat_traj"][c["traj_idx"].to(device)]
    p = torch.sigmoid((sel / float(bio.viscosity_mat_crit) - 1.0) / loss_temp)
    t = c["traj_gt"].to(device)
    return F.binary_cross_entropy(p[:, wall].clamp(1e-6, 1 - 1e-6), t[:, wall].clamp(0, 1))


def mask_loss(out, c, device, dice_w):
    """MeshGraphNet objective -- the final mask."""
    wall = c["w"].to(device)
    p = out["prob_clot"][wall].clamp(1e-6, 1 - 1e-6)
    t = c["gt"].to(device)[wall].clamp(0, 1)
    bce = F.binary_cross_entropy(p, t)
    inter = (p * t).sum()
    return bce + dice_w * (1.0 - (2 * inter + 1e-6) / (p.sum() + t.sum() + 1e-6))


def train_rung(rung, base_cfg, cfg, fit, dev, cache, device, flow, bio, seed, log=print):
    torch.manual_seed(seed)
    in_ch = cache[fit[0]]["d"].x.size(1)
    m, spatial, temporal = build(rung, base_cfg, cfg, in_ch, device)
    n_par = sum(p.numel() for p in spatial + temporal)
    if n_par == 0:
        return m, eval_model(m, dev, cache, device, flow, bio), [], 0
    opts = []
    if spatial:
        opts.append((torch.optim.AdamW(spatial, lr=cfg["s_lr"], weight_decay=1e-4), spatial, "mask"))
    if temporal:
        opts.append((torch.optim.AdamW(temporal, lr=cfg["t_lr"], weight_decay=1e-4), temporal, "traj"))
    scheds = [torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=cfg["epochs"]) for o, _, _ in opts]
    ema = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    ema_keys = [k for k, v in ema.items() if v.is_floating_point()]
    rng = random.Random(seed)
    best, stale, hist = -1.0, 0, []
    best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    log("    [%s] %d trainable params" % (rung, n_par))
    for ep in range(cfg["epochs"]):
        m.train()
        order = list(fit)
        rng.shuffle(order)
        acc = {"mask": 0.0, "traj": 0.0}
        t0 = time.time()
        for n in order:
            c = cache[n]
            for opt, plist, kind in opts:
                out = m(c["d"], flow_source=flow, device=device)
                loss = (traj_loss(out, c, bio, device, cfg["loss_temp"]) if kind == "traj"
                        else mask_loss(out, c, device, cfg["dice_w"]))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(plist, 1.0)
                opt.step()
                acc[kind] += float(loss.detach())
            with torch.no_grad():
                sd = m.state_dict()
                for k in ema_keys:
                    ema[k].mul_(cfg["ema"]).add_(sd[k].detach().cpu(), alpha=1 - cfg["ema"])
        for s in scheds:
            s.step()
        raw = eval_model(m, dev, cache, device, flow, bio)
        cur = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        m.load_state_dict(ema)
        er = eval_model(m, dev, cache, device, flow, bio)
        if med(er) <= med(raw):
            m.load_state_dict(cur)
        sel = er if med(er) > med(raw) else raw
        hist.append(dict(epoch=ep + 1, dev_median=med(sel), dev_mean=mean(sel)))
        flag = ""
        if med(sel) > best:
            best, stale = med(sel), 0
            best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            flag = "  [*]"
        else:
            stale += 1
        log("      ep %02d/%d mask %.4f traj %.4f | DEV med %.4f mean %.4f  %.0fs%s"
            % (ep + 1, cfg["epochs"], acc["mask"] / len(order), acc["traj"] / len(order),
               med(sel), mean(sel), time.time() - t0, flag))
        if stale >= cfg["patience"]:
            log("      early stop")
            break
    m.load_state_dict(best_state)
    return m, eval_model(m, dev, cache, device, flow, bio), hist, n_par


def select_base(cache, dev, device, flow, bio, log=print):
    log("\n[base selection] DEV median, phi_temp capped at 0.2 (0.5 breaks parity by -0.21)")
    best = None
    for wake in (0.0, 1.0, 2.0):
        for tau in (0.05, 0.25):
            for pt in (0.05, 0.2):
                cfg = dict(wake=wake, tau=tau, phi_temp=pt)
                r = eval_model(make_base(cfg, device=device), dev, cache, device, flow, bio)
                if best is None or med(r) > best[0]:
                    best = (med(r), cfg)
    log("   selected %s (DEV median %.4f)" % (best[1], best[0]))
    return best[1]


DEFAULT = dict(s_hidden=64, s_layers=2, t_hidden=32, delta_cap=2.0, rate_cap=1.0,
               logit_clamp=4.0, loss_temp=0.5, dice_w=4.0, s_lr=5e-4, t_lr=1e-3,
               ema=0.9, epochs=16, patience=5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", default="gt", choices=["gt", "pred"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=DEFAULT["epochs"])
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = dict(DEFAULT, epochs=args.epochs)
    tag = args.tag or f"{args.flow}_seed{args.seed}"

    fit, dev, sealed, cache, extra = build_splits(bio, phys, args.flow, not args.no_augment)
    print("device %s | flow=%s seed=%d" % (device, args.flow, args.seed))
    print("FIT    n=%2d (%d augmented: %s)" % (len(fit), len(extra),
                                               " ".join(e.replace("patient", "") for e in extra)))
    print("DEV    n=%2d %s" % (len(dev), " ".join(a[-3:] for a in dev)))
    print("SEALED n=%2d %s" % (len(sealed), " ".join(a[-3:] for a in sealed)))

    base_cfg = select_base(cache, dev, device, args.flow, bio)
    chk = fit + dev
    soft = eval_model(make_base(base_cfg, device=device), chk, cache, device, args.flow, bio)
    hard = {n: score_pred(cache[n], physics_pred(cache[n], bio, args.flow)) for n in chk}
    gap = mean(soft) - float(np.mean(list(hard.values())))
    print("[parity gate] soft %.4f vs hard %.4f -> %+.4f  %s"
          % (mean(soft), float(np.mean(list(hard.values()))), gap,
             "PASS" if gap > -PARITY_TOL else "FAIL"))
    if gap <= -PARITY_TOL:
        print("[ABORT] base not at parity; a head trained here measures the base defect.")
        return 2

    results, models = {}, {}
    for rung in ("base", "mgn", "cgnode", "both"):
        print("\n--- rung: %s ---" % rung)
        m, dr, hist, npar = train_rung(rung, base_cfg, cfg, fit, dev, cache, device,
                                       args.flow, bio, args.seed)
        models[rung] = m
        results[rung] = dict(dev=dr, hist=hist, n_params=npar)
        print("    DEV median %.4f mean %.4f" % (med(dr), mean(dr)))

    print("\n%s\nSEALED -- opened once\n%s" % ("=" * 84, "=" * 84))
    phys_s = {n: dict(score=score_pred(cache[n], physics_pred(cache[n], bio, args.flow)))
              for n in sealed}
    seal = {r: eval_model(models[r], sealed, cache, device, args.flow, bio) for r in models}
    print("%-12s %9s %9s %9s %9s %9s" % ("vessel", "physics", "base", "+MGN", "+cGNODE", "+both"))
    for n in sealed:
        print("%-12s %9.4f %9.4f %9.4f %9.4f %9.4f"
              % (n, phys_s[n]["score"], seal["base"][n]["score"], seal["mgn"][n]["score"],
                 seal["cgnode"][n]["score"], seal["both"][n]["score"]))
    pm, pmean = med(phys_s), mean(phys_s)
    print("\n%-14s median %.4f  mean %.4f   (zero parameters)" % ("physics", pm, pmean))
    for r in ("base", "mgn", "cgnode", "both"):
        print("%-14s median %.4f  mean %.4f  curveL1 %.4f | vs physics %+.4f med %+.4f mean"
              % (r, med(seal[r]), mean(seal[r]), mean(seal[r], "curve_l1"),
                 med(seal[r]) - pm, mean(seal[r]) - pmean))
    print("\nMARGINAL vs base:  MGN %+.4f   cGNODE %+.4f   both %+.4f  (sealed mean)"
          % (mean(seal["mgn"]) - mean(seal["base"]),
             mean(seal["cgnode"]) - mean(seal["base"]),
             mean(seal["both"]) - mean(seal["base"])))

    (OUT / f"{tag}.json").write_text(json.dumps(dict(
        flow=args.flow, seed=args.seed, cfg=cfg, base_cfg=base_cfg, parity=gap,
        fit=fit, augmented=extra, dev=dev, sealed=sealed,
        physics={n: phys_s[n]["score"] for n in sealed},
        rungs={r: dict(sealed={n: seal[r][n] for n in sealed},
                       n_params=results[r]["n_params"], hist=results[r]["hist"])
               for r in seal}), indent=2, default=float), encoding="utf-8")
    print("\nwrote %s" % (OUT / f"{tag}.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
