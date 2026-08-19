"""Temporal head ONLY, on a physics base that is first proven to match the hard model.

WHAT WENT WRONG BEFORE.  Every earlier "ML vs physics" number was dominated by the
differentiable base, not the heads.  Scoring the UNTRAINED soft base against the hard
physics model on sealed:

    arm      untrained soft base   hard physics   gap
    gt              0.7982            0.9093     -0.1111
    pred            0.6745            0.8567     -0.1821

Cause, pinned to patient031: the base ran ``wake=8``.  The wake feedback multiplies both
``sr`` and ``dsrx`` by ``1 - wake*phi``, but the separation branch of the law is
*proportional to* ``|dsrx|``, so wake suppresses exactly the mechanism a separation-gated
vessel depends on.  p031's ``mat/crit`` peak fell 1.69 -> 1.42, which on a vessel sitting
at the ignition bifurcation flipped 31 committed nodes to 1 (score 0.81 -> 0.14).  At
``wake<=2`` the soft base matches or beats the hard model.

SO THIS SCRIPT:
  * refuses to train until the base passes a PARITY GATE against the hard physics model
    (on FIT+DEV only -- never sealed);
  * selects the base scalars on DEV, by MEDIAN deploy_clot_score;
  * trains ONLY the temporal corrector.  No spatial head, no uncertainty gate.
  * reports a three-way ablation -- hard physics / tuned base alone / base + temporal head
    -- so the head's marginal contribution is isolated rather than confounded with the base.

DEPLOY-LEGALITY.  ``RateMultiplierCorrector`` reads only ``(mat, mas, d_mat, sr)`` and
``edge_index``.  It never touches ``data.x``, so the GT-t=0 prior channels that
contaminated the old pred arm cannot reach it -- the pred arm here is honestly bandaid-free.

SPLITS are identical for both arms (restricted to vessels carrying ``u0_pred``) so gt and
pred are directly comparable, and are chosen by a positional rule, not by score.

    python scripts/sweep_temporal_only.py --flow gt   --seed 0
    python scripts/sweep_temporal_only.py --flow pred --seed 0
"""
from __future__ import annotations

import argparse
import copy
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
from src.core_physics.species_pushforward_continuous import resolve_deploy_eval_time_index  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.core_physics.temporal_metrics import curve_l1, gt_onset_index  # noqa: E402
from src.differentiable_wall_model.improved_heads import (  # noqa: E402
    RateMultiplierCorrector, trajectory_probs,
)
from src.differentiable_wall_model.parameters import GlobalPhysicsParameters  # noqa: E402
from src.differentiable_wall_model.temporal_models import TemporalDifferentiableWallModel  # noqa: E402
from src.evaluation.clot_relaxed_metrics import (  # noqa: E402
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
)

DIR = Path("data/processed/graphs_biochem_anchors")
OUT = Path("outputs/temporal_only")
MIN_T = 150
RELAX, GROW, STENCIL = 2.0, 6, {"gt": 3, "pred": 4}
N_TRAJ = 8
DEV_STRIDE = 4          # positional rule: every 4th eligible vessel becomes DEV
PARITY_TOL = 0.02       # soft base must sit within this of the hard model on FIT+DEV


# ------------------------------------------------------------------ data / splits

def eligible(name, phys, require_pred):
    p = DIR / f"{name}.pt"
    if not p.exists():
        return None
    d = torch.load(p, map_location="cpu", weights_only=False)
    if int(d.y.shape[0]) < MIN_T:
        return None
    if require_pred and getattr(d, "u0_pred", None) is None:
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
    return dict(name=name, d=d, w=w, gt=gt, traj_idx=torch.tensor(idx, dtype=torch.long),
                traj_gt=traj_gt, gt_onset=gt_onset_index(d, phys, w.numpy()),
                t=d.t.reshape(-1).numpy().astype(np.float64))


def build_splits(phys):
    """Identical splits for both arms: only vessels carrying ``u0_pred`` are used.

    The old pred arm ended up with an EMPTY dev set (039/040/041/044 all lack ``u0_pred``),
    so ``best`` never rose above its -1.0 sentinel and the "result" was the untrained model.
    Requiring u0_pred everywhere makes the two arms comparable and that failure impossible.
    """
    cache = {}
    pool, sealed = [], []
    for n in sorted(set(WALL_COHORT_V2_TRAIN) | set(WALL_COHORT_V2_GENERALIZATION)):
        e = eligible(n, phys, require_pred=True)
        if e is None:
            continue
        cache[n] = e
        (sealed if n in WALL_COHORT_V2_GENERALIZATION else pool).append(n)
    dev = [n for i, n in enumerate(pool) if i % DEV_STRIDE == 0]
    fit = [n for n in pool if n not in dev]
    assert dev and fit and sealed
    assert not (set(fit) & set(dev)) and not (set(fit) & set(sealed)) and not (set(dev) & set(sealed))
    return fit, dev, sealed, cache


# ----------------------------------------------------------------------- scoring

def score_pred(c, pred):
    m = compute_clot_relaxed_metrics(pred.reshape(-1).cpu() * c["w"].float(), c["gt"],
                                     c["d"].edge_index, wall_mask=c["w"])
    return clot_score_from_deploy_dict(metrics_to_deploy_prefix(m))


def physics_pred(c, bio, flow):
    """The solid zero-parameter baseline: two t=0 gates + shear-admitted graph growth."""
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


def med(rows, key="score"):
    return float(np.median([v[key] for v in rows.values()]))


def mean(rows, key="score"):
    return float(np.mean([v[key] for v in rows.values()]))


# ------------------------------------------------------------------------- model

def make_base(wake, tau, phi_temp, corrector=None, device="cpu"):
    pp = GlobalPhysicsParameters(init_wake=wake, init_tau_low=tau, init_tau_sep=tau,
                                 init_phi_temp=phi_temp)
    for p in pp.parameters():
        p.requires_grad = False              # temporal head only
    return TemporalDifferentiableWallModel(parameter_provider=pp,
                                           temporal_corrector=corrector).to(device)


def select_base(cache, fit, dev, device, flow, bio, log=print):
    """Choose the base scalars on DEV by MEDIAN deploy score. Sealed is not consulted."""
    log("\n[base selection] on DEV only, by median deploy_clot_score")
    best = None
    for wake in (0.0, 0.5, 1.0, 2.0):
        for tau in (0.02, 0.05, 0.10, 0.25):
            for pt in (0.005, 0.05):
                m = make_base(wake, tau, pt, device=device)
                r = eval_model(m, dev, cache, device, flow, bio)
                log("   wake %-4.1f tau %-5.2f phi %-6.3f -> DEV median %.4f mean %.4f"
                    % (wake, tau, pt, med(r), mean(r)))
                if best is None or med(r) > best[0]:
                    best = (med(r), wake, tau, pt)
    log("   selected wake=%.1f tau=%.2f phi_temp=%.3f (DEV median %.4f)"
        % (best[1], best[2], best[3], best[0]))
    return dict(wake=best[1], tau=best[2], phi_temp=best[3])


def parity_gate(cache, names, base_cfg, device, flow, bio, log=print):
    """Refuse to train on a base that does not reproduce the hard physics model.

    This is the guard whose absence cost the entire previous round: the soft base was
    -0.11 below the hard model before any learning, and every "ML delta" measured that
    instead of the heads.
    """
    m = make_base(base_cfg["wake"], base_cfg["tau"], base_cfg["phi_temp"], device=device)
    soft = eval_model(m, names, cache, device, flow, bio)
    hard = {n: score_pred(cache[n], physics_pred(cache[n], bio, flow)) for n in names}
    gap = mean(soft) - float(np.mean(list(hard.values())))
    worst = min(((soft[n]["score"] - hard[n]), n) for n in names)
    log("\n[parity gate] FIT+DEV  soft base %.4f vs hard physics %.4f  -> gap %+.4f"
        % (mean(soft), float(np.mean(list(hard.values()))), gap))
    log("   worst vessel: %s %+.4f" % (worst[1], worst[0]))
    ok = gap > -PARITY_TOL
    log("   %s (tolerance %.3f)" % ("PASS" if ok else "FAIL -- refusing to train", PARITY_TOL))
    return ok, gap, {n: soft[n]["score"] for n in names}, hard


# ---------------------------------------------------------------------- training

def losses(out, c, bio, device, cfg):
    """Differentiable surrogate on ``Mat`` -- the only quantity the temporal head moves.

    CRITICAL: the loss uses ``cfg['loss_temp']``, NOT the model's ``phi_temp``.  The
    readout temperature is selected for SCORING (hard 0.5 threshold), and the winner is
    typically ~0.005, which makes the sigmoid a step function: measured, **0.00% of wall
    nodes** fall inside its gradient band, so the head receives essentially no signal no
    matter what the learning rate is.  At ``loss_temp=0.5`` that becomes ~95%.  Sharp for
    scoring, smooth for gradients.

    The final mask is supervised through the last trajectory slice rather than through
    ``prob_clot``: the head only affects ``Mat``, and ``prob_clot`` adds the graph-growth
    diffusion, which would inject the same step-function sharpness back into the loss.
    """
    wall = c["w"].to(device)
    wf = wall.float()
    crit = float(bio.viscosity_mat_crit)
    tmp = cfg["loss_temp"]
    p_tr = trajectory_probs(out["mat_traj"], crit, tmp, wf, c["traj_idx"].to(device))
    t_tr = c["traj_gt"].to(device)
    l_traj = F.binary_cross_entropy(p_tr[:, wall].clamp(1e-6, 1 - 1e-6),
                                    t_tr[:, wall].clamp(0, 1))
    # deploy time carries extra weight: it is the scored slice
    p_f = p_tr[-1][wall].clamp(1e-6, 1 - 1e-6)
    t_f = c["gt"].to(device)[wall].clamp(0, 1)
    inter = (p_f * t_f).sum()
    dice = 1.0 - (2.0 * inter + 1e-6) / (p_f.sum() + t_f.sum() + 1e-6)
    l_fin = F.binary_cross_entropy(p_f, t_f) + cfg["dice_weight"] * dice
    return cfg["traj_weight"] * l_traj + l_fin, float(l_traj), float(l_fin)


def train(cfg, base_cfg, fit, dev, cache, device, flow, bio, *, epochs, patience, seed,
          log=print):
    torch.manual_seed(seed)
    corr = RateMultiplierCorrector(hidden_channels=cfg["hidden"], cap=cfg["rate_cap"])
    model = make_base(base_cfg["wake"], base_cfg["tau"], base_cfg["phi_temp"],
                      corrector=corr, device=device)
    params = [p for n, p in model.named_parameters() if "temporal_corrector" in n]
    for p in params:
        p.requires_grad = True
    n_par = sum(p.numel() for p in params)
    log("\n[train] temporal head only: %d trainable parameters" % n_par)
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    # EMA damps the epoch-to-epoch swing (the DEV trace previously oscillated 0.87<->0.77).
    # Kept on CPU, and only over floating-point entries -- integer buffers cannot be averaged.
    ema = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    ema_keys = [k for k, v in ema.items() if v.is_floating_point()]
    rng = random.Random(seed)
    best, stale, hist = -1.0, 0, []
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for ep in range(epochs):
        model.train()
        order = list(fit)
        rng.shuffle(order)
        tl = tf = 0.0
        t0 = time.time()
        for n in order:
            out = model(cache[n]["d"], flow_source=flow, device=device)
            loss, a, b = losses(out, cache[n], bio, device, cfg)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg["clip"])
            opt.step()
            tl += a
            tf += b
            with torch.no_grad():
                sd = model.state_dict()
                for k in ema_keys:
                    ema[k].mul_(cfg["ema"]).add_(sd[k].detach().cpu(), alpha=1 - cfg["ema"])
        sched.step()
        raw = eval_model(model, dev, cache, device, flow, bio)
        cur = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema)
        ema_r = eval_model(model, dev, cache, device, flow, bio)
        use_ema = med(ema_r) > med(raw)
        if not use_ema:
            model.load_state_dict(cur)
        sel = ema_r if use_ema else raw
        hist.append(dict(epoch=ep + 1, dev_median=med(sel), dev_mean=mean(sel),
                         dev_curve=mean(sel, "curve_l1"), ema=bool(use_ema)))
        flag = ""
        if med(sel) > best:
            best, stale = med(sel), 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            flag = "  [*]"
        else:
            stale += 1
        log("    ep %02d/%d traj %.4f fin %.4f | DEV median %.4f mean %.4f curveL1 %.4f %s %.0fs%s"
            % (ep + 1, epochs, tl / len(order), tf / len(order), med(sel), mean(sel),
               mean(sel, "curve_l1"), "ema" if use_ema else "raw", time.time() - t0, flag))
        if stale >= patience:
            log("    early stop")
            break
    model.load_state_dict(best_state)
    return model, best, hist, n_par


DEFAULT_CFG = dict(hidden=32, rate_cap=1.0, lr=1e-3, weight_decay=1e-4, clip=1.0,
                   ema=0.9, traj_weight=3.0, dice_weight=4.0, loss_temp=0.5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", default="gt", choices=["gt", "pred"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.flow}_seed{args.seed}"

    fit, dev, sealed, cache = build_splits(phys)
    print("device %s | flow=%s | seed=%d" % (device, args.flow, args.seed))
    print("FIT    n=%2d %s" % (len(fit), " ".join(a[-3:] for a in fit)))
    print("DEV    n=%2d %s" % (len(dev), " ".join(a[-3:] for a in dev)))
    print("SEALED n=%2d %s" % (len(sealed), " ".join(a[-3:] for a in sealed)))

    base_cfg = select_base(cache, fit, dev, device, args.flow, bio)
    ok, gap, _, _ = parity_gate(cache, fit + dev, base_cfg, device, args.flow, bio)
    if not ok:
        print("[ABORT] base is not at parity with the physics model; a head trained here "
              "would only be measuring the base defect.")
        return 2

    model, best_dev, hist, n_par = train(DEFAULT_CFG, base_cfg, fit, dev, cache, device,
                                         args.flow, bio, epochs=args.epochs,
                                         patience=args.patience, seed=args.seed)

    print("\n%s\nSEALED -- opened once, all choices frozen\n%s" % ("=" * 78, "=" * 78))
    no_head = make_base(base_cfg["wake"], base_cfg["tau"], base_cfg["phi_temp"], device=device)
    r_base = eval_model(no_head, sealed, cache, device, args.flow, bio)
    r_head = eval_model(model, sealed, cache, device, args.flow, bio)
    r_phys = {n: dict(score=score_pred(cache[n], physics_pred(cache[n], bio, args.flow)))
              for n in sealed}
    print("%-12s %9s %9s %9s | %9s" % ("vessel", "physics", "base", "base+head", "head-base"))
    for n in sealed:
        print("%-12s %9.4f %9.4f %9.4f | %+9.4f"
              % (n, r_phys[n]["score"], r_base[n]["score"], r_head[n]["score"],
                 r_head[n]["score"] - r_base[n]["score"]))
    print("\n%-22s median %.4f  mean %.4f" % ("physics baseline", med(r_phys), mean(r_phys)))
    print("%-22s median %.4f  mean %.4f  curveL1 %.4f"
          % ("tuned base (no head)", med(r_base), mean(r_base), mean(r_base, "curve_l1")))
    print("%-22s median %.4f  mean %.4f  curveL1 %.4f"
          % ("base + temporal head", med(r_head), mean(r_head), mean(r_head, "curve_l1")))
    print("\nHEAD's marginal contribution : median %+.4f  mean %+.4f  curveL1 %+.4f"
          % (med(r_head) - med(r_base), mean(r_head) - mean(r_base),
             mean(r_head, "curve_l1") - mean(r_base, "curve_l1")))
    print("vs the physics baseline      : median %+.4f  mean %+.4f"
          % (med(r_head) - med(r_phys), mean(r_head) - mean(r_phys)))

    (OUT / f"{tag}.json").write_text(json.dumps(dict(
        flow=args.flow, seed=args.seed, cfg=DEFAULT_CFG, base_cfg=base_cfg,
        parity_gap=gap, n_params=n_par, fit=fit, dev=dev, sealed=sealed,
        best_dev_median=best_dev, hist=hist,
        sealed_physics={n: r_phys[n]["score"] for n in sealed},
        sealed_base={n: r_base[n] for n in sealed},
        sealed_head={n: r_head[n] for n in sealed}), indent=2, default=float), encoding="utf-8")
    torch.save(model.state_dict(), OUT / f"{tag}.pth")
    print("\nwrote %s" % (OUT / f"{tag}.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
