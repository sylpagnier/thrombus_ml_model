"""Does the learned corrector beat the zero-parameter physics model, under a clean protocol?

REPLACES ``scripts/sweep_optuna_generalization.py``, whose reported number was not a
generalization estimate (see ``scripts/audit_optuna_generalization.py``):

  * it trained on ONE vessel (patient020) and then scored that same vessel;
  * it early-stopped and picked checkpoints on the training vessel's own deploy score;
  * all 5 "validation" vessels were train-cohort or excluded -- **none held out**;
  * patient017 has zero GT clot, so ``empty_gt_fp_tol`` awards a free 1.0000;
  * patient015 is a truncated run (T=83) that the repo's own ``evaluate_cohort`` skips;
  * Optuna maximised the metric it reported over 28 trials.

Protocol here:

    FIT     full-horizon WALL_COHORT_V2_TRAIN minus the dev slice. Weights fitted here,
            one optimiser step per vessel per epoch (the old script's gradient saw a
            single graph, so there was no cohort signal in it at all).
    DEV     the repo's own dev designation (039/040/041/044) intersected with
            full-horizon. Early stopping, checkpoint selection and the Optuna objective
            all read DEV and nothing else. Never fitted.
    SEALED  WALL_COHORT_V2_GENERALIZATION. Touched ONCE, for the best trial only, after
            every choice is frozen.

Truncated runs (T<150) and zero-GT vessels are excluded everywhere: on a truncated run
the final map is a different quantity, and an empty-GT vessel scores 1.0 for predicting
nothing, which measures the tolerance rather than the model.

The physics baseline is scored on the same splits up front, so every trial is read
against a fixed bar rather than against the previous trial.

    python scripts/sweep_ml_clean_protocol.py --trials 1        # one default config
    python scripts/sweep_ml_clean_protocol.py --trials 12       # Optuna over DEV
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
from src.differentiable_wall_model.advanced_models import MeshGraphNetCorrector  # noqa: E402
from src.differentiable_wall_model.temporal_models import (  # noqa: E402
    PseudoCGNODE, TemporalDifferentiableWallModel,
)
from src.evaluation.clot_relaxed_metrics import (  # noqa: E402
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
)

DIR = Path("data/processed/graphs_biochem_anchors")
OUT = Path("outputs/ml_clean_protocol")
# The repo's own dev designation (WALL_COHORT_V2_DEV minus the sealed 042/043). Chosen
# because the project already calls these dev, not because they scored well.
DEV_CANDIDATES = ("patient039", "patient040", "patient041", "patient044")
MIN_T = 150
RELAX, GROW, STENCIL = 2.0, 6, {"gt": 3, "pred": 4}
FLOW = "gt"          # arm A; the physics bar below is the arm-A number too


# ----------------------------------------------------------------------- splits

def eligible(name, phys):
    p = DIR / f"{name}.pt"
    if not p.exists():
        return None
    d = torch.load(p, map_location="cpu", weights_only=False)
    if int(d.y.shape[0]) < MIN_T:
        return None                                   # truncated: different quantity
    w = d.mask_wall.reshape(-1).bool()
    te = resolve_deploy_eval_time_index(int(d.y.shape[0]))
    gt = gt_clot_phi_at_time(d, te, phys, device=torch.device("cpu")).reshape(-1) * w.float()
    if float(gt.sum()) <= 0:
        return None                                   # empty GT: a free 1.0, not a score
    return d, w, gt


def build_splits(phys):
    fit, dev, sealed, cache = [], [], [], {}
    for n in sorted(set(WALL_COHORT_V2_TRAIN) | set(WALL_COHORT_V2_GENERALIZATION)):
        e = eligible(n, phys)
        if e is None:
            continue
        cache[n] = e
        if n in WALL_COHORT_V2_GENERALIZATION:
            sealed.append(n)
        elif n in DEV_CANDIDATES:
            dev.append(n)
        else:
            fit.append(n)
    assert not (set(fit) & set(dev)), "fit/dev overlap"
    assert not (set(fit) & set(sealed)), "fit/sealed overlap"
    assert not (set(dev) & set(sealed)), "dev/sealed overlap"
    return fit, dev, sealed, cache


# --------------------------------------------------------------------- scoring

def score_pred(d, w, gt, pred):
    m = compute_clot_relaxed_metrics(pred.reshape(-1).cpu() * w.float(), gt,
                                     d.edge_index, wall_mask=w)
    return clot_score_from_deploy_dict(metrics_to_deploy_prefix(m))


def physics_pred(d, w, bio, flow=FLOW):
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
def ml_score(model, names, cache, device):
    model.eval()
    out = {}
    for n in names:
        d, w, gt = cache[n]
        p = model(d, flow_source=FLOW, device=device)["prob_clot"]
        out[n] = score_pred(d, w, gt, (p >= 0.5).float())
    return out


# ----------------------------------------------------------------------- model

class GradMeshGraphNet(MeshGraphNetCorrector):
    """MeshGraphNet residual head, with gradients allowed through the physics base.

    The parent runs ``base_model`` under ``no_grad``, which would leave the temporal
    corrector unreachable by the optimiser.
    """

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
        dl = self.node_decoder(v).squeeze(-1)
        p = torch.clamp(bp, 1e-5, 1 - 1e-5)
        base_out["prob_clot"] = torch.sigmoid(torch.log(p / (1 - p)) + dl) * wall
        return base_out


def make_model(cfg, in_channels, device):
    base = TemporalDifferentiableWallModel(temporal_corrector=PseudoCGNODE()).to(device)
    m = GradMeshGraphNet(base_model=base, in_channels=in_channels,
                         hidden_channels=cfg["hidden_channels"],
                         num_layers=cfg["num_layers"]).to(device)
    nn.init.zeros_(m.node_decoder.weight)
    nn.init.zeros_(m.node_decoder.bias)          # start as an identity on the physics base
    for name, p in m.named_parameters():
        if "temporal_corrector" in name:
            p.requires_grad = True
    spatial = [p for n, p in m.named_parameters()
               if "base_model" not in n and p.requires_grad]
    temporal = [p for n, p in m.named_parameters() if "temporal_corrector" in n]
    return m, spatial, temporal


def loss_fn(p_c, t_c, cfg):
    bce = F.binary_cross_entropy(p_c, t_c)
    inter = (p_c * t_c).sum()
    dice = 1.0 - (2.0 * inter + 1e-6) / (p_c.sum() + t_c.sum() + 1e-6)
    pt = p_c * t_c + (1 - p_c) * (1 - t_c)
    focal = (((1 - pt) ** 2.0) * F.binary_cross_entropy(p_c, t_c, reduction="none")).mean()
    return bce + cfg["dice_weight"] * dice + cfg["focal_weight"] * focal


# ---------------------------------------------------------------------- training

def train(cfg, fit, dev, cache, device, *, epochs, patience, seed=0, log=print):
    d0 = cache[fit[0]][0]
    model, spatial, temporal = make_model(cfg, d0.x.size(1), device)
    params = spatial + temporal
    opt = torch.optim.Adam([{"params": spatial, "lr": cfg["spatial_lr"]},
                            {"params": temporal, "lr": cfg["temporal_lr"]}])
    rng = random.Random(seed)
    best = -1.0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    stale = 0
    hist = []
    for ep in range(epochs):
        model.train()
        order = list(fit)
        rng.shuffle(order)
        tot, t0 = 0.0, time.time()
        for n in order:
            d, w, gt = cache[n]
            out = model(d, flow_source=FLOW, device=device)
            wm = w.to(device)
            p_c = out["prob_clot"][wm].clamp(1e-6, 1 - 1e-6)
            t_c = gt.to(device)[wm].clamp(0, 1)
            loss = loss_fn(p_c, t_c, cfg)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            opt.step()
            tot += float(loss.detach())
        # ---- selection reads DEV only ----
        ds = ml_score(model, dev, cache, device)
        dev_mean = float(np.mean(list(ds.values())))
        hist.append(dev_mean)
        flag = ""
        if dev_mean > best:
            best, stale = dev_mean, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            flag = "  [*] best"
        else:
            stale += 1
        log("    ep %02d/%d  loss %.4f  DEV mean %.4f  (%s)  %.0fs%s"
            % (ep + 1, epochs, tot / max(len(order), 1), dev_mean,
               " ".join("%s %.3f" % (k[-3:], v) for k, v in ds.items()),
               time.time() - t0, flag))
        if stale >= patience:
            log("    early stop (no DEV improvement for %d epochs)" % patience)
            break
    model.load_state_dict(best_state)
    return model, best, hist


DEFAULT_CFG = dict(hidden_channels=64, num_layers=2, dice_weight=8.5, focal_weight=5.0,
                   spatial_lr=5e-4, temporal_lr=5e-5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--max-fit", type=int, default=0, help="cap FIT vessels (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT / "result.json"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    OUT.mkdir(parents=True, exist_ok=True)

    fit, dev, sealed, cache = build_splits(phys)
    if args.max_fit:
        fit = fit[:args.max_fit]
    print("device: %s" % device)
    print("\nSPLITS  (T>=%d and non-empty GT enforced everywhere)" % MIN_T)
    print("  FIT    n=%2d  %s" % (len(fit), " ".join(a[-3:] for a in fit)))
    print("  DEV    n=%2d  %s   <- selection only, never fitted" % (len(dev), " ".join(a[-3:] for a in dev)))
    print("  SEALED n=%2d  %s   <- scored ONCE at the end" % (len(sealed), " ".join(a[-3:] for a in sealed)))

    print("\nPHYSICS BASELINE (zero learned parameters, flow=%s)" % FLOW)
    base = {}
    for n in fit + dev + sealed:
        d, w, gt = cache[n]
        base[n] = score_pred(d, w, gt, physics_pred(d, w, bio))
    for lbl, names in (("FIT", fit), ("DEV", dev), ("SEALED", sealed)):
        v = [base[n] for n in names]
        print("  %-6s mean %.4f  median %.4f" % (lbl, np.mean(v), np.median(v)))
    dev_bar = float(np.mean([base[n] for n in dev]))
    print("  --> the bar the ML model must clear on DEV: %.4f" % dev_bar)

    results = []

    def run_cfg(cfg, tag):
        print("\n%s\n### %s  %s\n%s" % ("#" * 70, tag, cfg, "#" * 70))
        model, best_dev, hist = train(cfg, fit, dev, cache, device,
                                      epochs=args.epochs, patience=args.patience, seed=args.seed)
        print("  best DEV mean %.4f   (physics bar %.4f, delta %+.4f)"
              % (best_dev, dev_bar, best_dev - dev_bar))
        results.append(dict(tag=tag, cfg=cfg, dev=best_dev, hist=hist))
        return model, best_dev

    if args.trials <= 1:
        model, best_dev = run_cfg(DEFAULT_CFG, "single default config")
        best_cfg = DEFAULT_CFG
    else:
        import optuna
        store = {}

        def objective(trial):
            cfg = dict(
                hidden_channels=trial.suggest_categorical("hidden_channels", [32, 64, 128]),
                num_layers=trial.suggest_categorical("num_layers", [2, 3, 4]),
                dice_weight=trial.suggest_float("dice_weight", 0.0, 10.0),
                focal_weight=trial.suggest_float("focal_weight", 0.0, 10.0),
                spatial_lr=trial.suggest_float("spatial_lr", 1e-5, 1e-3, log=True),
                temporal_lr=trial.suggest_float("temporal_lr", 1e-5, 1e-3, log=True),
            )
            m, s = run_cfg(cfg, "trial %d" % trial.number)
            store[trial.number] = (m, cfg)
            return s

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.trials)
        best_num = study.best_trial.number
        model, best_cfg = store[best_num]
        best_dev = study.best_value
        print("\nbest trial %d  DEV %.4f  %s" % (best_num, best_dev, best_cfg))

    # ---- SEALED: opened once, after everything is frozen -------------------
    print("\n%s\nSEALED SET -- opened once, all choices frozen\n%s" % ("=" * 70, "=" * 70))
    ml_sealed = ml_score(model, sealed, cache, device)
    ml_dev = ml_score(model, dev, cache, device)
    print("%-12s %9s %9s %9s" % ("vessel", "ML", "physics", "delta"))
    for n in sealed:
        print("%-12s %9.4f %9.4f %+9.4f" % (n, ml_sealed[n], base[n], ml_sealed[n] - base[n]))
    print()
    for lbl, mlv, names in (("DEV", ml_dev, dev), ("SEALED", ml_sealed, sealed)):
        m = [mlv[n] for n in names]
        b = [base[n] for n in names]
        print("%-7s ML mean %.4f median %.4f | physics mean %.4f median %.4f | delta %+.4f"
              % (lbl, np.mean(m), np.median(m), np.mean(b), np.median(b),
                 np.mean(m) - np.mean(b)))
    verdict = np.mean([ml_sealed[n] for n in sealed]) - np.mean([base[n] for n in sealed])
    print("\nVERDICT: the learned corrector is %s the physics model on SEALED (%+.4f)"
          % ("BETTER than" if verdict > 0 else "WORSE than", verdict))
    Path(args.out).write_text(json.dumps(
        dict(fit=fit, dev=dev, sealed=sealed, physics=base, ml_sealed=ml_sealed,
             ml_dev=ml_dev, trials=results, best_cfg=best_cfg), indent=2, default=float),
        encoding="utf-8")
    ck = OUT / ("model_seed%d.pth" % args.seed)
    torch.save(model.state_dict(), ck)
    print("wrote %s and %s" % (args.out, ck))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
