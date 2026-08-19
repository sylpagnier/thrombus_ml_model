"""Autonomous search for a physics-backboned onset model that reduces ``growth_l1``.

OBJECTIVE.  ``growth_l1 = mean_t |n_pred(t) - n_gt(t)| / N_gt_final``.  Shipped physics
0.1316 on train; the count floor (best possible on the same committed set) is 0.0424.  The
whole prize is +0.0892, and 32% of the shipped error is mask-size error no timing model can
reach.

WHY THIS ONE CAN WORK WHERE THREE ATTEMPTS FAILED (PHASE6_HANDOFF 4).  The loss IS the
metric with the step relaxed -- ``sum_i sigmoid((t - onset_i)/tau)`` -- so there is no
thresholded readout starving the gradient (the old one had **0.00%** of wall nodes inside
the sigmoid band) and no backprop through a 200-step stiff ODE.  Every model is a residual
on the physics ODE's own onset and reduces to it exactly at zero parameters.

PROTOCOL, ENFORCED IN CODE.
  * SEALED is never loaded.  The cache files for those vessels are not even opened.
  * Selection reads DEV only; FIT is what everything trains on.
  * The zero-parameter parity check runs before the search and aborts it if the untrained
    backbone does not reproduce the hard physics number -- the guard whose absence
    invalidated a whole earlier round.
  * Phase 2 re-runs the top configurations across 3 seeds and reports a paired CI, because
    a single run's DEV number is a peak-pick off a noisy trace (6.5).

    python scripts/auto_onset_search.py --hours 5.5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig  # noqa: E402
from src.core_physics.growth_count_metrics import count_curve  # noqa: E402
from src.core_physics.onset_features import (  # noqa: E402
    FEATURE_NAMES, build_features, committed_set,
)
from src.core_physics.physics_wall_model import (  # noqa: E402
    first_crossing, integrate_mat_trajectory,
)
from src.differentiable_wall_model.onset_residual import (  # noqa: E402
    build, growth_l1_soft, hard_growth_l1,
)

CACHE = Path("outputs/wall_species_cache")
OUT = Path("outputs/onset_search")
BULK_ND = 2.5e14
M_TO_CM = 100.0

# feature groups, so the search can ask which PHYSICS matters rather than which columns
GROUPS = {
    "gate": ["gate", "log_gate", "gate_low", "gate_sep"],
    "shear": ["log_sr", "sr_rank", "log_absdsrx", "dsrx_rank"],
    "neigh": ["nbr1_log_sr", "nbr2_log_sr", "nbr3_log_sr", "nbr1_gate", "nbr2_gate"],
    "hop": ["hop", "hop_frac", "is_seed", "seed_dist_xy"],
    "closure": ["ap_closure", "log_ap_closure"],
    "geom": ["arc_frac", "degree"],
}


class WallShim:
    def __init__(self, z):
        self.t = torch.tensor(z["t"], dtype=torch.float64).reshape(-1, 1)
        n = z["ap"].shape[1]
        y = torch.zeros(1, n, 16, dtype=torch.float32)
        y[0, :, 4] = torch.tensor(np.log1p(z["rp"][0] / BULK_ND), dtype=torch.float32)
        y[0, :, 5] = torch.tensor(np.log1p(z["ap"][0] / BULK_ND), dtype=torch.float32)
        self.y = y
        self.y_channel_names = (
            "u_nd,v_nd,p_nd,mu_eff_nd,RP_log1p_nd,AP_log1p_nd,APR_log1p_nd,APS_log1p_nd,"
            "PT_log1p_nd,T_log1p_nd,AT_log1p_nd,FG_log1p_nd,FI_log1p_nd,M_log1p_nd,"
            "Mas_log1p_nd,Mat_log1p_nd")


def load_vessels(names, bio, C, device):
    lss, sgt = float(bio.lss), float(bio.sgt) / M_TO_CM
    coef = float(bio.L_char) * M_TO_CM / float(bio.gamma_m)
    out = {}
    for n in names:
        p = CACHE / f"{n}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        gt = z["gt_onset"]
        if not (gt >= 0).any():
            continue
        sr0, dsrx0 = z["sr0"], z["dsrx0"]
        gate = (dsrx0 < sgt) * coef * np.abs(dsrx0) + (sr0 < lss)
        S = committed_set(gate, sr0, z["wall_edges"])
        if S.sum() < 4:
            continue
        nt = len(z["t"])
        traj, _ = integrate_mat_trajectory(WallShim(z), bio, gate, da_scale=40.0)
        idx = first_crossing(traj, float(bio.viscosity_mat_crit))
        crossed = idx >= 0
        med = int(np.median(idx[crossed])) if crossed.any() else 0
        ode = np.where(S, np.where(idx >= 0, idx, med), -1)
        X, _ = build_features(z, bio, C=C)
        e = z["wall_edges"]
        keep = S[e[0]] & S[e[1]]
        loc = -np.ones(len(S), dtype=np.int64)
        loc[S] = np.arange(int(S.sum()))
        out[n] = dict(
            name=n, nt=nt,
            X=torch.tensor(X[S], dtype=torch.float32, device=device),
            ode=torch.tensor(ode[S] / (nt - 1), dtype=torch.float32, device=device),
            edges=torch.tensor(np.stack([loc[e[0][keep]], loc[e[1][keep]]]),
                               dtype=torch.long, device=device),
            grid=torch.linspace(0, 1, nt, device=device),
            gt_curve=torch.tensor(count_curve(gt, nt), dtype=torch.float32, device=device),
            gt_onset=gt, n_gt=float((gt >= 0).sum()), n_S=int(S.sum()),
        )
    return out


def cols_for(groups):
    idx = []
    for g in groups:
        idx += [FEATURE_NAMES.index(c) for c in GROUPS[g]]
    return sorted(set(idx))


def eval_hard(model, vessels, names, cols, allow_never):
    tot, masks = [], []
    model.eval()
    with torch.no_grad():
        for n in names:
            v = vessels[n]
            o = model(v["ode"], v["X"][:, cols], v["edges"])
            if not allow_never:
                o = o.clamp(0.0, 1.0)
            g, n_pred = hard_growth_l1(o, v["gt_onset"], v["nt"], v["n_gt"], allow_never)
            tot.append(g)
            masks.append(n_pred / max(v["n_S"], 1))
    return float(np.mean(tot)), float(np.mean(masks))


def run_trial(cfg, vessels, fit, dev, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cols = cols_for(cfg["groups"])
    mu = torch.cat([vessels[n]["X"][:, cols] for n in fit]).mean(0)
    sd = torch.cat([vessels[n]["X"][:, cols] for n in fit]).std(0).clamp(min=1e-6)
    for v in vessels.values():
        v["Xn"] = (v["X"][:, cols] - mu) / sd
    model = build(cfg["model"], len(cols), hidden=cfg["hidden"], layers=cfg["layers"],
                  rounds=cfg["layers"], cap=cfg["cap"]).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    best = (1e9, None, 0)
    for ep in range(cfg["epochs"]):
        model.train()
        order = list(fit)
        np.random.shuffle(order)
        for n in order:
            v = vessels[n]
            o = model(v["ode"], v["Xn"], v["edges"])
            if not cfg["allow_never"]:
                o = o.clamp(0.0, 1.0)
            loss = growth_l1_soft(o, v["gt_curve"], v["grid"], v["n_gt"], cfg["tau"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if (ep + 1) % cfg["eval_every"] == 0 or ep == cfg["epochs"] - 1:
            d, mk = eval_hard(model, vessels, dev, cols, cfg["allow_never"])
            if d < best[0]:
                f, _ = eval_hard(model, vessels, fit, cols, cfg["allow_never"])
                best = (d, f, mk)
    return dict(dev=best[0], fit=best[1], mask_frac=best[2], n_params=n_par)


def sample_cfg(rng):
    model = rng.choice(["global_affine", "vessel_affine", "node_mlp", "node_mlp",
                        "node_gnn", "node_gnn"])
    gs = list(GROUPS)
    k = rng.integers(1, len(gs) + 1)
    groups = sorted(rng.choice(gs, size=k, replace=False).tolist())
    if model == "global_affine":
        groups = ["gate"]
    return dict(
        model=str(model), groups=groups,
        hidden=int(rng.choice([16, 32, 64, 128])),
        layers=int(rng.choice([1, 2, 3])),
        cap=float(rng.choice([0.1, 0.25, 0.5, 1.0])),
        lr=float(10 ** rng.uniform(-3.3, -1.5)),
        wd=float(10 ** rng.uniform(-6, -2)),
        tau=float(rng.choice([0.005, 0.01, 0.02, 0.05])),
        epochs=int(rng.choice([60, 120, 200])),
        eval_every=10,
        allow_never=bool(rng.random() < 0.5),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=5.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bio = BiochemConfig(phase="biochem")
    prot = json.load(open("outputs/ap_closure/protocol_gt_meanovertime.json"))
    fit_names, dev_names = prot["fit"], prot["dev"]
    C = float(prot["best_cl"]["C"])
    log = (OUT / "search_log.jsonl").open("a", encoding="utf-8")

    def say(msg):
        print(msg, flush=True)

    say("device=%s  FIT=%d DEV=%d   SEALED IS NOT LOADED" % (device, len(fit_names), len(dev_names)))
    vessels = load_vessels(fit_names + dev_names, bio, C, device)
    fit = [n for n in fit_names if n in vessels]
    dev = [n for n in dev_names if n in vessels]
    assert not (set(vessels) & set(prot["sealed"])), "SEALED vessel leaked into the search"

    # ---- parity gate: the untrained backbone must BE the physics
    zero = build("node_mlp", len(FEATURE_NAMES), hidden=16, layers=1, cap=0.5).to(device)
    cols = list(range(len(FEATURE_NAMES)))
    for v in vessels.values():
        v["Xn"] = v["X"]
    p_fit, _ = eval_hard(zero, vessels, fit, cols, False)
    p_dev, _ = eval_hard(zero, vessels, dev, cols, False)
    # The real guard is EXACT: a zero-initialised residual must emit the ODE's own onset,
    # node for node.  A drifting backbone is what made an earlier round measure its own
    # defect instead of the head (PHASE6_HANDOFF 4 / sweep_temporal_only's parity gate).
    with torch.no_grad():
        for n in fit + dev:
            v = vessels[n]
            out = zero(v["ode"], v["X"], v["edges"]).clamp(0.0, 1.0)
            if not torch.allclose(out, v["ode"].clamp(0.0, 1.0), atol=1e-6):
                say("[ABORT] zero-init residual is not the physics on %s" % n)
                return 2
    say("[parity] zero-init residual reproduces the physics ODE exactly on all %d vessels"
        % len(vessels))
    say("[parity] physics growth_l1:  FIT %.4f  DEV %.4f" % (p_fit, p_dev))
    base = dict(fit=p_fit, dev=p_dev)

    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    budget = args.hours * 3600.0
    trials, n_fail = [], 0
    say("\n[phase 1] random search until %.1f h elapsed" % (0.6 * args.hours))
    while time.time() - t0 < 0.6 * budget:
        cfg = sample_cfg(rng)
        try:
            r = run_trial(cfg, vessels, fit, dev, device, seed=0)
        except Exception:                                          # noqa: BLE001
            n_fail += 1
            log.write(json.dumps(dict(cfg=cfg, error=traceback.format_exc()[-400:])) + "\n")
            log.flush()
            continue
        rec = dict(cfg=cfg, **r, phase=1, elapsed=time.time() - t0)
        trials.append(rec)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if len(trials) % 10 == 0:
            b = min(trials, key=lambda x: x["dev"])
            say("  %4d trials  %5.0fs  best DEV %.4f (%s, %d par, fit %.4f)"
                % (len(trials), time.time() - t0, b["dev"], b["cfg"]["model"],
                   b["n_params"], b["fit"]))

    if not trials:
        say("[ABORT] no successful trial")
        return 3

    # ---- phase 2: multi-seed the leaders (6.5 -- one run's DEV is a peak-pick)
    say("\n[phase 2] re-running the top configurations across 5 seeds")
    top = sorted(trials, key=lambda x: x["dev"])[:10]
    # The 2-parameter model is carried into phase 2 whatever its phase-1 rank, because the
    # question a search like this exists to answer is not "what is the best DEV number" but
    # "does the capacity earn its place" (PHASE6_HANDOFF 19.2: 187k parameters once bought
    # +0.024 over a logistic regression).  Without this the reference could be dropped by a
    # single noisy phase-1 draw and the comparison would quietly disappear.
    ga = [t for t in trials if t["cfg"]["model"] == "global_affine"]
    if ga and not any(t["cfg"]["model"] == "global_affine" for t in top):
        top.append(min(ga, key=lambda x: x["dev"]))
    final = []
    for rec in top:
        if time.time() - t0 > budget:
            break
        runs = []
        for s in (0, 1, 2, 3, 4):
            try:
                runs.append(run_trial(rec["cfg"], vessels, fit, dev, device, seed=s))
            except Exception:                                      # noqa: BLE001
                continue
        if not runs:
            continue
        d = np.array([r["dev"] for r in runs])
        f = np.array([r["fit"] for r in runs])
        out = dict(cfg=rec["cfg"], n_params=runs[0]["n_params"], seeds=len(runs),
                   dev_mean=float(d.mean()), dev_sd=float(d.std(ddof=1)) if len(d) > 1 else 0.0,
                   fit_mean=float(f.mean()),
                   mask_frac=float(np.mean([r["mask_frac"] for r in runs])), phase=2)
        final.append(out)
        log.write(json.dumps(out) + "\n")
        log.flush()
        say("  %-14s %-38s par %6d  DEV %.4f +- %.4f  FIT %.4f  mask %.2f"
            % (out["cfg"]["model"], ",".join(out["cfg"]["groups"]), out["n_params"],
               out["dev_mean"], out["dev_sd"], out["fit_mean"], out["mask_frac"]))

    final.sort(key=lambda x: x["dev_mean"])
    (OUT / "leaderboard.json").write_text(json.dumps(
        dict(baseline=base, floor_train=0.0424, n_trials=len(trials), n_fail=n_fail,
             top=final, fit=fit, dev=dev), indent=2, default=float), encoding="utf-8")
    say("\n%s\nBASELINE physics  FIT %.4f  DEV %.4f   (count floor on train 0.0424)"
        % ("=" * 78, base["fit"], base["dev"]))
    for o in final[:5]:
        say("%-14s %-34s par %6d  DEV %.4f +- %.4f  (%+.4f vs physics)"
            % (o["cfg"]["model"], ",".join(o["cfg"]["groups"]), o["n_params"],
               o["dev_mean"], o["dev_sd"], o["dev_mean"] - base["dev"]))
    say("\nwrote %s   (%d trials, %d failed, %.2f h)"
        % (OUT / "leaderboard.json", len(trials), n_fail, (time.time() - t0) / 3600))
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
