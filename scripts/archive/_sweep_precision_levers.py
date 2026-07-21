"""Quick POST-HOC sweep of precision directions on a trained checkpoint (NO retraining).

Motivation (docs/SPECIES_LEARNING_STRATEGY.md 6.13): deployable ranking ceiling ~LOAO AUC 0.90;
deploy clot F1 ~0.45-0.63 is the AUC->F1 collapse under ~5% wall-band prevalence. Before paying
for full training runs, measure the F1 CEILING of each proposed lever on the existing model's
predicted-Mat field:

  L_default        model's own deploy clot phi (reproduces eval_deploy_clot_f1)
  L_matthr_global  threshold predicted Mat/crit at one GLOBAL oracle-calibrated cutoff
  L_matthr_vessel  per-vessel oracle threshold (CEILING for adaptive/per-geometry thresholding)
  L_geomcommit     LOAO logistic on GEOMETRY only -> commit prob (geometry commit-head ceiling)
  L_mat_x_geom     LOAO logistic on [log predMat + geometry] (commit head USING the Mat field)
  L_mat_x_geom_nbr L_mat_x_geom + neighbour predicted-Mat (closed-loop autocatalysis re-rank)

Also reports candidate-set prevalence after a geometry gate (the two-stage imbalance lever).

All F1 in the wall+Khop deploy band, at each anchor's deploy eval time. The LOAO legs are the
deployable estimate (fit on N-1 anchors, scored on the held-out one); the *_vessel/global oracle
legs are upper bounds (they peek at the held anchor's threshold) and are labelled as such.

Run:
  python scripts/_sweep_precision_levers.py
  python scripts/_sweep_precision_levers.py --ckpt outputs/biochem/biochem_gnn/species/best.pth
  python scripts/_sweep_precision_levers.py --anchors patient001,patient007
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.biochem_gnn.config import apply_deploy_env, apply_train_recipe_env, global_ckpt_path  # noqa: E402
from src.config import BiochemConfig, NodeFeat, PhysicsConfig  # noqa: E402
from src.core_physics.clot_growth_masks import resolve_ceiling_mask  # noqa: E402
from src.core_physics.clot_phi_simple import mat_si_for_gelation_from_log1p  # noqa: E402
from src.core_physics.species_deploy_rollout import (  # noqa: E402
    alloc_species_y_series,
    deploy_fimat_log_init,
    pin_species_block,
    reset_species_rollout_flow_cache,
)
from src.core_physics.species_pushforward_continuous import (  # noqa: E402
    band_speed_at_time,
    bind_band_geometry,
    deploy_eval_time_index,
    discover_biochem_anchors,
    load_continuous_bundle,
    model_vel_decay_alphas,
    predict_continuous_step_delta,
    pushforward_log_state_step,
    resolve_deploy_eval_time_index,
    train_deploy_eval_flow_source,
)
from src.core_physics.species_pushforward_gnn import build_band_base_features  # noqa: E402
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time, rollout_t0_clot_phi  # noqa: E402
from src.training.biochem_species_scope import scatter_log_state_to_species_block  # noqa: E402
from src.utils import species_channels as sc  # noqa: E402
from src.utils.kinematics_inference import load_kinematics_predictor, resolve_kinematics_checkpoint  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402

ANCHOR_DIR = get_project_root() / "data/processed/graphs_biochem_anchors"
OUT = get_project_root() / "outputs/reports/comsol_validation/precision_levers_sweep.json"
MAT_Y = sc.y_index("Mat")  # column in full species block series


# ----------------------------------------------------------------- geometry feats
def _sym_edges(edge_index):
    return (torch.cat([edge_index[0], edge_index[1]]), torch.cat([edge_index[1], edge_index[0]]))


def _nbr_mean(values, row, col, n):
    deg = torch.zeros(n, dtype=torch.float64)
    deg.index_add_(0, row, torch.ones(row.numel(), dtype=torch.float64))
    acc = torch.zeros(n, dtype=torch.float64)
    acc.index_add_(0, row, values[col])
    return acc / deg.clamp(min=1.0)


def _geom_feats(d, dev):
    x = d.x.cpu()
    n = int(d.num_nodes)
    row, col = _sym_edges(d.edge_index.cpu())
    sdf = x[:, NodeFeat.SDF].reshape(-1).double()
    width = x[:, NodeFeat.WIDTH_ND].reshape(-1).double()
    expansion = _nbr_mean(width, row, col, n) - width
    expansion_2hop = _nbr_mean(expansion, row, col, n)
    wn = x[:, NodeFeat.WALL_NORMAL].double()
    dot = (wn[row] * wn[col]).sum(dim=1)
    cacc = torch.zeros(n, dtype=torch.float64)
    cacc.index_add_(0, row, (1.0 - dot))
    deg = torch.zeros(n, dtype=torch.float64)
    deg.index_add_(0, row, torch.ones(row.numel(), dtype=torch.float64))
    wall_curv1 = cacc / deg.clamp(min=1.0)
    wall_curv2 = _nbr_mean(wall_curv1, row, col, n)
    feats = torch.stack([sdf, width, expansion, expansion_2hop, wall_curv1, wall_curv2], dim=1)
    return feats.cpu().numpy()  # [N, 6]


# ----------------------------------------------------------------- rollout capture
@torch.no_grad()
def _capture(model, data, static, phys, bio, device, kine, flow_eval):
    """Faithful copy of eval_deploy_clot_f1 rollout, returning predicted Mat SI + default phi."""
    model.eval()
    bind_band_geometry(model, static)
    node_idx = static["node_idx"]
    n_times = int(data.y.shape[0])
    t_eval = resolve_deploy_eval_time_index(n_times, time_index=deploy_eval_time_index(n_times))
    out = alloc_species_y_series(data, device)
    log_state = deploy_fimat_log_init(data, device, node_idx)
    vel_alphas = model_vel_decay_alphas(model)
    pos_band = static.get("pos_band")
    for t in range(n_times):
        sp = pin_species_block(data, t, device)
        sp = scatter_log_state_to_species_block(sp, log_state, node_idx)
        out[t, :, sc.SPECIES_BLOCK] = sp.clamp(min=0.0)
        if t >= n_times - 1:
            break
        spd = band_speed_at_time(data, t + 1, device, node_idx)
        pred_delta = predict_continuous_step_delta(
            model, static["base_feats"], static["edge_index"], log_state,
            training=False, pos_band=pos_band, time_index=t + 1,
        )
        log_state = pushforward_log_state_step(
            log_state, pred_delta, straight_through=False, wall_speed=spd, vel_decay_alphas=vel_alphas,
        )
    from src.core_physics.t0_rung_config import RUNG2_GAMMA_MODE, t0_rung2_env

    with t0_rung2_env():
        traj = rollout_t0_clot_phi(
            data, phys, bio, device, gamma_mode=RUNG2_GAMMA_MODE, flow_source=flow_eval,
            pred_species_series=out, nucleation=True, nucleation_hops=1,
        )
    phi_pred = traj[t_eval]["phi"].reshape(-1).detach().cpu().numpy()
    phi_gt = gt_clot_phi_at_time(data, t_eval, phys, device).reshape(-1).detach().cpu().numpy()
    mat_log = out[t_eval, :, MAT_Y]
    mat_si = mat_si_for_gelation_from_log1p(mat_log, bio).reshape(-1).detach().cpu().numpy()
    return dict(t_eval=int(t_eval), phi_pred=phi_pred, phi_gt=phi_gt, mat_si=mat_si)


# ----------------------------------------------------------------- metrics / fits
def _f1(pred_bool, gt_bool):
    tp = float(np.logical_and(pred_bool, gt_bool).sum())
    fp = float(np.logical_and(pred_bool, ~gt_bool).sum())
    fn = float(np.logical_and(~pred_bool, gt_bool).sum())
    p = tp / max(tp + fp, 1e-9)
    r = tp / max(tp + fn, 1e-9)
    return 2 * p * r / max(p + r, 1e-9), p, r


def _best_thr(score, gt, grid):
    best = (-1.0, 0.0)
    for thr in grid:
        f1, _, _ = _f1(score >= thr, gt)
        if f1 > best[0]:
            best = (f1, thr)
    return best


def _logit_fit(X, y, steps=500, lr=0.05):
    Xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    pos = float(yt.sum())
    pw = torch.tensor(max((len(yt) - pos) / max(pos, 1.0), 1.0), dtype=torch.float64)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        z = Xt @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, yt, pos_weight=pw)
        (loss + 1e-3 * (w * w).sum()).backward()
        opt.step()
    return w.detach().numpy(), float(b.detach().numpy()[0])


def _standardize(train, *mats):
    mu = train.mean(axis=0, keepdims=True)
    sd = train.std(axis=0, keepdims=True)
    sd[sd < 1e-9] = 1.0
    return [(m - mu) / sd for m in (train, *mats)]


def main() -> int:
    ap = argparse.ArgumentParser(description="Post-hoc precision-lever sweep (no retraining)")
    ap.add_argument("--ckpt", default="", help="checkpoint (default: global locked species/best.pth)")
    ap.add_argument("--anchors", default="")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    device = require_cuda_device()
    apply_train_recipe_env(force=True)
    ckpt = Path(args.ckpt) if args.ckpt.strip() else global_ckpt_path()
    if not ckpt.is_absolute():
        ckpt = get_project_root() / ckpt
    anchors = (
        [a.strip() for a in args.anchors.split(",") if a.strip()]
        if args.anchors.strip() else discover_biochem_anchors(ANCHOR_DIR)
    )
    print(f"[i] ckpt={ckpt}")
    print(f"[i] anchors={anchors}")

    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    meta = dict(payload.get("meta") or {})
    scope = meta.get("pushforward_species_scope") or meta.get("species_scope")
    if scope:
        os.environ["BIOCHEM_PUSHFORWARD_SPECIES_SCOPE"] = str(scope)
    if meta.get("dual_head") is not None:
        os.environ["SPECIES_CONTINUOUS_DUAL_HEAD"] = "1" if bool(meta["dual_head"]) else "0"
    bundle = load_continuous_bundle(ckpt, device=device, quiet=True)
    if bundle is None:
        print(f"[ERR] could not load bundle: {ckpt}")
        return 1
    model = bundle.model
    wall_hops = int(meta.get("wall_hops", 3))
    kine = load_kinematics_predictor(str(resolve_kinematics_checkpoint()), device, phys_cfg=PhysicsConfig(phase="kinematics"))
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    crit = float(bio.viscosity_mat_crit)
    flow_eval = train_deploy_eval_flow_source()

    # capture per-anchor predicted-Mat fields + geometry, restricted to wall+Khop band
    rows: dict[str, dict] = {}
    for anc in anchors:
        reset_species_rollout_flow_cache()
        data = torch.load(ANCHOR_DIR / f"{anc}.pt", map_location=device, weights_only=False)
        static = build_band_base_features(data, kine, device, wall_hops=wall_hops)
        static["n_times"] = int(data.y.shape[0])
        env_snap = {k: os.environ.get(k) for k in ("T0_R4_FLOW_SOURCE", "SPECIES_ROLLOUT_VEL_SOURCE")}
        apply_deploy_env(overrides={"T0_R4_FLOW_SOURCE": flow_eval})
        cap = _capture(model, data, static, phys, bio, device, kine, flow_eval)
        for k, v in env_snap.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        band = resolve_ceiling_mask(data, device, bio, ceiling_hops=wall_hops).reshape(-1).bool().cpu().numpy()
        geom = _geom_feats(data, device)  # [N,6]
        row, col = _sym_edges(data.edge_index.cpu())
        nbr_mat = _nbr_mean(torch.tensor(cap["mat_si"], dtype=torch.float64), row, col, int(data.num_nodes)).numpy()
        b = band
        rows[anc] = dict(
            t_eval=cap["t_eval"],
            gt=(cap["phi_gt"][b] > 0.5),
            phi=(cap["phi_pred"][b] > 0.5),
            mat_score=(cap["mat_si"][b] / crit),
            geom=geom[b],
            nbr_mat=np.log1p(np.clip(nbr_mat[b], 0, None)),
        )
        f1d, pd, rd = _f1(rows[anc]["phi"], rows[anc]["gt"])
        prev = float(rows[anc]["gt"].mean())
        print(f"  {anc}: t={cap['t_eval']} nband={b.sum()} clot_prev={prev:.3f} default_F1={f1d:.3f} (P={pd:.2f} R={rd:.2f})")

    names = list(rows.keys())
    grid = np.geomspace(0.05, 20.0, 60)  # multiples of crit

    def mean_default():
        return float(np.mean([_f1(rows[a]["phi"], rows[a]["gt"])[0] for a in names]))

    # L_matthr_global: one global threshold on Mat/crit (pooled oracle)
    def mean_matthr_global():
        def pooled_f1(thr):
            return float(np.mean([_f1(rows[a]["mat_score"] >= thr, rows[a]["gt"])[0] for a in names]))
        best = max(grid, key=pooled_f1)
        return pooled_f1(best), float(best)

    # L_matthr_vessel: per-anchor oracle threshold (CEILING for adaptive thresholding)
    def mean_matthr_vessel():
        vals = [_best_thr(rows[a]["mat_score"], rows[a]["gt"], grid)[0] for a in names]
        return float(np.mean(vals))

    # LOAO logistic helper over band nodes; returns mean held-out F1 (global thr fit on train)
    def loao_logit(feature_fn):
        held_f1 = []
        for held in names:
            tr = [a for a in names if a != held]
            Xtr = np.concatenate([feature_fn(a) for a in tr], axis=0)
            ytr = np.concatenate([rows[a]["gt"].astype(np.float64) for a in tr], axis=0)
            Xtr_s, = _standardize(Xtr)
            w, b = _logit_fit(Xtr_s, ytr)
            mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True); sd[sd < 1e-9] = 1.0
            # pick threshold on train probs (transferable), apply to held
            ptr = 1 / (1 + np.exp(-(((Xtr - mu) / sd) @ w + b)))
            tgrid = np.linspace(0.05, 0.95, 37)
            bestt = max(tgrid, key=lambda t: float(np.mean(
                [_f1(1 / (1 + np.exp(-(((feature_fn(a) - mu) / sd) @ w + b))) >= t, rows[a]["gt"])[0] for a in tr])))
            Xhe = feature_fn(held)
            phe = 1 / (1 + np.exp(-(((Xhe - mu) / sd) @ w + b)))
            held_f1.append(_f1(phe >= bestt, rows[held]["gt"])[0])
        return float(np.mean(held_f1))

    geom_fn = lambda a: rows[a]["geom"]
    matgeom_fn = lambda a: np.concatenate([np.log1p(np.clip(rows[a]["mat_score"], 0, None))[:, None], rows[a]["geom"]], axis=1)
    matgeomnbr_fn = lambda a: np.concatenate([
        np.log1p(np.clip(rows[a]["mat_score"], 0, None))[:, None], rows[a]["geom"], rows[a]["nbr_mat"][:, None]], axis=1)

    # candidate prevalence after a geometry gate (two-stage imbalance lever):
    # gate = top-50% by a quick pooled geometry-commit logistic; report prevalence inside gate
    def geom_gate_prevalence():
        Xall = np.concatenate([rows[a]["geom"] for a in names], axis=0)
        yall = np.concatenate([rows[a]["gt"].astype(np.float64) for a in names], axis=0)
        Xs, = _standardize(Xall)
        w, b = _logit_fit(Xs, yall)
        mu = Xall.mean(0, keepdims=True); sd = Xall.std(0, keepdims=True); sd[sd < 1e-9] = 1.0
        prevs_raw, prevs_gate = [], []
        for a in names:
            p = 1 / (1 + np.exp(-(((rows[a]["geom"] - mu) / sd) @ w + b)))
            keep = p >= np.quantile(p, 0.5)
            prevs_raw.append(float(rows[a]["gt"].mean()))
            prevs_gate.append(float(rows[a]["gt"][keep].mean()) if keep.any() else float("nan"))
        return float(np.mean(prevs_raw)), float(np.nanmean(prevs_gate))

    results = {}
    results["L_default"] = dict(mean_f1=mean_default(), kind="model deploy phi (reference)")
    g_f1, g_thr = mean_matthr_global()
    results["L_matthr_global"] = dict(mean_f1=g_f1, thr_x_crit=g_thr, kind="ORACLE global thr (upper bound)")
    results["L_matthr_vessel"] = dict(mean_f1=mean_matthr_vessel(), kind="ORACLE per-vessel thr (upper bound)")
    results["L_geomcommit"] = dict(mean_f1=loao_logit(geom_fn), kind="LOAO geometry-only commit head (deployable)")
    results["L_mat_x_geom"] = dict(mean_f1=loao_logit(matgeom_fn), kind="LOAO Mat+geometry commit head (deployable)")
    results["L_mat_x_geom_nbr"] = dict(mean_f1=loao_logit(matgeomnbr_fn), kind="LOAO Mat+geom+nbrMat (deployable)")
    prev_raw, prev_gate = geom_gate_prevalence()
    results["candidate_prevalence"] = dict(raw=prev_raw, after_geom_gate_top50=prev_gate)

    base = results["L_default"]["mean_f1"]
    print("\n==================== PRECISION-LEVER SWEEP (mean deploy clot F1, band) ====================")
    print(f"{'lever':22}{'mean_F1':>9}{'d_vs_default':>14}   kind")
    for k in ("L_default", "L_matthr_global", "L_matthr_vessel", "L_geomcommit", "L_mat_x_geom", "L_mat_x_geom_nbr"):
        r = results[k]
        print(f"{k:22}{r['mean_f1']:9.3f}{r['mean_f1']-base:+14.3f}   {r['kind']}")
    print(f"\n[i] clot prevalence in band: raw={prev_raw:.3f} -> after geom-gate(top50%)={prev_gate:.3f} "
          f"({prev_gate/max(prev_raw,1e-9):.2f}x)")
    print("[i] ORACLE legs peek at held-out threshold (upper bounds); LOAO legs are deployable estimates.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dict(
        ckpt=str(ckpt), anchors=names, meta_scope=scope, wall_hops=wall_hops,
        per_anchor={a: dict(t_eval=rows[a]["t_eval"], n_band=int(rows[a]["gt"].size),
                            clot_prev=float(rows[a]["gt"].mean())) for a in names},
        results=results,
    ), indent=2), encoding="utf-8")
    print(f"[save] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
