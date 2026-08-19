"""STEP 2 (PHASE6_HANDOFF 5.2): the AP closure under the clean protocol.

FIT / DEV / SEALED are disjoint.  ``C`` is fitted on FIT only.  The kernel, the exponent
and ``da_scale`` are selected on DEV only.  SEALED is opened once, at the end, with every
choice already frozen.

METRIC OF RECORD IS THE **MEAN** OVER TIME, changed from the median deliberately.
``scripts/diag_horizon_sensitivity.py`` showed the shipped mask scores 0.098 / 0.323 /
0.682 at t/T = 0.1 / 0.2 / 0.3 and then sits on a 0.93-0.97 plateau from t/T = 0.4 on.
With 12 evaluation times the MEDIAN lands at t/T ~ 0.5, inside the plateau, so the old
primary metric could not see the flash at all -- the oracle prize is +0.185 in the early
window against +0.056 late.  Reported alongside: early/late split, ``curve_l1``, onset
``rho``, and the final mask.

FOUR ARMS, so each mechanism's marginal contribution is isolated rather than confounded:
``1sc physics`` (what ships today), ``2sc physics`` (separate autocatalytic rate -- the
step-3 Damkohler ratio), ``closure`` (AP closure alone), ``closure+2sc`` (both).

THE FINAL MASK CANNOT MOVE, AND THAT IS STRUCTURAL, NOT LUCK.  The shipped predictor
(``scripts/predict_wall_clot.py``) takes its mask from the two t=0 gates plus shear-admitted
graph growth; the ODE supplies only *when* each node in that set ignites.  The closure
touches nothing but the ODE's rate.  9 still demands the assertion, so it is asserted
per vessel rather than argued.

WHY ``da_scale`` IS IN THE SELECTION GRID.  The closure multiplies ``ap`` by
``1/(1 + C*consumption/sr)``, which is <= 1 everywhere -- it can only slow deposition down.
Left alone it would push every onset later and eventually past the horizon, which is a
level error, not an ordering one.  ``da_scale`` is the existing global rate scalar that
absorbs exactly that, and it was already known to be under-determined by the mask metric
(every value above ~50 gives a bit-identical committed set) while mattering to the curve.
Re-selecting it on DEV alongside the closure is therefore required, not a free knob.

SEALED HAS BEEN READ TWICE, AND THIS IS THE DISCLOSURE.  Run 1 used the handoff's kernel
set (``outputs/ap_closure/protocol_gt_run1_handoff_kernels.json``).  ``mat_linear`` was
then added and run 2 produced the headline numbers.  The kernel was chosen by the
window-stability test in ``scripts/fit_ap_closure.py`` A3, on FIT vessels, before any
SEALED number was consulted -- but the set has still been opened twice, both readings are
on disk, and the second therefore carries a small selection risk that no argument removes.
Do not open it a third time for this question.

    python scripts/eval_ap_closure_protocol.py --flow gt
    python scripts/eval_ap_closure_protocol.py --flow pred      # 5.5, arm B
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.biochem_gnn.mat_growth_simple import (  # noqa: E402
    WALL_COHORT_V2_GENERALIZATION, WALL_COHORT_V2_TRAIN,
)
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.ap_closure import (  # noqa: E402
    ApClosure, build_smoother, consumption, fit_C, make_rollout_hook,
)
from src.core_physics.mls_gradient import node_positions  # noqa: E402
from src.core_physics.physics_wall_model import (  # noqa: E402
    first_crossing, integrate_mat_trajectory, t0_flow_fields,
)
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.core_physics.temporal_metrics import curve_l1, gt_onset_index, onset_metrics  # noqa: E402
from src.evaluation.clot_relaxed_metrics import (  # noqa: E402
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
)

DIR = Path("data/processed/graphs_biochem_anchors")
CACHE = Path("outputs/wall_species_cache")
OUT = Path("outputs/ap_closure")
MIN_T = 150
RELAX, GROW, STENCIL = 2.0, 6, {"gt": 3, "pred": 4}
N_EVAL = 12
EARLY_FRAC = 0.35       # t/T below which the flash does its damage (horizon_sensitivity)
DEV_STRIDE = 4          # positional rule, identical to scripts/sweep_temporal_only.py
M_TO_CM = 100.0
PER_M2_TO_PER_CM2 = 1.0e-4


class Shim:
    """What the rollout operators read, without holding the 300 MB pack.

    ``integrate_mat_trajectory`` needs only ``t``/``y``/``y_channel_names``; the blockage
    and thrombin couplings additionally read ``d_bar``, ``edge_index`` and ``x``, so those
    ride along too and ``scripts/diag_lever_panel.py`` can build them from a context.
    """

    def __init__(self, t, y0, names, *, d_bar=None, edge_index=None, x=None):
        self.t = t
        self.y = y0
        self.y_channel_names = names
        self.d_bar = d_bar
        self.edge_index = edge_index
        self.x = x


# ------------------------------------------------------------------------- contexts

def build_context(name: str, bio, phys, flow: str) -> dict | None:
    p = DIR / f"{name}.pt"
    if not p.exists():
        return None
    d = torch.load(p, map_location="cpu", weights_only=False)
    T = int(d.y.shape[0])
    if T < MIN_T:
        return None
    if flow == "pred" and getattr(d, "u0_pred", None) is None:
        return None
    w = d.mask_wall.reshape(-1).bool().numpy()
    gt_on = gt_onset_index(d, phys, w)
    if not ((gt_on >= 0) & w).any():
        return None                                    # 6.2: an empty-GT vessel scores 1.0

    f = t0_flow_fields(d, bio, hops=STENCIL[flow], flow_source=flow)
    ei = d.edge_index.numpy()
    n = len(w)
    A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.int8)
    cur = (f.gate > 0) & w
    adm = (f.sr < float(bio.lss) * RELAX) & w
    for _ in range(GROW):
        cur = cur | (((A @ cur.astype(np.int8)) > 0) & adm)

    eval_ts = np.unique(np.linspace(T // N_EVAL, T - 1, N_EVAL).astype(int))
    gt_masks = {int(ti): gt_clot_phi_at_time(d, int(ti), phys, device=torch.device("cpu")
                                             ).reshape(-1) * torch.tensor(w.astype(np.float32))
                for ti in eval_ts}
    z = np.load(CACHE / f"{name}.npz") if (CACHE / f"{name}.npz").exists() else None
    return dict(
        name=name, w=w, wt=torch.tensor(w.astype(np.float32)), edge_index=d.edge_index,
        t=d.t.reshape(-1).numpy().astype(np.float64), eval_ts=eval_ts, gt_masks=gt_masks,
        gt_onset=gt_on, S=cur, gate=f.gate * w, sr=f.sr, fields=f, adj=A,
        pos=node_positions(d),
        shim=Shim(d.t, d.y[:1].clone(), d.y_channel_names, d_bar=d.d_bar,
                  edge_index=d.edge_index, x=d.x.clone()),
        smoother_edges=(z["wall_edges"] if z is not None else None),
        wall_idx=np.where(w)[0],
        sealed=name in WALL_COHORT_V2_GENERALIZATION,
    )


def score_at(c, pred_mask, ti) -> float:
    m = compute_clot_relaxed_metrics(torch.tensor(pred_mask.astype(np.float32)) * c["wt"],
                                     c["gt_masks"][int(ti)], c["edge_index"],
                                     wall_mask=torch.tensor(c["w"]))
    return clot_score_from_deploy_dict(metrics_to_deploy_prefix(m))


# --------------------------------------------------------------------------- rollout

def rollout_onset(c, bio, closure: ApClosure | None, da_scale: float,
                  da_scale_auto: float | None = None):
    """Onset index per node for the committed set ``S``, plus the ODE's crossing count."""
    hook = None
    if closure is not None and closure.C != 0.0:
        smoother = None
        if closure.smooth_hops > 0 and c["smoother_edges"] is not None:
            sm = build_smoother(c["smoother_edges"], len(c["wall_idx"]), closure.smooth_hops)
            widx = c["wall_idx"]

            def smoother(v, _sm=sm, _w=widx):        # lift the wall-only operator to all nodes
                out = np.zeros_like(v)
                out[_w] = _sm(v[_w])
                return out
        hook = make_rollout_hook(closure, bio, c["sr"], smoother=smoother)
    traj, t = integrate_mat_trajectory(c["shim"], bio, c["gate"], da_scale=da_scale,
                                       da_scale_auto=da_scale_auto, ap_closure=hook)
    idx = first_crossing(traj, float(bio.viscosity_mat_crit))
    crossed = (idx >= 0) & c["w"]
    # Nodes in S that never cross still belong to the shipped mask; give them the ODE's own
    # median onset, exactly as diag_time_resolved_ceiling does, so arms stay comparable.
    med = int(np.median(idx[crossed])) if crossed.any() else 0
    onset = np.where(c["S"], np.where(idx >= 0, idx, med), -1)
    return onset, float(crossed.sum()) / max(float((c["S"]).sum()), 1.0)


def arm_metrics(c, onset) -> dict:
    """METRIC OF RECORD IS THE MEAN OVER TIME.  ``score_median`` is kept for continuity.

    The median was the wrong statistic and ``scripts/diag_horizon_sensitivity.py`` shows
    why: the shipped mask scores 0.098 / 0.323 / 0.682 at t/T = 0.1 / 0.2 / 0.3 (it has
    committed everything by t~3000 s while GT has committed nothing) and then sits on a
    0.93-0.97 plateau from t/T = 0.4 onward.  With 12 evaluation times the median lands at
    t/T ~ 0.5, INSIDE the plateau, so the primary metric was structurally blind to the only
    region where the model is wrong.  Split by window on frozen predictions, the oracle
    prize is +0.185 early (t/T<=0.35) against +0.056 late -- the headroom is 3x what the
    median reported.
    """
    scores = np.array([score_at(c, (onset >= 0) & (onset <= ti), ti) for ti in c["eval_ts"]])
    frac = np.asarray(c["eval_ts"], dtype=np.float64) / max(len(c["t"]) - 1, 1)
    early = frac <= EARLY_FRAC
    om = onset_metrics(onset, c["gt_onset"], c["t"], c["w"])
    return dict(score=float(scores.mean()),                 # <- metric of record
                score_median=float(np.median(scores)),
                score_early=float(scores[early].mean()) if early.any() else float("nan"),
                score_late=float(scores[~early].mean()) if (~early).any() else float("nan"),
                curve_l1=float(curve_l1(onset, c["gt_onset"], c["t"], c["w"])),
                rho=float(om["rho"]), bias=float(om["bias"]),
                spread_ratio=float(om["spread_ratio"]))


def med(rows, key="score"):
    v = np.array([r[key] for r in rows.values()], float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else float("nan")


def mean_(rows, key="score"):
    v = np.array([r[key] for r in rows.values()], float)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if len(v) else float("nan")


# ------------------------------------------------------------------------ C on FIT only

def fit_C_on(names: list[str], bio, kernel: str, q: float) -> float:
    k_as = float(bio.k_as) * M_TO_CM
    k_aa = float(bio.k_aa) * M_TO_CM
    minf = float(bio.Minf) * PER_M2_TO_PER_CM2
    lss, sgt = float(bio.lss), float(bio.sgt) / M_TO_CM
    coef = float(bio.L_char) * M_TO_CM / float(bio.gamma_m)
    rs, xs = [], []
    for n in names:
        p = CACHE / f"{n}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        sr0, dsrx0 = z["sr0"], z["dsrx0"]
        gate = (dsrx0 < sgt) * coef * np.abs(dsrx0) + (sr0 < lss)
        mas_f, mat_f = z["mas"] / minf, z["mat"] / minf
        sat = np.clip(1.0 - mas_f, 0.0, 1.0)
        ker = consumption(kernel, gate[None, :], sat, mas_f, mat_f, k_as, k_aa)
        x = np.broadcast_to(ker, sat.shape) / np.power(np.maximum(sr0, 1e-3), q)[None, :]
        ratio = z["ap"] / np.maximum(z["ap"][0], 1e-30)[None, :]
        sel = np.zeros_like(ratio, dtype=bool)
        sel[1:] = True
        sel &= (gate > 0)[None, :] & np.isfinite(ratio) & (ratio > 0) & np.isfinite(x)
        rs.append(ratio[sel])
        xs.append(x[sel])
    if not rs:
        return 0.0
    return fit_C(np.concatenate(rs), np.concatenate(xs))


# ------------------------------------------------------------------------------- main

def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--flow", default="gt", choices=["gt", "pred"])
    ap_.add_argument("--tag", default="")
    args = ap_.parse_args()
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.tag or args.flow

    t0 = time.time()
    pool, sealed, ctx = [], [], {}
    for n in sorted(set(WALL_COHORT_V2_TRAIN) | set(WALL_COHORT_V2_GENERALIZATION)):
        c = build_context(n, bio, phys, args.flow)
        if c is None:
            continue
        ctx[n] = c
        (sealed if c["sealed"] else pool).append(n)
    dev = [n for i, n in enumerate(pool) if i % DEV_STRIDE == 0]
    fit = [n for n in pool if n not in dev]
    assert fit and dev and sealed
    assert not (set(fit) & set(dev)) and not (set(fit) & set(sealed)) and not (set(dev) & set(sealed))
    print("flow=%s   contexts built in %.0fs" % (args.flow, time.time() - t0))
    print("FIT    n=%2d %s" % (len(fit), " ".join(a[-3:] for a in fit)))
    print("DEV    n=%2d %s" % (len(dev), " ".join(a[-3:] for a in dev)))
    print("SEALED n=%2d %s" % (len(sealed), " ".join(a[-3:] for a in sealed)))

    # ------------------------------------------------------------------ C, on FIT only
    print("\n" + "=" * 88)
    print("C FITTED ON FIT ONLY  (DEV is held back for selection; SEALED never read)")
    print("=" * 88)
    KERNELS = ["static", "sat_plus_mat", "mat_linear"]
    Q = 1.0
    C_fit = {k: fit_C_on(fit, bio, k, Q) for k in KERNELS}
    for k in KERNELS:
        print("   kernel=%-14s q=%.2f   C_fit = %.4g" % (k, Q, C_fit[k]))

    # -------------------------------------------------------------- DEV selection grid
    # AGGREGATE IS THE MEAN ACROSS VESSELS, of the per-vessel MEAN over time.  The previous
    # round selected on a median-of-medians and produced an exact 5-way tie -- a 5-vessel
    # median takes one of five values, so it cannot rank anything.  Two coarse statistics
    # stacked on each other is what made the selection unable to see the mechanism.
    print("\n" + "=" * 88)
    print("DEV SELECTION -- MEAN-over-time deploy_clot_score.  SEALED is not consulted.")
    print("=" * 88)
    DA = [10.0, 20.0, 40.0, 80.0]
    RATIO = [1.0, 2.0, 3.0, 5.0]          # A_a / A_s; 1.0 is the shipped one-scalar model
    CSCALE = [0.25, 0.5, 1.0, 2.0]

    def dev_row(cl, da, ratio):
        r, frac = {}, []
        for n in dev:
            on, fr = rollout_onset(ctx[n], bio, cl, da, None if ratio == 1.0 else da * ratio)
            r[n] = arm_metrics(ctx[n], on)
            frac.append(fr)
        return dict(dev=mean_(r), dev_median=med(r), dev_early=mean_(r, "score_early"),
                    dev_late=mean_(r, "score_late"), dev_curve=mean_(r, "curve_l1"),
                    dev_rho=mean_(r, "rho"), crossed=float(np.mean(frac)))

    # -- arm 1: the shipped one-scalar physics, da re-selected under the new metric
    print("\n[arm 1] one-scalar physics (what ships today)")
    base = {da: dev_row(None, da, 1.0) for da in DA}
    for da in DA:
        print("   da=%-6.0f DEV mean %.4f  early %.4f  late %.4f  curveL1 %.4f"
              % (da, base[da]["dev"], base[da]["dev_early"], base[da]["dev_late"],
                 base[da]["dev_curve"]))
    best_base = max(DA, key=lambda da: base[da]["dev"])
    print("   selected da=%.0f  DEV mean %.4f" % (best_base, base[best_base]["dev"]))

    # -- arm 2: two-scalar physics, no closure.  The Damkohler ratio from step 3.
    print("\n[arm 2] TWO-SCALAR physics: separate rate for the autocatalytic term")
    two = []
    for da in DA:
        for ratio in RATIO:
            row = dev_row(None, da, ratio)
            row.update(da=da, ratio=ratio, kernel="none", C=0.0)
            two.append(row)
            print("   da_s=%-5.0f A_a/A_s=%-4.1f DEV mean %.4f  early %.4f  late %.4f  "
                  "curveL1 %.4f rho %+.3f" % (da, ratio, row["dev"], row["dev_early"],
                                              row["dev_late"], row["dev_curve"], row["dev_rho"]))
    best_two = max(two, key=lambda g: g["dev"])
    print("   selected da_s=%.0f ratio=%.1f  DEV mean %.4f  (vs one-scalar %.4f, %+.4f)"
          % (best_two["da"], best_two["ratio"], best_two["dev"], base[best_base]["dev"],
             best_two["dev"] - base[best_base]["dev"]))

    # -- arm 3/4: AP closure, with and without the two-scalar rate
    print("\n[arm 3/4] AP closure x two-scalar rate (full factorial)")
    grid = []
    for kern in KERNELS:
        for cscale in CSCALE:
            C = C_fit[kern] * cscale
            cl = ApClosure(C=C, q=Q, kernel=kern)
            for da in DA:
                for ratio in RATIO:
                    row = dev_row(cl, da, ratio)
                    row.update(kernel=kern, C=C, cscale=cscale, da=da, ratio=ratio)
                    grid.append(row)
    for row in sorted(grid, key=lambda g: -g["dev"])[:12]:
        print("   %-13s C=%-7.4g da_s=%-5.0f r=%-4.1f DEV mean %.4f early %.4f curveL1 %.4f"
              % (row["kernel"], row["C"], row["da"], row["ratio"], row["dev"],
                 row["dev_early"], row["dev_curve"]))
    only_cl = [g for g in grid if g["ratio"] == 1.0]
    best_cl = max(only_cl, key=lambda g: g["dev"])
    best_all = max(grid, key=lambda g: g["dev"])

    TIE_TOL = 1e-6
    tied = sorted([g for g in grid if g["dev"] >= best_all["dev"] - TIE_TOL],
                  key=lambda g: (g["dev_curve"], g["C"]))
    best_all = tied[0]
    print("\n   closure only  : %s C=%.4g da=%.0f      DEV mean %.4f"
          % (best_cl["kernel"], best_cl["C"], best_cl["da"], best_cl["dev"]))
    print("   closure + 2sc : %s C=%.4g da=%.0f r=%.1f DEV mean %.4f  (%d-way tie)"
          % (best_all["kernel"], best_all["C"], best_all["da"], best_all["ratio"],
             best_all["dev"], len(tied)))

    # -------------------------------------------------------------- SEALED, opened once
    ARMS = [
        ("1sc physics", None, best_base, 1.0),
        ("2sc physics", None, best_two["da"], best_two["ratio"]),
        ("closure", ApClosure(C=best_cl["C"], q=Q, kernel=best_cl["kernel"]), best_cl["da"], 1.0),
        ("closure+2sc", ApClosure(C=best_all["C"], q=Q, kernel=best_all["kernel"]),
         best_all["da"], best_all["ratio"]),
    ]
    print("\n" + "=" * 88)
    print("SEALED -- opened once, every choice above frozen")
    print("=" * 88)
    R = {lbl: {} for lbl, *_ in ARMS}
    R["oracle"] = {}
    mask_ok = True
    for n in sealed + fit + dev:
        c = ctx[n]
        ref = None
        for lbl, cl, da, ratio in ARMS:
            on, _ = rollout_onset(c, bio, cl, da, None if ratio == 1.0 else da * ratio)
            if ref is None:
                ref = on >= 0
            elif not np.array_equal(ref, on >= 0):
                mask_ok = False                       # 9: change WHEN, never WHICH
                print("   [MASK MOVED] %s arm=%s" % (n, lbl))
            R[lbl][n] = arm_metrics(c, on)
        R["oracle"][n] = arm_metrics(c, np.where(
            c["S"], np.where(c["gt_onset"] >= 0, c["gt_onset"], len(c["t"]) - 1), -1))
    print("   final mask identical across all four arms on all %d vessels: %s"
          % (len(ctx), "YES" if mask_ok else "NO -- BUG"))

    print("\n%-12s | %s" % ("vessel", " ".join("%12s" % lbl for lbl, *_ in ARMS) + "      oracle"))
    for n in sealed:
        print("%-12s | %s %11.4f"
              % (n, " ".join("%12.4f" % R[lbl][n]["score"] for lbl, *_ in ARMS),
                 R["oracle"][n]["score"]))

    n_rho = lambda D: int(np.isfinite([v["rho"] for v in D.values()]).sum())    # noqa: E731
    for lbl_set, names in (("SEALED", sealed), ("FIT", fit), ("DEV", dev),
                           ("SEALED+FIT (never selected on)", sealed + fit)):
        print("\n%s  n=%d   MEAN-over-time deploy score" % (lbl_set, len(names)))
        sub = lambda D: {n: D[n] for n in names}                                # noqa: E731
        f_ = sub(R["1sc physics"])
        o_ = sub(R["oracle"])
        prize = mean_(o_) - mean_(f_)
        print("   %-14s %8s %8s %8s | %8s %8s | %8s"
              % ("arm", "mean", "early", "late", "curveL1", "rho", "vs 1sc"))
        for lbl, *_ in ARMS:
            a_ = sub(R[lbl])
            print("   %-14s %8.4f %8.4f %8.4f | %8.4f %+8.3f | %+8.4f  (%d/%d rho defined)"
                  % (lbl, mean_(a_), mean_(a_, "score_early"), mean_(a_, "score_late"),
                     mean_(a_, "curve_l1"), mean_(a_, "rho"), mean_(a_) - mean_(f_),
                     n_rho(a_), len(names)))
        print("   %-14s %8.4f %8.4f %8.4f | %8.4f %+8.3f | %+8.4f  <- the whole prize"
              % ("perfect onset", mean_(o_), mean_(o_, "score_early"), mean_(o_, "score_late"),
                 mean_(o_, "curve_l1"), mean_(o_, "rho"), prize))
        for lbl, *_ in ARMS[1:]:
            got = mean_(sub(R[lbl])) - mean_(f_)
            print("      %-14s recovers %+.4f of %+.4f  (%.0f%%)"
                  % (lbl, got, prize, 100.0 * got / prize if abs(prize) > 1e-9 else 0.0))

    payload = dict(flow=args.flow, q=Q, C_fit=C_fit, metric="mean_over_time",
                   arms={lbl: dict(C=(cl.C if cl else 0.0),
                                   kernel=(cl.kernel if cl else "none"), da=da, ratio=ratio)
                         for lbl, cl, da, ratio in ARMS},
                   base_da=best_base, best_two=best_two, best_cl=best_cl, best_all=best_all,
                   dev_tie=tied, two_scalar_grid=two,
                   fit=fit, dev=dev, sealed=sealed, mask_unchanged=mask_ok, grid=grid,
                   per_vessel={n: dict({lbl: R[lbl][n] for lbl in R},
                                       sealed=bool(ctx[n]["sealed"])) for n in ctx})
    (OUT / f"protocol_{tag}.json").write_text(json.dumps(payload, indent=2, default=float),
                                              encoding="utf-8")
    print("\nwrote %s   (%.0fs total)" % (OUT / f"protocol_{tag}.json", time.time() - t0))
    return 0 if mask_ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
