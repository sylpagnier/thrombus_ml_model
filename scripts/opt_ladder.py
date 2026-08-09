"""Optimisation ladder for the temporal wall model -- items 1-5.

Where the remaining headroom is, from docs/PHASE3_RESULTS.md 13:

    sealed mask, GT flow      0.9093  vs oracle 0.9066   -> saturated
    curve shape (curve_l1)    0.0649  vs oracle 0.0670   -> already beats it
    sealed mask, pred flow    0.8567  vs 0.9093 GT flow  -> 0.053
    onset rho, GT flow        0.685   vs oracle 0.795    -> 0.11 flow + 0.205 NON-flow
    onset rho, pred flow      0.393   vs 0.685 GT flow   -> 0.29   <- largest deficit

Subcommands:
  chem      1. Is the non-flow rho ceiling chemistry?  2x2 oracle ladder over
               {frozen, GT gate} x {t=0 constants, GT species}.
  ode       2. Derive the deposition constants from COMSOL's own export instead of
               fitting da_scale on 19 vessels; resolve the 146 vs 25.2 discrepancy.
  thrombin  3. Replace the fitted graph dilation with a screened-Poisson thrombin field.
  flow      4. The pred-flow ordering collapse: does the stencil that helps the MASK
               hurt the MAGNITUDE that ordering needs?
  lovo      5. Leave-one-vessel-out over the fitted scalars + per-parameter ablation.
               A beating C on sealed is the overfit signature this tests.
  all       everything, in order.

Usage:  python scripts/opt_ladder.py all
"""
from __future__ import annotations

import argparse
import itertools
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
from src.core_physics.mls_gradient import build_mls_gradient, node_positions, shear_rate_2d  # noqa: E402
from src.core_physics.physics_wall_model import (  # noqa: E402
    T0Fields, first_crossing, graded_gate, integrate_mat_trajectory, t0_flow_fields,
)
from src.core_physics.shear_redistribution import (  # noqa: E402
    build_crosssection_operator, make_blockage, sdf_nd,
)
from src.core_physics.species_fields import (  # noqa: E402
    constant_species, depletion_report, gt_species_trajectory,
)
from src.core_physics.species_pushforward_continuous import resolve_deploy_eval_time_index  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.core_physics.temporal_metrics import curve_l1, gt_onset_index, onset_metrics  # noqa: E402
from src.core_physics.thrombin_field import make_ap_boost, make_thrombin_solver  # noqa: E402
from src.evaluation.clot_relaxed_metrics import (  # noqa: E402
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
)

DIR = Path("data/processed/graphs_biochem_anchors")
CACHE = Path("outputs/cache")
OUT = Path("outputs/opt_ladder")
STENCIL = {"gt": 3, "pred": 4}
DA, RM, WAKE, EVERY = 40.0, 0.30, 8.0, 5
RELAX, GROW = 2.0, 6
M_TO_CM = 100.0


# --------------------------------------------------------------------------- shared

def full_horizon_names():
    return sorted(set(WALL_COHORT_V2_TRAIN) | set(WALL_COHORT_V2_GENERALIZATION))


def load_vessel(a, bio, phys, *, need=("gt",)):
    p = DIR / f"{a}.pt"
    if not p.exists():
        return None
    d = torch.load(p, map_location="cpu", weights_only=False)
    if int(d.y.shape[0]) < 150:
        return None
    wall = d.mask_wall.reshape(-1).bool().numpy()
    fields = {}
    for arm in need:
        try:
            fields[arm] = t0_flow_fields(d, bio, hops=STENCIL[arm], flow_source=arm)
        except ValueError:
            pass
    if "gt" not in fields:
        return None
    CACHE.mkdir(parents=True, exist_ok=True)
    cf = CACHE / f"{a}_gt_onset.npy"
    if cf.exists():
        gt_idx = np.load(cf)
    else:
        gt_idx = gt_onset_index(d, phys, wall)
        np.save(cf, gt_idx)
    t_eval = resolve_deploy_eval_time_index(int(d.y.shape[0]))
    pg = gt_clot_phi_at_time(d, t_eval, phys, device=torch.device("cpu")).reshape(-1)
    return dict(a=a, d=d, wall=wall, f=fields, gt_idx=gt_idx,
                phi_gt=pg * torch.tensor(wall.astype(np.float32)),
                split="sealed" if a in WALL_COHORT_V2_GENERALIZATION else "train")


def score_run(c, idx, t):
    m = onset_metrics(idx, c["gt_idx"], t, c["wall"])
    m["curve_l1"] = curve_l1(idx, c["gt_idx"], t, c["wall"])
    pred = torch.tensor((((idx >= 0) & c["wall"])).astype(np.float32))
    mm = compute_clot_relaxed_metrics(pred, c["phi_gt"], c["d"].edge_index,
                                      wall_mask=torch.tensor(c["wall"]))
    m["score"] = clot_score_from_deploy_dict(metrics_to_deploy_prefix(mm))
    return m


def agg(rows, sel, key):
    v = [rows[a][key] for a in sel if a in rows and rows[a][key] == rows[a][key]]
    return float(np.mean(v)) if v else float("nan")


def splits(cache):
    tr = [a for a in WALL_COHORT_V2_TRAIN if a in cache]
    sl = [a for a in WALL_COHORT_V2_GENERALIZATION if a in cache]
    return tr, sl


def n_valid(rows, sel, key="rho"):
    return sum(1 for a in sel if a in rows and rows[a][key] == rows[a][key])


def report(tag, rows, tr, sl):
    # rho is undefined when every node ignites in the same step (the flash), so print the
    # count it was averaged over -- otherwise two rungs get compared on different vessels.
    print("  %-36s train score %.4f rho %.3f (n=%2d) curveL1 %.4f | sealed score %.4f rho %.3f"
          % (tag, agg(rows, tr, "score"), agg(rows, tr, "rho"), n_valid(rows, tr),
             agg(rows, tr, "curve_l1"), agg(rows, sl, "score"), agg(rows, sl, "rho")))
    return {"tag": tag, "n_rho_train": n_valid(rows, tr),
            "train": {k: agg(rows, tr, k) for k in ("score", "rho", "curve_l1", "spread_ratio")},
            "sealed": {k: agg(rows, sl, k) for k in ("score", "rho", "curve_l1", "spread_ratio")},
            "per_vessel": {a: {k: rows[a][k] for k in ("score", "rho", "curve_l1")} for a in rows}}


def gt_gate_series(d, bio, wall):
    """Gate recomputed from the GT velocity at every timestep (flow oracle)."""
    pos = node_positions(d)
    Dx, Dy = build_mls_gradient(pos, d.edge_index.numpy(), hops=3)
    u_ref = float(d.u_ref.reshape(-1)[0])
    d_bar = float(d.d_bar.reshape(-1)[0])
    lss, sgt = float(bio.lss), float(bio.sgt) / M_TO_CM
    nt = int(d.y.shape[0])
    gates = np.zeros((nt, len(wall)))
    for ti in range(nt):
        u = d.y[ti, :, 0].numpy().astype(np.float64)
        v = d.y[ti, :, 1].numpy().astype(np.float64)
        sr = shear_rate_2d(Dx @ u, Dy @ u, Dx @ v, Dy @ v) * (u_ref / d_bar)
        dsrx = (Dx @ sr) / (d_bar * M_TO_CM)
        f = T0Fields(sr=sr, dsrx=dsrx, gate_low=(sr < lss).astype(np.float64),
                     gate_sep=(dsrx < sgt).astype(np.float64), gate=None)
        gates[ti] = graded_gate(f, bio, mode="hard") * wall
    return gates


# --------------------------------------------------------------------- 1. chemistry

def cmd_chem(bio, phys, args):
    """Is the non-flow rho ceiling chemistry?  2x2: {frozen, GT gate} x {const, GT species}."""
    print("\n=== 1. CHEMISTRY ORACLE LADDER ===")
    print("flow oracle already measured: rho 0.713 -> 0.795 with PERFECT flow.")
    print("If GT species lifts rho well past 0.795, the ceiling is chemistry, not flow.\n")
    cache = {}
    for a in full_horizon_names():
        c = load_vessel(a, bio, phys)
        if c:
            cache[a] = c
    tr, sl = splits(cache)
    print("full-horizon: %d train, %d sealed" % (len(tr), len(sl)))

    print("\n-- AP depletion at the wall (does the 'near-constant' premise hold?) --")
    dep = {a: depletion_report(c["d"], bio, c["wall"]) for a, c in cache.items()}
    for k in ("ap_depletion_ratio", "ap_min_frac_of_inlet", "ap_spatial_cv_t0",
              "ap_spatial_cv_tfinal", "rp_depletion_ratio"):
        v = np.array([dep[a][k] for a in dep])
        print("   %-24s median %.4f   min %.4f   max %.4f" % (k, np.median(v), v.min(), v.max()))
    print("   (26.16 claims AP spatial CV ~0.095 and 'essentially inlet everywhere';")
    print("    COMSOL's p007 export has ap spanning 5.1e5..1.25e7, a 24x range.)")

    out = []
    combos = list(itertools.product(("frozen", "gt_gate"), ("const", "gt_species"), (DA,)))
    # Subcommand 'ode' derives da_scale = 145 from COMSOL's own export, but the timing fit
    # wants 40 -- a factor of ~3.6. AP depletion (to 4% of inlet) is a brake of about that
    # size, so the derived constant should only work WITH the real species field. Test it.
    combos += [("frozen", "const", 145.0), ("frozen", "gt_species", 145.0)]
    for gate_mode, spec_mode, da in combos:
        rows = {}
        for a, c in cache.items():
            d, wall = c["d"], c["wall"]
            species = (gt_species_trajectory(d, bio) if spec_mode == "gt_species"
                       else constant_species(d, bio))
            if gate_mode == "frozen":
                g0 = graded_gate(c["f"]["gt"], bio, mode="hard") * wall
                blk = None
            else:
                gs = gt_gate_series(d, bio, wall)
                g0 = gs[0]
                blk = lambda mat, g, i, _gs=gs: _gs[min(i, len(_gs) - 1)]
            traj, t = integrate_mat_trajectory(d, bio, g0, da_scale=da,
                                               blockage=blk, species=species)
            rows[a] = score_run(c, first_crossing(traj, float(bio.viscosity_mat_crit)), t)
        out.append(report("%s gate + %s (da=%g)" % (gate_mode, spec_mode, da), rows, tr, sl))

    base, chem = out[0]["train"]["rho"], out[1]["train"]["rho"]
    flow, both = out[2]["train"]["rho"], out[3]["train"]["rho"]
    print("\n  VERDICT (train rho):")
    print("    frozen+const   %.3f  (baseline)" % base)
    print("    +GT species    %.3f  (chemistry alone: %+.3f)" % (chem, chem - base))
    print("    +GT flow       %.3f  (flow alone:      %+.3f)" % (flow, flow - base))
    print("    +both          %.3f  (%+.3f)" % (both, both - base))
    if chem - base > 1.5 * max(flow - base, 1e-9):
        print("    -> CHEMISTRY dominates. Build the thrombin/AP coupling (subcommand 'thrombin').")
    elif chem - base < 0.3 * max(flow - base, 1e-9):
        print("    -> chemistry is NOT the ceiling. Ordering headroom is flow or irreducible.")
    else:
        print("    -> both matter comparably; neither alone closes the gap.")
    return {"depletion": dep, "ladder": out}


# ------------------------------------------------------------------ 2. derive the ODE

def cmd_ode(bio, phys, args):
    """Fit COMSOL's actual surface ODE from its own export, incl. the terms we dropped."""
    print("\n=== 2. DERIVE THE DEPOSITION CONSTANTS FROM THE COMSOL EXPORT ===")
    npz = Path("outputs/comsol_p007_wall.npz")
    if not npz.exists():
        print("  [!] %s missing -- run: python scripts/parse_comsol_wall_export.py" % npz)
        return {"error": "missing export"}
    e = np.load(npz)
    MINF, K_RS, K_AS, K_AA = 7.0e6, 3.7e-3, 4.5e-2, 4.5e-2
    L, GM, LSS, SGT = 7.5e-2, 150.0, 25.0, -750.0
    gate = ((e["dsrx"] < SGT) * (L / GM) * np.abs(e["dsrx"]) + (e["sr"] < LSS)).astype(float)
    s2t = e["step2t"]
    terms = {
        "gate*Sat*(krs*rp+kas*ap)": gate * e["Sat"] * (K_RS * e["rp"] + K_AS * e["ap"]),
        "gate*(Mas/Minf)*kaa*ap": gate * (e["Mas"] / MINF) * K_AA * e["ap"],
        "gate*(Mat/Minf)*kaa*ap": gate * (e["Mat"] / MINF) * K_AA * e["ap"],
        "gate*Mat*PT (thrombin src)": gate * e["Mat"] * e["PT"],
        "Mat*th": e["Mat"] * e["th"],
        "gate*(Mas/Minf)*kaa*AP0": gate * (e["Mas"] / MINF) * K_AA * 1.25e7,
    }
    y = (e["dMatt"] * s2t).reshape(-1)
    keep = np.isfinite(y)
    print("\n  single-term fits for d(Mat,t):")
    for k, v in terms.items():
        x = (v * s2t).reshape(-1)[keep]
        c = (x @ y[keep]) / max(x @ x, 1e-30)
        r2 = 1 - ((y[keep] - c * x) ** 2).sum() / ((y[keep] - y[keep].mean()) ** 2).sum()
        print("    %-28s coef %-12.5g R2 %7.4f  (coef/Da = %.4g)" % (k, c, r2, c / 1e-4))

    print("\n  is the AP variation what the constant-AP model loses?")
    a = (terms["gate*(Mas/Minf)*kaa*ap"] * s2t).reshape(-1)[keep]
    b = (terms["gate*(Mas/Minf)*kaa*AP0"] * s2t).reshape(-1)[keep]
    for nm, x in (("true ap", a), ("constant AP0", b)):
        c = (x @ y[keep]) / max(x @ x, 1e-30)
        r2 = 1 - ((y[keep] - c * x) ** 2).sum() / ((y[keep] - y[keep].mean()) ** 2).sum()
        print("    %-14s coef %-12.5g R2 %7.4f" % (nm, c, r2))

    names = list(terms)
    A = np.stack([(terms[k] * s2t).reshape(-1)[keep] for k in names], 1)
    coef, *_ = np.linalg.lstsq(A, y[keep], rcond=None)
    pred = A @ coef
    r2 = 1 - ((y[keep] - pred) ** 2).sum() / ((y[keep] - y[keep].mean()) ** 2).sum()
    print("\n  full dictionary  R2 %.4f" % r2)
    for k, cc in zip(names, coef):
        print("    %-28s %+.6g   (x Da = %+.4g)" % (k, cc, cc / 1e-4))

    print("\n  IMPLIED da_scale for the repo's law (autocat term coef / Da):")
    x = (terms["gate*(Mas/Minf)*kaa*ap"] * s2t).reshape(-1)[keep]
    c = (x @ y[keep]) / max(x @ x, 1e-30)
    print("    derived  %.1f      currently FITTED at %.0f on 19 vessels" % (c / 1e-4, DA))
    print("    (a derived constant transfers; a fitted one is 1 of the 6 overfit risks)")
    return {"single": {k: float((((terms[k] * s2t).reshape(-1)[keep]) @ y[keep])
                                / max(((terms[k] * s2t).reshape(-1)[keep]) @
                                      ((terms[k] * s2t).reshape(-1)[keep]), 1e-30))
                       for k in terms},
            "full_r2": float(r2), "coef": dict(zip(names, coef.tolist()))}


# ------------------------------------------------------------------- 3. thrombin field

def cmd_thrombin(bio, phys, args):
    """Screened-Poisson thrombin field vs the fitted graph dilation."""
    print("\n=== 3. THROMBIN FIELD REPLACES THE FITTED GRAPH DILATION ===")
    d_bar_m = 0.015
    ld = np.sqrt(2 * float(bio.D_T) * 30000.0)
    print("  thrombin diffusion length sqrt(2*D_T*t_final) = %.3f mm = %.3f d_bar"
          % (ld * 1e3, ld / d_bar_m))
    print("  fitted grow_hops = %d; 26.13.2 measured late commits within ~2 hops\n" % GROW)
    cache = {}
    for a in full_horizon_names():
        c = load_vessel(a, bio, phys)
        if c:
            cache[a] = c
    tr, sl = splits(cache)
    out = []

    WASH = (0.0, 0.003, 0.01, 0.03, 0.1)
    for a, c in cache.items():
        d, wall = c["d"], c["wall"]
        pos = node_positions(d)
        c["solvers"] = {}
        for wc in WASH:
            c["solvers"][wc] = make_thrombin_solver(
                d, bio, pos, c["f"]["gt"].sr, wash_coef=wc, wall=wall)
        c["B"] = build_crosssection_operator(pos, sdf_nd(d), wall, radius_mult=RM)
        ei = d.edge_index.numpy()
        n = len(wall)
        A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
        c["A"] = ((A + A.T) > 0).astype(np.int8)
    print("  screened range at the WALL, in mesh hops (needs >= 1 to couple at all):")
    for wc in WASH:
        r = np.array([c["solvers"][wc][1]["range_in_hops"] for c in cache.values()])
        print("    wash_coef %-6g median %8.3f hops   [%.3f .. %.3f]"
              % (wc, np.median(r), r.min(), r.max()))

    def run(c, *, gain, wake, dilate, wash=0.0, feedback="wake"):
        d, wall = c["d"], c["wall"]
        g0 = graded_gate(c["f"]["gt"], bio, mode="hard") * wall
        blk = (make_blockage(c["f"]["gt"], bio, c["B"], wall, every=EVERY,
                             feedback=feedback, wake=wake,
                             thrombin_solve=c["solvers"][wash][0]) if wake > 0 else None)
        boost = (make_ap_boost(c["solvers"][wash][0], bio, gain=gain, every=EVERY)
                 if gain > 0 else None)
        traj, t = integrate_mat_trajectory(d, bio, g0, da_scale=DA, blockage=blk,
                                           ap_boost=boost)
        idx = first_crossing(traj, float(bio.viscosity_mat_crit))
        if dilate:
            cur = (idx >= 0) & wall
            adm = (c["f"]["gt"].sr < float(bio.lss) * RELAX) & wall
            for _ in range(GROW):
                cur = cur | (((c["A"] @ cur.astype(np.int8)) > 0) & adm)
            idx = np.where(cur & (idx < 0), len(t) - 1, idx)
        return score_run(c, idx, t)

    print()
    for tag, kw in (("wake only (current B)", dict(gain=0.0, wake=WAKE, dilate=False)),
                    ("wake + graph dilation (C)", dict(gain=0.0, wake=WAKE, dilate=True))):
        out.append(report(tag, {a: run(c, **kw) for a, c in cache.items()}, tr, sl))
    # (a) thrombin field REPLACES the fitted-radius wake: derived range vs fitted ball.
    for wash, wk in itertools.product(WASH, (2.0, 8.0)):
        out.append(report("thrombin-wake wash=%-6g wake=%-4.1f" % (wash, wk),
                          {a: run(c, gain=0.0, wake=wk, dilate=False, wash=wash,
                                  feedback="thrombin") for a, c in cache.items()}, tr, sl))
    # (b) AP boost on top: this can only reorder WITHIN the gated set -- every deposition
    #     term is gated, so chemistry cannot ignite an ungated node. Ordering, not spread.
    for gain in (4.0, 16.0):
        out.append(report("wake + AP boost gain=%-4.1f (ordering)" % gain,
                          {a: run(c, gain=gain, wake=WAKE, dilate=False, wash=0.0)
                           for a, c in cache.items()}, tr, sl))
    best = max([o for o in out if "thrombin" in o["tag"]], key=lambda z: z["train"]["score"])
    dilb = [o for o in out if "dilation" in o["tag"]][0]
    print("\n  best thrombin %.4f train / %.4f sealed   vs graph dilation %.4f / %.4f"
          % (best["train"]["score"], best["sealed"]["score"],
             dilb["train"]["score"], dilb["sealed"]["score"]))
    print("  (thrombin trades 2 fitted scalars for 1, and derives its RANGE from D_T)")
    return out


# ------------------------------------------------------- 4. pred-flow ordering collapse

def cmd_flow(bio, phys, args):
    """Does the stencil that helps the MASK hurt the MAGNITUDE that ordering needs?"""
    print("\n=== 4. PRED-FLOW ORDERING COLLAPSE (rho 0.685 -> 0.393) ===")
    print("  mask needs the gate's SIGN; ordering needs its MAGNITUDE. Sweep them apart.\n")
    cache = {}
    for a in full_horizon_names():
        c = load_vessel(a, bio, phys, need=("gt", "pred"))
        if c and "pred" in c["f"]:
            cache[a] = c
    tr, sl = splits(cache)
    print("  vessels with u0_pred: %d train, %d sealed" % (len(tr), len(sl)))
    out = []
    for arm, st in itertools.product(("gt", "pred"), (2, 3, 4, 5)):
        rows = {}
        for a, c in cache.items():
            try:
                f = t0_flow_fields(c["d"], bio, hops=st, flow_source=arm)
            except ValueError:
                continue
            g0 = graded_gate(f, bio, mode="hard") * c["wall"]
            pos = node_positions(c["d"])
            B = build_crosssection_operator(pos, sdf_nd(c["d"]), c["wall"], radius_mult=RM)
            blk = make_blockage(f, bio, B, c["wall"], every=EVERY, feedback="wake", wake=WAKE)
            traj, t = integrate_mat_trajectory(c["d"], bio, g0, da_scale=DA, blockage=blk)
            rows[a] = score_run(c, first_crossing(traj, float(bio.viscosity_mat_crit)), t)
        out.append(report("%s flow, stencil %d" % (arm, st), rows, tr, sl))
    pr = [o for o in out if o["tag"].startswith("pred")]
    bm = max(pr, key=lambda z: z["train"]["score"])
    br = max(pr, key=lambda z: z["train"]["rho"])
    print("\n  pred arm: best MASK at %s (score %.4f, rho %.3f)"
          % (bm["tag"], bm["train"]["score"], bm["train"]["rho"]))
    print("            best RHO  at %s (score %.4f, rho %.3f)"
          % (br["tag"], br["train"]["score"], br["train"]["rho"]))
    print("  -> if these differ, mask and ordering want different operators and the")
    print("     deployable model should carry both.")
    return out


# ------------------------------------------------------------------------- 5. LOVO

def cmd_lovo(bio, phys, args):
    """Leave-one-vessel-out over the fitted scalars, plus per-parameter ablation."""
    print("\n=== 5. LEAVE-ONE-VESSEL-OUT + PARAMETER ABLATION ===")
    print("  6 scalars fitted on 19 vessels. A beating C on sealed is the overfit tell.\n")
    cache = {}
    for a in full_horizon_names():
        c = load_vessel(a, bio, phys)
        if c:
            cache[a] = c
    tr, sl = splits(cache)
    for a, c in cache.items():
        pos = node_positions(c["d"])
        c["B"] = build_crosssection_operator(pos, sdf_nd(c["d"]), c["wall"], radius_mult=RM)

    grid = [dict(da=da, wake=wk, mode=md, tau=tu)
            for da, wk, (md, tu) in itertools.product(
                (30, 40, 60, 100), (0.0, 4.0, 8.0, 12.0),
                (("hard", 0.0), ("sigmoid_low", 0.10)))]
    print("  scoring %d configs x %d vessels ..." % (len(grid), len(cache)))
    M = {}
    for gi, g in enumerate(grid):
        for a, c in cache.items():
            gate = graded_gate(c["f"]["gt"], bio, mode=g["mode"], tau_low=g["tau"]) * c["wall"]
            blk = (make_blockage(c["f"]["gt"], bio, c["B"], c["wall"], every=EVERY,
                                 graded_mode=g["mode"], tau_low=g["tau"],
                                 feedback="wake", wake=g["wake"]) if g["wake"] > 0 else None)
            traj, t = integrate_mat_trajectory(c["d"], bio, gate, da_scale=g["da"], blockage=blk)
            M[(gi, a)] = score_run(c, first_crossing(traj, float(bio.viscosity_mat_crit)), t)

    def fit_on(sel):
        return max(range(len(grid)),
                   key=lambda gi: float(np.mean([M[(gi, a)]["score"] for a in sel])))

    gi_all = fit_on(tr)
    print("\n  fit on ALL train : %s" % grid[gi_all])
    print("    train (in-sample) %.4f | sealed %.4f"
          % (float(np.mean([M[(gi_all, a)]["score"] for a in tr])),
             float(np.mean([M[(gi_all, a)]["score"] for a in sl]))))

    held, picks = [], []
    for a in tr:
        gi = fit_on([x for x in tr if x != a])
        held.append(M[(gi, a)]["score"])
        picks.append(gi)
    print("\n  LOVO (honest) train estimate %.4f   vs in-sample %.4f   optimism %+.4f"
          % (float(np.mean(held)), float(np.mean([M[(gi_all, a)]["score"] for a in tr])),
             float(np.mean([M[(gi_all, a)]["score"] for a in tr])) - float(np.mean(held))))
    uniq = len(set(picks))
    print("  config chosen by %d/%d folds is the same one: %s"
          % (picks.count(gi_all), len(picks), "STABLE" if uniq <= 2 else "UNSTABLE (%d distinct)" % uniq))

    print("\n  per-parameter ablation (freeze one to a neutral value, refit the rest on train):")
    for name, neutral in (("wake", 0.0), ("mode", "hard"), ("da", 40)):
        sub = [gi for gi, g in enumerate(grid) if g[name] == neutral]
        gi = max(sub, key=lambda gi: float(np.mean([M[(gi, a)]["score"] for a in tr])))
        print("    %-6s frozen at %-12s train %.4f (%+.4f)  sealed %.4f (%+.4f)"
              % (name, str(neutral),
                 float(np.mean([M[(gi, a)]["score"] for a in tr])),
                 float(np.mean([M[(gi, a)]["score"] for a in tr]))
                 - float(np.mean([M[(gi_all, a)]["score"] for a in tr])),
                 float(np.mean([M[(gi, a)]["score"] for a in sl])),
                 float(np.mean([M[(gi, a)]["score"] for a in sl]))
                 - float(np.mean([M[(gi_all, a)]["score"] for a in sl]))))
    return {"grid": grid, "best": grid[gi_all], "lovo": float(np.mean(held)),
            "in_sample": float(np.mean([M[(gi_all, a)]["score"] for a in tr])),
            "sealed": float(np.mean([M[(gi_all, a)]["score"] for a in sl]))}


# --------------------------------------------------------------------------- driver

CMDS = {"chem": cmd_chem, "ode": cmd_ode, "thrombin": cmd_thrombin,
        "flow": cmd_flow, "lovo": cmd_lovo}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=list(CMDS) + ["all"])
    args = ap.parse_args()
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    OUT.mkdir(parents=True, exist_ok=True)
    todo = list(CMDS) if args.cmd == "all" else [args.cmd]
    results = {}
    for name in todo:
        t0 = time.time()
        try:
            results[name] = CMDS[name](bio, phys, args)
        except Exception as e:  # keep the ladder going; a broken rung is a result too
            import traceback
            traceback.print_exc()
            results[name] = {"error": f"{type(e).__name__}: {e}"}
        print("  [%s done in %.1fs]" % (name, time.time() - t0))
        Path(OUT / f"{name}.json").write_text(
            json.dumps(results[name], indent=2, default=float), encoding="utf-8")
    print("\nwrote %s/*.json" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
