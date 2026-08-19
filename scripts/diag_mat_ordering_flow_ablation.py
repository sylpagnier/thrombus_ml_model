"""PHASE 7 5.1 top item: is the Mat MAGNITUDE ordering a flow-coupling bug or an ML problem?

``docs/PHASE7_FINDINGS.md`` 4.1 measured ``spearman(model Mat, GT Mat) = 0.586`` at the wall
and showed it is invariant to every scalar in the physics model (``da_scale`` is a monotone
rescale, so it cannot move a rank correlation).  That 0.586 -> 1.0 interval is worth
**+0.0757** full-mesh deploy score through the 3.2 attenuation rule.

``PHASE6_RESULTS`` 18 showed that recomputing the gate from the EVOLVING GT flow fixes the
wall *mask* (F1 0.8405 -> 0.8953).  The open question is whether it also fixes the
*magnitude ordering*.  It is the question that decides what gets built next:

    oracle recovers most of 0.586 -> 1.0   ==>  a flow-coupling fix, not an ML problem
    oracle recovers little                ==>  the ordering is missing information the
                                               flow does not carry, and the static
                                               Mat-magnitude regression in 5 is the job

Arms.  Every arm produces one thing -- a wall ``Mat`` field -- and is scored the same way.
The wall MASK is held at the shipped construction throughout so the comparison isolates
magnitude (5's gate 2: "the wall mask must not move").

    frozen t0 / gt        shipped:  gate frozen at the COMSOL t=0 velocity      <- the 0.586
    frozen t0 / pred      gate frozen at RGP-DEQ ``u0_pred``                    deploy-legal
    evolving / GT flow    gate re-evaluated on the GT velocity every step       ORACLE
    evolving / corrector  gate re-evaluated on corrector-diverted flow          deploy-legal
    ORACLE Mat            GT Mat itself, i.e. rho = 1 by construction           the ceiling

RESULT: the oracle recovers 21% of the ordering gap and 0.4% of the score gap, so it is the
second branch -- see FINDINGS 7.  Two things the run also produced, both of which change the
target rather than the answer:

    rho_corner   0.586 is inflated.  Half the wall nodes are the quadratic mesh's mid-edge
                 nodes, GT Mat is structurally zero on 44.5% of them, and the model is zero
                 there too, so a rank correlation credits a big block of agreed ties.  On
                 species-carrying nodes the ordering is 0.193 (FINDINGS 8.5).
    +qmatch      remapping each arm's Mat onto GT's histogram separates CALIBRATION from
                 ORDERING, and calibration alone is 53% of the score gap (FINDINGS 7.2).

    python scripts/diag_mat_ordering_flow_ablation.py                  # physics arms only
    python scripts/diag_mat_ordering_flow_ablation.py --corrector      # + the learned arms
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from predict_wall_clot import GROW_HOPS, RELAX, STENCIL, node_pos  # noqa: E402
from src.biochem_gnn.mat_growth_simple import WALL_COHORT_V2_TRAIN  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.ap_closure import SHIPPED, SHIPPED_DA_SCALE, make_rollout_hook  # noqa: E402
from src.core_physics.physics_lumen_model import (  # noqa: E402
    MAT_ATTENUATION, fill_grown_wall_mat, grow_into_lumen_by_mat, midside_nodes,
)
from src.core_physics.physics_wall_model import (  # noqa: E402
    CorrectorArm, corrector_blockage, gate_from_shear, integrate_mat_trajectory,
    predicted_seed_mask, t0_flow_fields,
)
from src.core_physics.species_pushforward_continuous import (  # noqa: E402
    resolve_deploy_eval_time_index,
)
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.core_physics.temporal_metrics import spearman  # noqa: E402
from src.evaluation.clot_relaxed_metrics import (  # noqa: E402
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
)

DIR = REPO / "data/processed/graphs_biochem_anchors"
CACHE = REPO / "outputs/wall_species_cache"
CORR_CKPT = REPO / "outputs/kinematics/local_corrector/local_kinematic_corrector_best.pth"
MAT_S = 7e10        # pack Mat_log1p_nd -> COMSOL model units (docs/PHASE6_RESULTS.md 1)

QM = " +qmatch"     # same arm, Mat monotone-remapped onto GT's histogram (ordering only)
PHYSICS_ARMS = ["frozen t0 / gt", "frozen t0 / pred", "evolving / GT flow"]
CORR_ARMS = ["evolving / corr gt", "evolving / corr pred"]
KEYS = ("rho_all", "rho_corner", "z_both_ms", "z_both_co", "rho_fill", "rho_gt", "ratio",
        "n_trig", "off_f1", "score", "n_ode")
# p012 / p041 / p044 carry no ``u0_pred`` and are three of the four largest off-wall-GT
# vessels, so a raw mean over "whatever each arm could run" compares the pred arms on a
# strictly easier cohort.  Every headline number below is on the COMMON subset.


def f1(pred: np.ndarray, gt: np.ndarray) -> float:
    if gt.sum() == 0 and pred.sum() == 0:
        return float("nan")
    tp = int((pred & gt).sum())
    p = tp / max(int(pred.sum()), 1)
    r = tp / max(int(gt.sum()), 1)
    return 2 * p * r / max(p + r, 1e-9)


def quantile_match(src: np.ndarray, ref: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """Remap ``src`` onto ``ref``'s distribution over ``sel``, preserving rank order.

    WHY.  ``rho`` and the deploy score disagree about what is broken, so they have to be
    separated.  A monotone remap destroys all level/spread information and keeps ORDERING
    exactly, so scoring the remapped field asks one clean question: **given GT's own Mat
    histogram, does the arm's ordering put the top of it on the right nodes?**

    Near it, ``n_trig`` is forced to GT's by construction, so any remaining off-wall F1
    deficit is ordering and nothing else.  Ties are averaged, which matters a lot here: the
    model leaves most wall nodes at Mat = 0 and has genuinely no ordering among them, so
    they must all receive the SAME remapped value rather than being spread at random over
    GT's upper tail (which would manufacture precision the arm has not earned).
    """
    from scipy.stats import rankdata

    out = np.asarray(src, dtype=np.float64).copy()
    s, r = out[sel], np.sort(np.asarray(ref, dtype=np.float64)[sel])
    if len(s) < 2:
        return out
    q = (rankdata(s, method="average") - 0.5) / len(s)
    out[sel] = np.interp(q, (np.arange(len(r)) + 0.5) / len(r), r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corrector", action="store_true",
                    help="also run the two corrector-driven evolving-flow arms (needs CUDA)")
    ap.add_argument("--every", type=int, default=10, help="rollout steps between corrector calls")
    ap.add_argument("--save", default="outputs/phase7_mat_ordering_flow_ablation.json")
    args = ap.parse_args()

    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    crit = float(bio.viscosity_mat_crit)
    flow_arms = list(PHYSICS_ARMS) + (list(CORR_ARMS) if args.corrector else [])
    arms = [x for a in flow_arms for x in (a, a + QM)] + ["ORACLE Mat"]
    acc = {a: {k: [] for k in KEYS} for a in arms}
    per_vessel: dict[str, dict] = {}

    corr = device = None
    if args.corrector:
        from src.core_physics.coupled_shear_gnn import load_local_corrector
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        corr = load_local_corrector(CORR_CKPT, device)
        print("[i] corrector on %s, every=%d steps\n" % (device, args.every))

    for anchor in WALL_COHORT_V2_TRAIN:
        pk, cf = DIR / f"{anchor}.pt", CACHE / f"{anchor}.npz"
        if not pk.exists() or not cf.exists():
            continue
        z = np.load(cf)
        if "sr_t" not in z.files:
            continue
        d = torch.load(pk, map_location="cpu", weights_only=False)
        if int(d.y.shape[0]) < 150:
            continue
        wall = d.mask_wall.reshape(-1).bool().numpy()
        n, widx, nt = len(wall), z["wall_idx"], len(z["t"])
        ei = d.edge_index.detach().cpu().numpy()
        A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
        A = ((A + A.T) > 0).astype(np.int8)
        pos = node_pos(d)
        ms = midside_nodes(pos, ei)

        t_eval = resolve_deploy_eval_time_index(int(d.y.shape[0]))
        gt_f = gt_clot_phi_at_time(d, t_eval, phys, device=torch.device("cpu")).reshape(-1)
        gt = gt_f.numpy() > 0.5
        if gt.sum() == 0:
            continue
        names = d.y_channel_names.split(",")
        mat_gt = np.expm1(d.y[-1, :, names.index("Mat_log1p_nd")].double().numpy()) * MAT_S

        # The gate series is the ONLY thing that differs between arms.  ``None`` = frozen.
        series: dict[str, tuple[np.ndarray, object]] = {}
        f_gt = t0_flow_fields(d, bio, hops=STENCIL["gt"], flow_source="gt")
        series["frozen t0 / gt"] = (f_gt.gate, None)
        has_pred = getattr(d, "u0_pred", None) is not None
        f_pr = None
        if has_pred:
            f_pr = t0_flow_fields(d, bio, hops=STENCIL["pred"], flow_source="pred")
            series["frozen t0 / pred"] = (f_pr.gate, None)

        g_or = np.zeros((nt, n))
        g_or[:, widx] = gate_from_shear(z["sr_t"], z["dsrx_t"], bio)
        series["evolving / GT flow"] = (
            g_or[0], lambda m, g0, i, _g=g_or: _g[min(i, nt - 1)] * wall)

        if args.corrector:
            arm = CorrectorArm(corrector=corr, phys_cfg=phys, device=device,
                               every=int(args.every), da_scale=SHIPPED_DA_SCALE)
            series["evolving / corr gt"] = (
                f_gt.gate, corrector_blockage(d, bio, f_gt, arm, hops=STENCIL["gt"],
                                              flow_source="gt"))
            if has_pred:
                series["evolving / corr pred"] = (
                    f_pr.gate, corrector_blockage(d, bio, f_pr, arm, hops=STENCIL["pred"],
                                                  flow_source="pred"))

        # Shipped wall mask, from the GT-t=0 fields, identical for every arm.
        base, _, _ = predicted_seed_mask(d, bio, f_gt, relax=RELAX, grow_hops=GROW_HOPS, adj=A)
        hot = wall & (mat_gt > 0)                     # GT ever deposited anything
        gtc = wall & (mat_gt >= crit)                 # GT committed
        row: dict[str, dict] = {}

        mats: dict[str, tuple[np.ndarray, float]] = {"ORACLE Mat": (mat_gt, float("nan"))}
        for a, (g0, blk) in series.items():
            hook = make_rollout_hook(SHIPPED, bio, f_gt.sr)
            traj, _ = integrate_mat_trajectory(
                d, bio, g0 * wall, da_scale=SHIPPED_DA_SCALE, blockage=blk, ap_closure=hook)
            mats[a] = (traj[-1], float((wall & (traj[-1] >= crit)).sum()))
            mats[a + QM] = (quantile_match(traj[-1], mat_gt, wall), mats[a][1])

        for a in arms:
            if a not in mats:
                continue
            mat_m, n_ode = mats[a]
            mw = fill_grown_wall_mat(mat_m, base, wall, A, hops=6)
            pred = base | grow_into_lumen_by_mat(mw, wall, pos, ei, crit)
            m = compute_clot_relaxed_metrics(
                torch.tensor(pred.astype(np.float32)), gt_f, d.edge_index)
            r = dict(
                rho_all=spearman(mat_m[wall], mat_gt[wall]),
                # 49.6% of wall nodes are the quadratic mesh's mid-edge nodes, and GT Mat is
                # zero on 44.6% of those against 17.6% of corner nodes (FINDINGS 8.5).  A
                # block of tied zeros in the reference depresses a rank correlation
                # mechanically, so rho_all may be scoring an export artefact.  This is the
                # same measurement over corner wall nodes only.
                rho_corner=spearman(mat_m[wall & ~ms], mat_gt[wall & ~ms]),
                # Fraction of each family where model AND GT are both zero.  A shared block
                # of tied zeros is scored as perfect agreement by a rank correlation, so if
                # this is large on the mid-side half it explains rho_all > rho_corner
                # without the model having ordered anything.
                z_both_ms=float(((mat_m <= 0) & (mat_gt <= 0))[wall & ms].mean()),
                z_both_co=float(((mat_m <= 0) & (mat_gt <= 0))[wall & ~ms].mean()),
                rho_fill=spearman(mw[wall], mat_gt[wall]),
                rho_gt=spearman(mat_m[gtc], mat_gt[gtc]),
                # Magnitude bias where the criterion actually bites: GT-committed wall
                # nodes.  The off-wall rule needs Mat_owner >= crit/0.16 = 6.25*crit, so a
                # ratio below 1 is the whole reason the deployable arm finds nothing.
                ratio=float(np.median(mat_m[gtc]) / max(np.median(mat_gt[gtc]), 1e-30)),
                # Wall nodes that clear the off-wall TRIGGER, crit/0.16.  This is the
                # number the lumen arm actually consumes: at zero it contributes nothing
                # regardless of how good the ordering is.
                n_trig=float((wall & (mw >= crit / MAT_ATTENUATION)).sum()),
                off_f1=f1(pred & ~wall, gt & ~wall),
                score=clot_score_from_deploy_dict(metrics_to_deploy_prefix(m)),
                n_ode=n_ode,
            )
            row[a] = r
            for k in KEYS:
                if r[k] == r[k]:
                    acc[a][k].append(r[k])
        row["_n_off"] = int((gt & ~wall).sum())
        per_vessel[anchor] = row
        print("%-12s off-GT %3d | %s" % (
            anchor, int((gt & ~wall).sum()),
            "  ".join("%s rho %+.3f score %.4f/%.4f"
                      % (a.split(" / ")[-1], row[a]["rho_all"], row[a]["score"],
                         row[a + QM]["score"])
                      for a in flow_arms if a in row)))

    print("\n%d vessels ran.  rho_* = spearman(arm Mat, GT Mat) at the wall (FINDINGS 4.1"
          " quotes 0.586).  Each table shows only arms that ran on ALL of its vessels, so"
          " every row is like-for-like." % len(per_vessel))

    def table(sel: list[str], label: str) -> dict:
        got = {}
        print("\n[%s]  %d vessels: %s" % (label, len(sel), " ".join(s[-3:] for s in sel)))
        print("%-27s %8s %10s %9s %9s %7s %7s %8s %9s %7s"
              % ("arm", "rho_all", "rho_corner", "0=0 mid", "0=0 corn", "ratio", "n_trig",
                 "off F1", "score", "n_ODE"))
        for a in arms:
            if not sel or not all(a in per_vessel[v] for v in sel):
                continue
            vals = {k: [per_vessel[v][a][k] for v in sel
                        if per_vessel[v][a][k] == per_vessel[v][a][k]] for k in KEYS}
            got[a] = {k: float(np.mean(v)) if v else None for k, v in vals.items()}
            print("%-27s %8.3f %10.3f %9.3f %9.3f %7.3f %7.1f %8.4f %9.4f %7.1f"
                  % (a, got[a]["rho_all"], got[a]["rho_corner"], got[a]["z_both_ms"],
                     got[a]["z_both_co"], got[a]["ratio"], got[a]["n_trig"], got[a]["off_f1"],
                     got[a]["score"],
                     got[a]["n_ode"] if got[a]["n_ode"] is not None else float("nan")))
        return got

    allv = list(per_vessel)
    predv = [v for v in allv if "frozen t0 / pred" in per_vessel[v]]
    offv = [v for v in allv if per_vessel[v].get("_n_off", 0) > 0]
    summ = {
        "all": table(allv, "ALL -- the GT-flow ablation, this is the headline"),
        "offwall": table(offv, "OFF-WALL-CARRYING only -- where the lumen arm can score"),
        "pred": table(predv, "u0_pred subset -- the only place the pred arms are fair"),
    }

    for label in ("all", "offwall", "pred"):
        g = summ[label]
        if "frozen t0 / gt" not in g or "ORACLE Mat" not in g:
            continue
        b, c = g["frozen t0 / gt"], g["ORACLE Mat"]
        gap = c["score"] - b["score"]
        print("\n[%s] interval to close: score %.4f -> %.4f (%+.4f), rho %.3f -> 1.000"
              % (label, b["score"], c["score"], gap, b["rho_all"]))
        for a in arms:
            if a in ("frozen t0 / gt", "ORACLE Mat") or a not in g:
                continue
            drho = g[a]["rho_all"] - b["rho_all"]
            print("   %-27s d_rho %+.3f (%6.1f%% of rho gap)  d_score %+.4f (%6.1f%%"
                  " of score gap)"
                  % (a, drho, 100.0 * drho / max(1.0 - b["rho_all"], 1e-9),
                     g[a]["score"] - b["score"],
                     100.0 * (g[a]["score"] - b["score"]) / gap if gap else float("nan")))

    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"n_vessels": len(per_vessel), "subsets": {"all": allv, "offwall": offv,
                                                   "pred": predv},
         "per_vessel": per_vessel, "summary": summ}, indent=2))
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
