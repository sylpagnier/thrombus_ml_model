"""Should Track A stay a static t=0 rule, or become a rollout?  Bound it with a flow oracle.

THE QUESTION.  The shipped mask is a static calculation: read the t=0 flow, open two gates,
grow 6 mesh hops, done.  It never integrates.  A rollout Track A would instead ask, at each
step, "given what has committed so far, what commits next" -- which requires the FLOW to
change as the clot grows, and that is the whole difficulty.

WHY THE EXISTING EVIDENCE DOES NOT SETTLE IT.  ``scripts/diag_lever_panel.py`` scored a
time-varying-gate oracle at -0.0091 and algebraic self-blockage at -0.0018, but (a) both
were measured on the mean-over-time overlap score, since retired as discontinuous in commit
time, and (b) the gate oracle's mask was **clipped to the static mask S** -- so the one
thing a rollout would actually do, grow a different mask, was suppressed by construction.

THIS RUNS IT UNCLIPPED, under ``growth_l1``, which is count-based and therefore the only
metric here that can see a mask-size change at all.

    static Track A        the shipped model
    flow-oracle rollout   gate recomputed from GT flow every step; whatever crosses IS the
                          mask.  No graph growth, no clipping.  ORACLE -- it reads GT
                          velocity at all times, so it is a CEILING, not a deployable arm.
    + front admission     same, plus the rolling analogue of the 6-hop growth: tissue
                          adjacent to committed tissue becomes admissible if its CURRENT
                          shear is low enough
    self-blockage         deployable: the gate responds to the model's OWN occlusion
    count floors          best achievable on each mask, so the two architectures are
                          compared on their ceilings and not just their current tuning

If the oracle rollout does not beat the static mask's count floor, no rollout can, and the
static rule is correct.  SEALED is not opened.

    python scripts/diag_rollout_trackA.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig  # noqa: E402
from src.core_physics.ap_closure import SHIPPED, make_rollout_hook  # noqa: E402
from src.core_physics.growth_count_metrics import (  # noqa: E402
    count_optimal_onset, growth_error,
)
from src.core_physics.onset_features import committed_set  # noqa: E402
from src.core_physics.physics_wall_model import (  # noqa: E402
    first_crossing, integrate_mat_trajectory,
)

CACHE = Path("outputs/wall_species_cache")
OUT = Path("outputs/rollout_trackA")
M_TO_CM = 100.0
RELAX = 2.0


def _shim(z):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gc", str(REPO / "scripts" / "eval_growth_count.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.WallShim(z)


def gate_series(z, bio):
    """The law's bracket prefactor at EVERY timestep, from GT flow.  [T, N]."""
    lss, sgt = float(bio.lss), float(bio.sgt) / M_TO_CM
    coef = float(bio.L_char) * M_TO_CM / float(bio.gamma_m)
    sr, dsx = z["sr_t"], z["dsrx_t"]
    return (dsx < sgt) * coef * np.abs(dsx) + (sr < lss), sr


def adjacency(edges, n):
    A = sp.coo_matrix((np.ones(edges.shape[1]), (edges[0], edges[1])), shape=(n, n)).tocsr()
    return ((A + A.T) > 0).astype(np.int8)


def f1(pred, gt):
    tp = float((pred & gt).sum())
    if tp == 0:
        return 0.0
    return float(2 * tp / (2 * tp + (pred & ~gt).sum() + (~pred & gt).sum()))


def main() -> int:
    bio = BiochemConfig(phase="biochem")
    OUT.mkdir(parents=True, exist_ok=True)
    prot = json.load(open("outputs/ap_closure/protocol_gt_meanovertime.json"))
    names = prot["fit"] + prot["dev"]
    crit = float(bio.viscosity_mat_crit)
    lss = float(bio.lss)
    t0 = time.time()

    ARMS = ["static Track A", "flow-oracle rollout", "+ front admission", "self-blockage",
            "FLOOR static mask", "FLOOR oracle-rollout mask"]
    R = {a: {} for a in ARMS}
    sizes = {}

    for n in names:
        p = CACHE / f"{n}.npz"
        if not p.exists() or "sr_t" not in np.load(p).files:
            continue
        z = np.load(p)
        gt = z["gt_onset"]
        if not (gt >= 0).any():
            continue
        nt = len(z["t"])
        nw = len(z["sr0"])
        gate_t, sr_t = gate_series(z, bio)
        gate0 = gate_t[0]
        A = adjacency(z["wall_edges"], nw)
        shim = _shim(z)
        gt_set = gt >= 0

        def run(blockage, g0):
            hook = make_rollout_hook(SHIPPED, bio, z["sr0"])
            traj, _ = integrate_mat_trajectory(shim, bio, g0, da_scale=40.0,
                                               blockage=blockage, ap_closure=hook)
            idx = first_crossing(traj, crit)
            return idx

        # ---- 1. the shipped static rule
        S = committed_set(gate0, z["sr0"], z["wall_edges"])
        idx = run(None, gate0)
        cr = idx >= 0
        med = int(np.median(idx[cr])) if cr.any() else 0
        on_static = np.where(S, np.where(idx >= 0, idx, med), -1)

        # ---- 2. flow-oracle rollout: the evolving gate decides everything
        on_oracle = run(lambda mat, g, i: gate_t[min(i, nt - 1)], gate0)

        # ---- 3. + rolling front admission (the time-resolved analogue of 6-hop growth)
        def blk_front(mat, g, i, _A=A, _gt=gate_t, _sr=sr_t):
            k = min(i, nt - 1)
            occ = (mat >= crit).astype(np.int8)
            adj = (np.asarray(_A @ occ).reshape(-1) > 0) & (_sr[k] < lss * RELAX)
            out = _gt[k].copy()
            out[adj] = np.maximum(out[adj], 1.0)
            return np.where(occ > 0, np.maximum(out, gate0), out)

        on_front = run(blk_front, gate0)

        # ---- 4. deployable self-blockage: gate responds to the model's OWN occlusion
        def blk_self(mat, g, i, _A=A):
            occ = (mat >= crit).astype(np.float64)
            phi = np.clip(np.asarray(_A @ occ).reshape(-1) / np.maximum(
                np.asarray(_A.sum(1)).reshape(-1), 1.0), 0.0, 0.85)
            amp = np.clip(1.0 - 1.0 * phi, 0.02, 1.0)          # 'wake' feedback
            srx = z["sr0"] * amp
            dsx = z["dsrx0"] * amp
            sgt = float(bio.sgt) / M_TO_CM
            coef = float(bio.L_char) * M_TO_CM / float(bio.gamma_m)
            out = (dsx < sgt) * coef * np.abs(dsx) + (srx < lss)
            return np.where(occ > 0, np.maximum(out, gate0), out)

        on_self = run(blk_self, gate0)

        masks = {"static Track A": S,
                 "flow-oracle rollout": on_oracle >= 0,
                 "+ front admission": on_front >= 0,
                 "self-blockage": on_self >= 0}
        for tag, on in (("static Track A", on_static), ("flow-oracle rollout", on_oracle),
                        ("+ front admission", on_front), ("self-blockage", on_self)):
            e = growth_error(on, gt, nt)
            e["f1"] = f1(masks[tag], gt_set)
            e["n_mask"] = int(masks[tag].sum())
            R[tag][n] = e
        for tag, mk in (("FLOOR static mask", S),
                        ("FLOOR oracle-rollout mask", masks["flow-oracle rollout"])):
            e = growth_error(count_optimal_onset(mk, gt, nt), gt, nt)
            e["f1"] = f1(mk, gt_set)
            e["n_mask"] = int(mk.sum())
            R[tag][n] = e
        sizes[n] = dict(n_gt=int(gt_set.sum()), n_static=int(S.sum()),
                        n_oracle=int(masks["flow-oracle rollout"].sum()))

    ok = sorted(R["static Track A"])
    print("%d vessels, %.0fs\n" % (len(ok), time.time() - t0))
    print("=" * 90)
    print("ROLLOUT vs STATIC Track A   growth_l1 (0 = perfect), %d train vessels" % len(ok))
    print("=" * 90)
    print("%-28s %10s %11s %10s %9s" % ("arm", "growth_l1", "final_err", "wall F1", "n_mask"))
    for a in ARMS:
        g = lambda k: float(np.nanmean([R[a][n][k] for n in ok]))          # noqa: E731
        print("%-28s %10.4f %+11.4f %10.4f %9.1f"
              % (a, g("growth_l1"), g("final_err"), g("f1"), g("n_mask")))

    base = float(np.nanmean([R["static Track A"][n]["growth_l1"] for n in ok]))
    fs = float(np.nanmean([R["FLOOR static mask"][n]["growth_l1"] for n in ok]))
    fo = float(np.nanmean([R["FLOOR oracle-rollout mask"][n]["growth_l1"] for n in ok]))
    print("\n   static mask ceiling         %.4f" % fs)
    print("   oracle-rollout mask ceiling %.4f   (%+.4f vs the static mask's)" % (fo, fo - fs))
    print("\n   VERDICT: %s" % (
        "the rollout mask has a BETTER ceiling -- a rollout Track A is worth building"
        if fo < fs - 0.002 else
        "the rollout mask's ceiling is NOT better -- the static rule is not the limitation"))

    print("\n%-12s %6s %8s %8s | %10s %10s" % ("vessel", "n_gt", "static", "oracle",
                                               "gl1 static", "gl1 oracle"))
    for n in ok:
        print("%-12s %6d %8d %8d | %10.4f %10.4f"
              % (n, sizes[n]["n_gt"], sizes[n]["n_static"], sizes[n]["n_oracle"],
                 R["static Track A"][n]["growth_l1"], R["flow-oracle rollout"][n]["growth_l1"]))

    (OUT / "rollout_trackA.json").write_text(json.dumps(
        dict(per_vessel={a: R[a] for a in ARMS}, sizes=sizes, names=ok),
        indent=2, default=float), encoding="utf-8")
    print("\nwrote %s" % (OUT / "rollout_trackA.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
