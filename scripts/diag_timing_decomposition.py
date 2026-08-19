"""What is the remaining 0.0800 of timing error actually MADE OF?

Eight methods have now been aimed at timing and every one caps at 10-15% of the prize.  That
pattern says the error is not understood, so this decomposes it instead of adding a ninth.

The shipped model has a structural seam.  Track A picks the mask (two t=0 gates + 6 hops of
graph growth); Track B integrates the surface ODE on the frozen gate.  They are computed
INDEPENDENTLY, and the graph-grown nodes have ``gate == 0`` -- their ODE never moves, so they
never cross, and the scoring convention hands them the ODE's **median onset, one constant**.
That is 15-26% of the predicted mask on most vessels.

So the 0.0800 splits two ways, and the split decides where the next month goes:

    ODE-timed nodes     the gated seeds, whose onset the surface ODE genuinely computes
    stitch nodes        the graph-grown ones, all given one constant

Each arm below replaces ONE class with its GT onset and leaves the other alone.  Whichever
recovers more is where the error lives.  ``both`` is the onset oracle and ``FLOOR`` is the
best any reassignment could do on this mask.

Cache-only, no packs, SEALED not opened.

    python scripts/diag_timing_decomposition.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import importlib.util  # noqa: E402

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


def main() -> int:
    spec = importlib.util.spec_from_file_location(
        "gc", str(REPO / "scripts" / "eval_growth_count.py"))
    gc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gc)
    bio = BiochemConfig(phase="biochem")
    prot = json.load(open("outputs/ap_closure/protocol_gt_meanovertime.json"))
    names = prot["fit"] + prot["dev"]
    crit = float(bio.viscosity_mat_crit)
    lss, sgt = float(bio.lss), float(bio.sgt) / M_TO_CM
    coef = float(bio.L_char) * M_TO_CM / float(bio.gamma_m)
    OUT.mkdir(parents=True, exist_ok=True)

    ARMS = ["shipped", "GT on stitch nodes", "GT on ODE nodes", "GT on both (oracle)",
            "FLOOR (reassign)"]
    R = {a: {} for a in ARMS}
    frac = []
    for n in names:
        p = CACHE / f"{n}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        gt = z["gt_onset"]
        if not (gt >= 0).any():
            continue
        nt = len(z["t"])
        gate = (z["dsrx0"] < sgt) * coef * np.abs(z["dsrx0"]) + (z["sr0"] < lss)
        S = committed_set(gate, z["sr0"], z["wall_edges"])
        hook = make_rollout_hook(SHIPPED, bio, z["sr0"])
        traj, _ = integrate_mat_trajectory(gc.WallShim(z), bio, gate, da_scale=40.0,
                                           ap_closure=hook)
        idx = first_crossing(traj, crit)
        cr = idx >= 0
        med = int(np.median(idx[cr])) if cr.any() else 0

        ode_nodes = S & cr                      # the ODE genuinely times these
        stitch = S & ~cr                        # these get one constant
        frac.append(stitch.sum() / max(S.sum(), 1))

        def blend(use_gt_on):
            """Shipped onset, with ``use_gt_on`` nodes replaced by their GT onset."""
            out = np.where(S, np.where(cr, idx, med), -1)
            g_ok = use_gt_on & (gt >= 0)
            out[g_ok] = gt[g_ok]
            # a node with no GT onset keeps whatever it had
            return out

        arms = {
            "shipped": blend(np.zeros_like(S)),
            "GT on stitch nodes": blend(stitch),
            "GT on ODE nodes": blend(ode_nodes),
            "GT on both (oracle)": blend(S),
            "FLOOR (reassign)": count_optimal_onset(S, gt, nt),
        }
        for a in ARMS:
            R[a][n] = growth_error(arms[a], gt, nt)["growth_l1"]

    ok = sorted(R["shipped"])
    base = float(np.mean([R["shipped"][n] for n in ok]))
    floor = float(np.mean([R["FLOOR (reassign)"][n] for n in ok]))
    print("=" * 82)
    print("TIMING ERROR DECOMPOSITION   %d train vessels" % len(ok))
    print("=" * 82)
    print("   stitch nodes (constant onset) are %.0f%% of the predicted mask on average\n"
          % (100 * np.mean(frac)))
    print("%-26s %11s %12s %10s" % ("arm", "growth_l1", "vs shipped", "% of prize"))
    prize = base - floor
    for a in ARMS:
        v = float(np.mean([R[a][n] for n in ok]))
        print("%-26s %11.4f %+12.4f %9.0f%%"
              % (a, v, v - base, 100.0 * (base - v) / prize if prize > 0 else 0))
    print("\n   total timing prize on this mask: %+.4f" % -prize)
    s = base - float(np.mean([R["GT on stitch nodes"][n] for n in ok]))
    o = base - float(np.mean([R["GT on ODE nodes"][n] for n in ok]))
    print("\n   perfect timing on the STITCH nodes alone recovers %+.4f (%.0f%%)"
          % (-s, 100 * s / prize))
    print("   perfect timing on the ODE   nodes alone recovers %+.4f (%.0f%%)"
          % (-o, 100 * o / prize))
    print("\n   VERDICT: the residual timing error lives mostly in the %s."
          % ("STITCH -- fix how graph-grown nodes are timed" if s > o else
             "ODE-TIMED nodes -- the surface ODE itself mis-times the seeds"))

    (OUT / "timing_decomposition.json").write_text(json.dumps(
        dict(per_vessel=R, stitch_frac=float(np.mean(frac)), names=ok),
        indent=2, default=float), encoding="utf-8")
    print("\nwrote %s" % (OUT / "timing_decomposition.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
