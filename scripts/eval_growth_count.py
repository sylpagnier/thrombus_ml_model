"""How much growth-curve error does the zero-parameter physics model actually carry?

Metric: ``growth_l1 = mean_t |n_pred(t) - n_gt(t)| / N_gt_final`` -- see
``src/core_physics/growth_count_metrics.py`` for why the overlap score was retired for this
question (it is a cliff, discontinuous in commit time, and it ranked the best growth curve
last).

THE DECOMPOSITION IS THE POINT.  Three numbers bound the whole problem:

    shipped model      what the zero-parameter physics ODE delivers today
    count floor        the best ANY timing model can do on this committed set -- the
                       k-th node committing exactly when GT's k-th node does.  What is
                       left at the floor is pure mask-size error, which no onset model
                       can touch.
    mask-only error    the floor's own residual, i.e. |S| vs N_gt

``shipped - floor`` is the entire prize for onset work.  If it is small, the timing problem
is closed and the remaining error is a MASK problem; if it is large, timing is still worth
attacking and the earlier null results were the old metric's fault.

Runs entirely off ``outputs/wall_species_cache`` -- the wall subgraph reproduces the shipped
mask exactly (a non-wall node can never be admitted by the growth rule), so no 300 MB pack
is loaded and the whole cohort takes seconds.

    python scripts/eval_growth_count.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig  # noqa: E402
from src.core_physics.ap_closure import ApClosure, make_rollout_hook  # noqa: E402
from src.core_physics.growth_count_metrics import (  # noqa: E402
    count_optimal_onset, growth_error,
)
from src.core_physics.onset_features import committed_set, hop_distance  # noqa: E402
from src.core_physics.physics_wall_model import (  # noqa: E402
    first_crossing, integrate_mat_trajectory,
)

CACHE = Path("outputs/wall_species_cache")
OUT = Path("outputs/growth_count")
BULK_ND = 2.5e14
M_TO_CM = 100.0


class WallShim:
    """Wall-only stand-in so the REAL integrator runs without loading the pack.

    ``wall_platelet_constants`` reads ``y[0]`` and converts nd -> CGS with the bulk scale,
    so the cached CGS values are inverted back through the same constant.  Verified against
    the full-mesh rollout below.
    """

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


def ode_onset(z, bio, gate, S, *, da=40.0, closure=None):
    hook = make_rollout_hook(closure, bio, z["sr0"]) if closure is not None else None
    traj, _ = integrate_mat_trajectory(WallShim(z), bio, gate, da_scale=da, ap_closure=hook)
    idx = first_crossing(traj, float(bio.viscosity_mat_crit))
    crossed = idx >= 0
    med = int(np.median(idx[crossed])) if crossed.any() else 0
    return np.where(S, np.where(idx >= 0, idx, med), -1)


def main() -> int:
    bio = BiochemConfig(phase="biochem")
    OUT.mkdir(parents=True, exist_ok=True)
    prot = json.load(open("outputs/ap_closure/protocol_gt_meanovertime.json"))
    fit, dev, sealed = prot["fit"], prot["dev"], prot["sealed"]
    C = float(prot["best_cl"]["C"])
    lss, sgt = float(bio.lss), float(bio.sgt) / M_TO_CM
    coef = float(bio.L_char) * M_TO_CM / float(bio.gamma_m)

    ARMS = ["shipped physics", "+ AP closure", "+ hop delay", "+ global shift (oracle)",
            "COUNT FLOOR (mask)", "onset oracle"]
    R = {a: {} for a in ARMS}
    meta = {}
    for n in sorted(set(fit + dev + sealed)):
        p = CACHE / f"{n}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        sr0, dsrx0, gt = z["sr0"], z["dsrx0"], z["gt_onset"]
        nt = len(z["t"])
        if not (gt >= 0).any():
            continue
        gate = (dsrx0 < sgt) * coef * np.abs(dsrx0) + (sr0 < lss)
        S = committed_set(gate, sr0, z["wall_edges"])
        base = ode_onset(z, bio, gate, S)
        arms = {"shipped physics": base}
        arms["+ AP closure"] = ode_onset(
            z, bio, gate, S, closure=ApClosure(C=C, q=1.0, kernel=prot["best_cl"]["kernel"]))
        # hop delay: grown nodes inherit nearest seed's onset + 16 steps per hop (the DEV
        # value from the lever panel) instead of the ODE's global median
        hop = hop_distance(gate > 0, z["wall_edges"])
        seeded = base.copy()
        grown = S & (hop > 0)
        seed_on = float(np.median(base[S & (hop == 0)])) if (S & (hop == 0)).any() else 0.0
        seeded[grown] = np.clip(seed_on + 16.0 * hop[grown], 0, nt - 1).astype(int)
        arms["+ hop delay"] = seeded
        gt_first = int(gt[gt >= 0].min())
        m_first = int(base[base >= 0].min()) if (base >= 0).any() else 0
        arms["+ global shift (oracle)"] = np.where(
            base >= 0, np.clip(base + (gt_first - m_first), 0, nt - 1), -1)
        arms["COUNT FLOOR (mask)"] = count_optimal_onset(S, gt, nt)
        arms["onset oracle"] = np.where(S, np.where(gt >= 0, gt, nt - 1), -1)
        for a in ARMS:
            R[a][n] = growth_error(arms[a], gt, nt)
        meta[n] = dict(n_wall=int(len(sr0)), n_S=int(S.sum()), n_gt=int((gt >= 0).sum()),
                       sealed=n in sealed)

    def agg(names, arm, key="growth_l1"):
        v = np.array([R[arm][n][key] for n in names if n in R[arm]], float)
        v = v[np.isfinite(v)]
        return float(np.mean(v)) if len(v) else float("nan")

    groups = [("train (FIT+DEV)", [n for n in fit + dev if n in meta]),
              ("SEALED", [n for n in sealed if n in meta])]
    for tag, names in groups:
        if not names:
            continue
        print("\n" + "=" * 92)
        print("%s   n=%d      growth_l1 = mean_t |n_pred - n_gt| / N_gt_final  (0 = perfect)"
              % (tag, len(names)))
        print("=" * 92)
        print("%-26s %10s %10s %10s %10s" % ("arm", "growth_l1", "worst_t", "final_err", "vs floor"))
        floor = agg(names, "COUNT FLOOR (mask)")
        for a in ARMS:
            print("%-26s %10.4f %10.4f %+10.4f %+10.4f"
                  % (a, agg(names, a), agg(names, a, "growth_linf"),
                     agg(names, a, "final_err"), agg(names, a) - floor))
        ship = agg(names, "shipped physics")
        print("\n   shipped %.4f | count floor %.4f | PRIZE for any timing model %+.4f (%.0f%%)"
              % (ship, floor, ship - floor, 100.0 * (ship - floor) / max(ship, 1e-9)))
        print("   irreducible mask-size error at the floor: %.4f (%.0f%% of the shipped error)"
              % (floor, 100.0 * floor / max(ship, 1e-9)))

    names = [n for n in fit + dev if n in meta]
    print("\n" + "=" * 92)
    print("PER-VESSEL, train  (final_err > 0 means the model commits MORE nodes than GT)")
    print("=" * 92)
    print("%-12s %5s %5s %5s | %9s %9s %9s | %9s"
          % ("vessel", "wall", "|S|", "n_gt", "shipped", "floor", "prize", "final_err"))
    for n in names:
        print("%-12s %5d %5d %5d | %9.4f %9.4f %9.4f | %+9.3f"
              % (n, meta[n]["n_wall"], meta[n]["n_S"], meta[n]["n_gt"],
                 R["shipped physics"][n]["growth_l1"], R["COUNT FLOOR (mask)"][n]["growth_l1"],
                 R["shipped physics"][n]["growth_l1"] - R["COUNT FLOOR (mask)"][n]["growth_l1"],
                 R["shipped physics"][n]["final_err"]))

    (OUT / "growth_count.json").write_text(json.dumps(
        dict(per_vessel={a: R[a] for a in ARMS}, meta=meta, fit=fit, dev=dev, sealed=sealed),
        indent=2, default=float), encoding="utf-8")
    print("\nwrote %s" % (OUT / "growth_count.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
