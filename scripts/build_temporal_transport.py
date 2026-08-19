"""Time-resolved advective transport of the wall source -- per-time physics for the head.

WHY.  The time-conditioned head (`docs/PHASE9_ML.md` 13.9) is given exactly two pieces of
time-varying information: the query time itself, and a **binary** "has the ODE fired by now"
at this node.  Everything else it sees is a t=0 static field.  Off the wall it is given
nothing time-varying at all, because the ODE is a wall object -- and that is precisely where
the temporal arm is weakest (mean-over-time off-wall 0.649 against an oracle 0.84).

PHASE9 12.2 tried to fix this with the owner-threshold rule (an off-wall node fires when its
owner crosses `crit/att`) and measured it **worse than doing nothing** (0.490 against
0.5015), diagnosing that the ODE's `Mat` is biased low so `crit/att` is unreachable.  That
diagnosis is about a hand-written threshold rule.  It says nothing about handing the model
the underlying field and letting it calibrate.

WHAT THIS COMPUTES.  The transport operator of `src/clot_ml/transport.py` is **linear and
time-independent** -- the flow is frozen at t=0, so only the source changes with time.  So
the whole time-resolved off-wall field costs one solve per stored time:

    mat_adv(t) = L^-1 [ Mat_ODE(t) restricted to the wall ]

which is the physics' own answer to "how much deposited species has reached this off-wall
node by time t", under COMSOL's own operator (PHASE7 1.1: `dMat/dt + u.grad(Mat) = 0`,
zero diffusion, wall flux BC).  Alongside it, the two wall-side per-time quantities the head
also never saw: the node's own ODE `Mat(t)` and its owner's, as continuous values rather
than the single fired/not-fired bit.

    python scripts/build_temporal_transport.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.clot_ml.data import load_cache  # noqa: E402
from src.clot_ml.temporal import ode_trajectory  # noqa: E402
from src.clot_ml.transport import _node_volume, _solve_upwind, upwind_operator  # noqa: E402
from src.config import BiochemConfig  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
OUT = REPO / "outputs/temporal_transport"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-times", type=int, default=11)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    bio = BiochemConfig(phase="biochem")
    cache = load_cache("gt")

    for a, S in sorted(cache.items()):
        dst = OUT / f"{a}.npz"
        if dst.exists() and not args.force:
            print("[skip] %s" % a, flush=True)
            continue
        t0 = time.time()
        d = torch.load(PACKS / f"{a}.pt", map_location="cpu", weights_only=False)
        T = int(d.y.shape[0])
        times = [int(round(x)) for x in np.linspace(0, T - 1, args.n_times)]
        traj, _ = ode_trajectory(d, bio, flow="gt")           # [T, N], wall-supported

        wall, ei, owner = S["wall"], S["edge_index"], S["owner"]
        pos = S["pos"].astype(np.float64)
        u, v = S["u"].astype(np.float64), S["v"].astype(np.float64)
        L = float(np.ptp(pos[:, 0]) + np.ptp(pos[:, 1]))
        H = L / (float(np.median(np.hypot(u, v)[~wall])) + 1e-12)

        # one factorisation-worth of work per time; the operator itself never changes
        F, out = upwind_operator(pos, ei, u, v)
        vol = _node_volume(pos, ei)
        adv = np.zeros((len(times), len(wall)), dtype=np.float32)
        own = np.zeros_like(adv)
        slf = np.zeros_like(adv)
        for j, ti in enumerate(times):
            src = np.zeros(len(wall))
            src[wall] = np.maximum(traj[ti][wall], 0.0)
            adv[j] = _solve_upwind(F, out, src * vol, vol, H).astype(np.float32)
            own[j] = traj[ti][owner].astype(np.float32)
            slf[j] = traj[ti].astype(np.float32)
        np.savez_compressed(dst, times=np.array(times), T=T,
                            mat_adv_t=adv, mat_owner_t=own, mat_self_t=slf)
        print("[ok  ] %-11s T=%d  %d times  %.1fs" % (a, T, len(times), time.time() - t0),
              flush=True)
    print("done -> %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
