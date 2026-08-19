"""What is a PERFECT wall-AP field worth in the rollout?  The precondition for the GNN.

PHASE6_RESULTS 13 proposes supervising a graph model on the early-time ``ap`` field, on the
strength of a PROXY: ``rho(gate*ap_early, onset)`` is -0.877 against the closure's -0.748.
A proxy is not a ceiling.  Before any network is written, substitute the TRUE field into
the rollout and measure what it actually buys on the metric of record.

This is the same discipline that produced the +0.099 prize number: measure the ceiling, then
decide whether to build.  If a perfect ``ap`` field does not move mean-over-time, no model
that predicts ``ap`` can, and 13's recommendation is dead regardless of architecture.

ARMS (all share the committed set, so only timing varies):
    frozen ap        ap = ap0 everywhere, constant           -- what ships today
    ap closure       the algebraic 1/(1 + C*consumption/sr)  -- PHASE6_RESULTS 4
    ap oracle EARLY  GT ap at 10% of horizon, then FROZEN    -- what a STATIC field model
                                                                could reach at best
    ap oracle FULL   GT ap at every timestep                 -- the full chemistry oracle
    onset oracle     GT onset directly                       -- the metric ceiling

``ap oracle EARLY`` is the one that matters: it is deploy-shaped (one field, predicted once,
no recurrence), which is exactly the failure mode 4 says to avoid.  ``ap oracle FULL`` says
how much more a time-resolved model could add on top.

``da_scale`` is re-selected on DEV per arm -- suppressing ``ap`` lowers the effective rate,
so holding da fixed would penalise the oracle for a level shift rather than for its field.

GT species are ORACLE input: legal for a ceiling, never for a deployable arm.

    python scripts/diag_ap_oracle_ceiling.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import importlib.util  # noqa: E402

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.ap_closure import ApClosure, make_rollout_hook  # noqa: E402
from src.core_physics.physics_wall_model import (  # noqa: E402
    first_crossing, integrate_mat_trajectory, wall_platelet_constants,
)

CACHE = Path("outputs/wall_species_cache")
OUT = Path("outputs/ap_closure")
DA = [20.0, 40.0, 80.0, 160.0, 320.0]
EARLY_FRAC = 0.10


def _ev():
    spec = importlib.util.spec_from_file_location(
        "ev", str(REPO / "scripts" / "eval_ap_closure_protocol.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def gt_species_full(c, bio, *, mode: str):
    """Lift the cache's wall-only GT species to full-node arrays for the integrator.

    Off-wall nodes carry ``gate == 0`` so their value is irrelevant; they are filled with
    the t=0 constant rather than left at zero so nothing downstream can divide by it.
    """
    z = np.load(CACHE / f"{c['name']}.npz")
    rp0, ap0 = wall_platelet_constants(c["shim"], bio)
    w = c["wall_idx"]
    T = z["ap"].shape[0]
    if mode == "early":
        k = max(1, int(round(EARLY_FRAC * (T - 1))))
        ap = np.tile(ap0, (T, 1))
        rp = np.tile(rp0, (T, 1))
        ap[:, w] = z["ap"][k][None, :]
        rp[:, w] = z["rp"][k][None, :]
        return rp, ap
    ap = np.tile(ap0, (T, 1))
    rp = np.tile(rp0, (T, 1))
    ap[:, w] = z["ap"]
    rp[:, w] = z["rp"]
    return rp, ap


def onset_of(c, bio, *, species=None, closure=None, da=40.0):
    hook = make_rollout_hook(closure, bio, c["sr"]) if closure is not None else None
    traj, _ = integrate_mat_trajectory(c["shim"], bio, c["gate"], da_scale=da,
                                       species=species, ap_closure=hook)
    idx = first_crossing(traj, float(bio.viscosity_mat_crit))
    crossed = (idx >= 0) & c["w"]
    med = int(np.median(idx[crossed])) if crossed.any() else 0
    return np.where(c["S"], np.where(idx >= 0, idx, med), -1)


def main() -> int:
    ev = _ev()
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    prot = json.load(open(OUT / "protocol_gt_meanovertime.json"))
    fit, dev, sealed = prot["fit"], prot["dev"], prot["sealed"]
    t0 = time.time()

    ctx = {}
    for n in fit + dev + sealed:
        c = ev.build_context(n, bio, phys, "gt")
        if c is not None and (CACHE / f"{n}.npz").exists():
            ctx[n] = c
    print("contexts %d in %.0fs" % (len(ctx), time.time() - t0))

    cl = ApClosure(C=prot["best_cl"]["C"], q=1.0, kernel=prot["best_cl"]["kernel"])
    species = {}
    for n in ctx:
        species[n] = dict(early=gt_species_full(ctx[n], bio, mode="early"),
                          full=gt_species_full(ctx[n], bio, mode="full"))

    def run(n, arm, da):
        c = ctx[n]
        if arm == "frozen ap":
            return onset_of(c, bio, da=da)
        if arm == "ap closure":
            return onset_of(c, bio, closure=cl, da=da)
        if arm == "onset oracle":
            return np.where(c["S"], np.where(c["gt_onset"] >= 0, c["gt_onset"],
                                             len(c["t"]) - 1), -1)
        return onset_of(c, bio, species=species[n]["early" if "EARLY" in arm else "full"], da=da)

    ARMS = ["frozen ap", "ap closure", "ap oracle EARLY", "ap oracle FULL", "onset oracle"]

    # ---------------------------------------------------------- da re-selected on DEV
    print("\n" + "=" * 86)
    print("da_scale re-selected on DEV per arm (a suppressed ap is a level shift, not a defect)")
    print("=" * 86)
    best_da = {}
    for arm in ARMS:
        if arm == "onset oracle":
            best_da[arm] = 40.0
            continue
        scores = {}
        for da in DA:
            r = {n: ev.arm_metrics(ctx[n], run(n, arm, da)) for n in dev}
            scores[da] = float(np.mean([v["score"] for v in r.values()]))
        best_da[arm] = max(DA, key=lambda d: scores[d])
        print("   %-17s %s   -> da=%.0f"
              % (arm, "  ".join("da%-4.0f %.4f" % (d, scores[d]) for d in DA), best_da[arm]))

    # ------------------------------------------------------------------------- results
    R = {a: {n: ev.arm_metrics(ctx[n], run(n, a, best_da[a])) for n in ctx} for a in ARMS}
    names_ho = [n for n in ctx if n in sealed or n in fit]
    print("\n" + "=" * 86)
    print("WHAT A PERFECT ap FIELD IS WORTH   (%d never-selected-on vessels)" % len(names_ho))
    print("=" * 86)
    print("%-17s %8s %8s %8s | %8s %8s %8s"
          % ("arm", "mean", "early", "late", "curveL1", "rho", "spread"))
    base = None
    rows = {}
    for a in ARMS:
        g = lambda k: float(np.nanmean([R[a][n][k] for n in names_ho]))    # noqa: E731
        row = dict(mean=g("score"), early=g("score_early"), late=g("score_late"),
                   curve=g("curve_l1"), rho=g("rho"), spread=g("spread_ratio"),
                   da=best_da[a])
        rows[a] = row
        if base is None:
            base = row["mean"]
        print("%-17s %8.4f %8.4f %8.4f | %8.4f %+8.3f %8.3f   %+.4f"
              % (a, row["mean"], row["early"], row["late"], row["curve"], row["rho"],
                 row["spread"], row["mean"] - base))

    prize = rows["onset oracle"]["mean"] - base
    print("\n   prize (perfect onset)            %+.4f" % prize)
    for a in ("ap closure", "ap oracle EARLY", "ap oracle FULL"):
        got = rows[a]["mean"] - base
        print("   %-32s %+.4f   (%3.0f%% of prize)"
              % (a, got, 100.0 * got / prize if abs(prize) > 1e-9 else 0.0))

    print("\n   GO/NO-GO for a learned ap-field model (PHASE6_RESULTS 13):")
    e = rows["ap oracle EARLY"]
    print("      it must clear rho > 0.60 at spread_ratio > 0.4.")
    print("      a PERFECT early ap field reaches rho %+.3f at spread %.3f -> %s"
          % (e["rho"], e["spread"],
             "HEADROOM EXISTS" if (e["rho"] > 0.60 and e["spread"] > 0.4) else
             "CEILING IS BELOW THE BAR -- do not build the model"))

    (OUT / "ap_oracle_ceiling.json").write_text(
        json.dumps(dict(rows=rows, per_vessel={a: R[a] for a in ARMS},
                        held_out=names_ho), indent=2, default=float), encoding="utf-8")
    print("\nwrote %s   (%.0fs)" % (OUT / "ap_oracle_ceiling.json", time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
