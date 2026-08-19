"""Which LEVER moves the growth curve?  All of them, measured head to head.

PHASE6_RESULTS 12-13 pointed the next round at the wall-AP field.  That was reasoning from
half the evidence.  PHASE6_HANDOFF 9 already records the other half:

    perfect time-varying FLOW          -> onset rho 0.795
    perfect flow AND GT species        -> onset rho 0.866
    the physics model as it ships           ~0.60

Flow evolution alone carries +0.195 of ordering; perfect species on top of it adds only
+0.071.  So AP is the SMALLER lever, and an AP-only plan aims at the weaker half.

This panel measures every lever on the metric of record (mean-over-time), on the same
committed set, so only timing varies.  Deploy-legal levers are separated from oracles --
an oracle says whether a mechanism is worth modelling at all, a deploy-legal arm says what
we can actually ship.

DEPLOY-LEGAL (could ship today):
    ap closure      the algebraic wall-AP Damkohler balance
    hop delay       graph-grown mask nodes currently ALL receive the ODE's median onset --
                    measured, 15% of the mask (up to 26% on some vessels) has a constant,
                    necessarily-wrong onset.  This propagates onset outward from the
                    igniting seeds instead, one scalar (delay per hop).
    thrombin        committed nodes source thrombin, thrombin activates platelets
                    (src/core_physics/thrombin_field.py, already written, unused here)
    self-blockage   the model's own clot perturbs the shear it sees, reopening/closing
                    gates over the run (src/core_physics/shear_redistribution.py)

ORACLES (ceilings, never deployable):
    ap oracle EARLY   GT ap at 10% of horizon, then frozen -- the ceiling for a STATIC
                      field predictor, which is the deploy-shaped version of 13's proposal
    ap oracle FULL    GT ap at every step
    gate oracle       the gate recomputed from GT flow at every step -- the flow lever
    onset oracle      GT onset directly, the metric ceiling

Every lever's own scalar and ``da_scale`` are selected on DEV.  SEALED is reported but
selects nothing.

    python scripts/diag_lever_panel.py
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
M_TO_CM = 100.0
GROW = 6
EARLY_FRAC = 0.10


def _ev():
    spec = importlib.util.spec_from_file_location(
        "ev", str(REPO / "scripts" / "eval_ap_closure_protocol.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------------- lever pieces

def gt_species_full(c, bio, mode):
    z = np.load(CACHE / f"{c['name']}.npz")
    rp0, ap0 = wall_platelet_constants(c["shim"], bio)
    w = c["wall_idx"]
    T = z["ap"].shape[0]
    ap, rp = np.tile(ap0, (T, 1)), np.tile(rp0, (T, 1))
    if mode == "early":
        k = max(1, int(round(EARLY_FRAC * (T - 1))))
        ap[:, w] = z["ap"][k][None, :]
        rp[:, w] = z["rp"][k][None, :]
    else:
        ap[:, w] = z["ap"]
        rp[:, w] = z["rp"]
    return rp, ap


def gt_gate_blockage(c, bio):
    """``blockage(mat, gate0, step)`` returning the gate recomputed from GT flow at step."""
    z = np.load(CACHE / f"{c['name']}.npz")
    if "dsrx_t" not in z.files:
        return None
    lss, sgt = float(bio.lss), float(bio.sgt) / M_TO_CM
    coef = float(bio.L_char) * M_TO_CM / float(bio.gamma_m)
    w = c["wall_idx"]
    n = len(c["w"])
    T = z["sr_t"].shape[0]
    gate_t = np.zeros((T, n))
    g = (z["dsrx_t"] < sgt) * coef * np.abs(z["dsrx_t"]) + (z["sr_t"] < lss)
    gate_t[:, w] = g
    crit = float(bio.viscosity_mat_crit)

    def blockage(mat, gate0, step):
        gg = gate_t[min(int(step), T - 1)]
        return np.where(mat >= crit, np.maximum(gg, gate0), gg)      # committed keeps depositing

    return blockage


def hop_onset(c, idx, delay, nt):
    """Propagate onset outward from the igniting seeds instead of flattening to the median.

    The shipped mask is the gate-open set grown ``GROW`` hops along the mesh, but the ODE
    only ever fires on gate-open nodes -- so every grown node inherits ``median(onset)``,
    a constant.  That is a guaranteed ordering error on 15-26% of the mask and it also
    compresses the spread.  Here each grown node ignites ``delay`` steps after whichever
    committed neighbour reaches it first, which is what a propagating clot front does.
    """
    crossed = (idx >= 0) & c["w"]
    val = np.where(crossed, idx.astype(np.float64), np.inf)
    coo = c["adj"].tocoo()
    src, dst = coo.row, coo.col
    for _ in range(GROW):
        prev = val.copy()
        np.minimum.at(val, dst, prev[src] + float(delay))
    fin = np.isfinite(val)
    out = np.where(fin, np.clip(np.round(val), 0, nt - 1), -1).astype(int)
    med = int(np.median(idx[crossed])) if crossed.any() else 0
    return np.where(c["S"], np.where(out >= 0, out, med), -1)


def thrombin_boost(c, bio, gain):
    from src.core_physics.thrombin_field import make_ap_boost, make_thrombin_solver

    solve, _ = make_thrombin_solver(c["shim"], bio, c["pos"], c["sr"], wall=c["w"])
    return make_ap_boost(solve, bio, gain=gain)


def self_blockage(c, bio, wake):
    from src.core_physics.shear_redistribution import (
        build_crosssection_operator, make_blockage, sdf_nd,
    )

    sdf = sdf_nd(c["shim"])
    B = build_crosssection_operator(c["pos"], sdf, c["w"])
    return make_blockage(c["fields"], bio, B, c["w"], feedback="wake", wake=wake)


# ------------------------------------------------------------------------------- arms

def run_arm(c, bio, spec, da):
    """Return the onset index array for one lever configuration."""
    nt = len(c["t"])
    if spec["kind"] == "onset_oracle":
        return np.where(c["S"], np.where(c["gt_onset"] >= 0, c["gt_onset"], nt - 1), -1)

    species = None
    if spec.get("ap_oracle"):
        species = gt_species_full(c, bio, spec["ap_oracle"])
    hook = None
    if spec.get("closure"):
        hook = make_rollout_hook(spec["closure"], bio, c["sr"])
    blockage = None
    if spec.get("gate_oracle"):
        blockage = gt_gate_blockage(c, bio)
    elif spec.get("wake"):
        blockage = self_blockage(c, bio, spec["wake"])
    boost = thrombin_boost(c, bio, spec["gain"]) if spec.get("gain") else None

    traj, _ = integrate_mat_trajectory(c["shim"], bio, c["gate"], da_scale=da,
                                       species=species, ap_closure=hook,
                                       blockage=blockage, ap_boost=boost)
    idx = first_crossing(traj, float(bio.viscosity_mat_crit))
    if spec.get("hop") is not None:
        return hop_onset(c, idx, spec["hop"], nt)
    crossed = (idx >= 0) & c["w"]
    med = int(np.median(idx[crossed])) if crossed.any() else 0
    return np.where(c["S"], np.where(idx >= 0, idx, med), -1)


def main() -> int:
    ev = _ev()
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    prot = json.load(open(OUT / "protocol_gt_meanovertime.json"))
    fit, dev, sealed = prot["fit"], prot["dev"], prot["sealed"]
    cl = ApClosure(C=prot["best_cl"]["C"], q=1.0, kernel=prot["best_cl"]["kernel"])
    t0 = time.time()

    ctx = {}
    for n in fit + dev + sealed:
        c = ev.build_context(n, bio, phys, "gt")
        if c is not None and (CACHE / f"{n}.npz").exists():
            ctx[n] = c
    print("contexts %d in %.0fs\n" % (len(ctx), time.time() - t0))

    DA = [20.0, 40.0, 80.0, 160.0]
    HOP = [1, 2, 4, 8, 16]
    GAIN = [1.0, 4.0, 12.0]
    WAKE = [0.5, 1.0, 2.0]

    # (label, list of candidate specs to select among on DEV, deployable?)
    CAND = [
        ("baseline", [dict(kind="ode")], True),
        ("ap closure", [dict(kind="ode", closure=cl)], True),
        ("hop delay", [dict(kind="ode", hop=h) for h in HOP], True),
        ("thrombin", [dict(kind="ode", gain=g) for g in GAIN], True),
        ("self-blockage", [dict(kind="ode", wake=w) for w in WAKE], True),
        ("closure+hop", [dict(kind="ode", closure=cl, hop=h) for h in HOP], True),
        ("thrombin+hop", [dict(kind="ode", gain=g, hop=h) for g in GAIN for h in HOP], True),
        ("closure+thromb+hop",
         [dict(kind="ode", closure=cl, gain=g, hop=h) for g in GAIN for h in HOP], True),
        ("ap oracle EARLY", [dict(kind="ode", ap_oracle="early")], False),
        ("ap oracle FULL", [dict(kind="ode", ap_oracle="full")], False),
        ("gate oracle", [dict(kind="ode", gate_oracle=True)], False),
        ("gate oracle+hop", [dict(kind="ode", gate_oracle=True, hop=h) for h in HOP], False),
        ("gate+ap oracle", [dict(kind="ode", gate_oracle=True, ap_oracle="full")], False),
        ("gate+ap+hop", [dict(kind="ode", gate_oracle=True, ap_oracle="full", hop=h)
                         for h in HOP], False),
        ("onset oracle", [dict(kind="onset_oracle")], False),
    ]

    print("=" * 100)
    print("DEV SELECTION per lever (mean-over-time; each lever's own scalar and da_scale)")
    print("=" * 100)
    chosen = {}
    for lbl, specs, deployable in CAND:
        best = None
        das = [40.0] if lbl == "onset oracle" else DA
        for spec in specs:
            for da in das:
                try:
                    r = {n: ev.arm_metrics(ctx[n], run_arm(ctx[n], bio, spec, da)) for n in dev}
                except Exception as exc:                             # noqa: BLE001
                    print("   %-20s FAILED: %s" % (lbl, exc))
                    best = None
                    break
                s = float(np.mean([v["score"] for v in r.values()]))
                if best is None or s > best[0]:
                    best = (s, spec, da)
            if best is None:
                break
        if best is None:
            continue
        chosen[lbl] = dict(spec=best[1], da=best[2], dev=best[0], deployable=deployable)
        extra = " ".join("%s=%s" % (k, v) for k, v in best[1].items()
                         if k not in ("kind", "closure"))
        print("   %-20s da=%-6.0f %-28s DEV mean %.4f" % (lbl, best[2], extra, best[0]))

    # ------------------------------------------------------------------------- report
    held = [n for n in ctx if n in sealed or n in fit]
    print("\n" + "=" * 100)
    print("LEVER PANEL -- %d never-selected-on vessels (SEALED %d + FIT %d)"
          % (len(held), len(sealed), len(fit)))
    print("=" * 100)
    print("%-20s %4s %8s %8s %8s | %8s %8s %8s | %8s"
          % ("lever", "dep", "mean", "early", "late", "curveL1", "rho", "spread", "vs base"))
    rows, base = {}, None
    for lbl in [c[0] for c in CAND]:
        if lbl not in chosen:
            continue
        ch = chosen[lbl]
        R = {n: ev.arm_metrics(ctx[n], run_arm(ctx[n], bio, ch["spec"], ch["da"])) for n in held}
        g = lambda k: float(np.nanmean([R[n][k] for n in held]))          # noqa: E731
        row = dict(mean=g("score"), early=g("score_early"), late=g("score_late"),
                   curve=g("curve_l1"), rho=g("rho"), spread=g("spread_ratio"),
                   da=ch["da"], deployable=ch["deployable"], dev=ch["dev"],
                   per_vessel={n: R[n] for n in held})
        if base is None:
            base = row["mean"]
        rows[lbl] = row
        print("%-20s %4s %8.4f %8.4f %8.4f | %8.4f %+8.3f %8.3f | %+8.4f"
              % (lbl, "yes" if ch["deployable"] else "--", row["mean"], row["early"],
                 row["late"], row["curve"], row["rho"], row["spread"], row["mean"] - base))

    prize = rows["onset oracle"]["mean"] - base
    dep = {k: v for k, v in rows.items() if v["deployable"] and k != "baseline"}
    print("\n   prize (perfect onset): %+.4f" % prize)
    if dep:
        bd = max(dep, key=lambda k: dep[k]["mean"])
        print("   best DEPLOYABLE lever: %-20s %+.4f  (%.0f%% of prize)"
              % (bd, dep[bd]["mean"] - base,
                 100.0 * (dep[bd]["mean"] - base) / prize if abs(prize) > 1e-9 else 0.0))
    print("\n   GO/NO-GO bar (PHASE6_RESULTS 13): rho > 0.60 at spread_ratio > 0.4")
    for lbl, r in rows.items():
        if lbl == "baseline":
            continue
        ok = r["rho"] > 0.60 and r["spread"] > 0.4
        print("      %-20s rho %+.3f spread %.3f  -> %s%s"
              % (lbl, r["rho"], r["spread"], "CLEARS" if ok else "fails",
                 "" if r["deployable"] else "   (oracle)"))

    (OUT / "lever_panel.json").write_text(
        json.dumps(dict(rows={k: {kk: vv for kk, vv in v.items() if kk != "per_vessel"}
                              for k, v in rows.items()},
                        per_vessel={k: v["per_vessel"] for k, v in rows.items()},
                        chosen={k: dict(da=v["da"], dev=v["dev"],
                                        spec={kk: (str(vv) if kk == "closure" else vv)
                                              for kk, vv in v["spec"].items()})
                                for k, v in chosen.items()},
                        held_out=held), indent=2, default=float), encoding="utf-8")
    print("\nwrote %s   (%.0fs)" % (OUT / "lever_panel.json", time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
