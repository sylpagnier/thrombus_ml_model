"""Strictly-nested evaluation of the time-conditioned head: mean-over-time AND final time.

`scripts/train_time_conditioned.py` produced the numbers `docs/PHASE9_ML.md` 13.9 reports
and that `clot_gnn_v3` ships.  It has three selection leaks, all removable without
retraining the GNN:

  1. the committed SET uses hard-coded cuts `score >= 0.73` / `>= 0.92`, chosen on the whole
     pool;
  2. the per-domain time thresholds are tuned on the fold's own training vessels using
     **that fold's model**, whose scores on those vessels are in-sample;
  3. the head itself is fitted on the same in-sample scores it is later applied to
     out-of-fold, so its `score` feature has a different distribution at train and test
     time -- a subtle one, and it always flatters training.

Here every one of the four readout scalars and the head are selected inside a properly
nested loop:

    outer fold k          held-out vessels, never touched by anything below
      selection set       the other 14-15 vessels, each carrying ITS OWN out-of-fold score
      inner 3-fold        head fitted on inner-train, predicted on inner-val
      thresholds          tuned on those inner out-of-fold predictions
      final head          refitted on the whole selection set, applied to fold k

So the `score` feature is out-of-fold at train time as well as at test time, and no vessel's
reported number depends on a quantity fitted while it was visible.

Two metrics are reported, because they answer different questions and the project has only
ever quoted the first:

    mean-over-time   the average severity score over the 11-point time grid, GT-empty
                     timesteps skipped (`SeverityScorer` returns NaN there)
    FINAL            the score at the last timestep -- the fully-formed clot, which is what
                     a reader of the prediction actually acts on

    python scripts/eval_strict_temporal.py --tags cv5a,cv5b,cv5c --cache gt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

from src.clot_ml.data import attach_physics, load_cache  # noqa: E402
from src.clot_ml.geometry_splits import classes_for, is_priority  # noqa: E402
from src.clot_ml.severity_metric import DEFAULT, SeverityScorer  # noqa: E402
from src.clot_ml.temporal import ode_trajectory  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
SET_GRID = np.array([0.30, 0.45, 0.60, 0.70, 0.80, 0.88, 0.94])
TIME_GRID = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
HEAD = dict(max_iter=300, max_depth=5, learning_rate=0.07, l2_regularization=1.0,
            class_weight="balanced", random_state=0)


TT = REPO / "outputs/temporal_transport"


def load_temporal_transport(anchor, times, crit):
    """Per-(node, time) physics channels; ``None`` when the cache has not been built.

    ``scripts/build_temporal_transport.py`` writes these.  They are the only time-varying
    inputs the head gets besides the query time and the ODE's fired/not-fired bit, and the
    only time-varying input it has ever had OFF the wall.
    """
    p = TT / f"{anchor}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    if list(z["times"]) != list(times):
        raise ValueError("%s: temporal_transport time grid does not match" % anchor)
    return {k: np.log1p(np.maximum(z[k], 0.0) / crit).astype(np.float32)
            for k in ("mat_adv_t", "mat_owner_t", "mat_self_t")}


def precompute(pool, cache, n_times):
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    crit = float(bio.viscosity_mat_crit)
    V = {}
    for a in pool:
        S = cache[a]
        d = torch.load(PACKS / f"{a}.pt", map_location="cpu", weights_only=False)
        T = int(d.y.shape[0])
        times = [int(round(x)) for x in np.linspace(0, T - 1, n_times)]
        gt = {ti: (gt_clot_phi_at_time(d, ti, phys, device=torch.device("cpu"))
                   .reshape(-1).numpy() > 0.5) for ti in times}
        go = np.full(len(S["wall"]), T, dtype=int)          # GT onset index
        for ti in reversed(times):
            go[gt[ti]] = ti
        traj, t = ode_trajectory(d, bio, flow="gt")
        # The owner's crossing of c*crit, for several c.  Off-wall commits when
        # att*Mat_owner >= crit, i.e. when the owner reaches crit/att -- PHASE9 12.2 found
        # crit/att unreachable as a hard RULE because the ODE's Mat is biased low, but as
        # the ANCHOR of a learned residual an unreachable level is just a shifted clock, and
        # c=1 (the plain ODE crossing) is in the grid so this can only move if it pays.
        oon_c = {}
        for c in ANCHOR_C:
            hc = traj >= c * crit
            oon_c[c] = np.where(hc.any(0), hc.argmax(0), traj.shape[0])
        r0 = traj[1] / max(t[1] - t[0], 1e-9)               # t=0 deposition rate
        hot = traj >= crit
        oon = np.where(hot.any(0), hot.argmax(0), T)        # ODE crossing index
        tt = load_temporal_transport(a, times, crit)
        # fraction of the ODE's eventually-igniting nodes that have fired by each grid time
        fire = oon[oon < T]
        clock = [np.array([float((fire <= t_).mean()) if fire.size else 0.0
                           for t_ in times], dtype=np.float32)]
        if tt is not None:
            # same idea from the transport side: fraction of nodes the advected field has
            # taken past `crit` (log1p(Mat/crit) >= log 2) by each grid time
            hot_t = (tt["mat_adv_t"] >= np.log(2.0))
            clock.append(hot_t.mean(axis=1).astype(np.float32))
        # grid step at which the ADVECTED field crosses crit at this node -- the physics'
        # own direct prediction of when an off-wall node clots, with no owner indirection.
        if tt is not None:
            hot_a = tt["mat_adv_t"] >= float(np.log(2.0))
            t_adv = np.where(hot_a.any(0), hot_a.argmax(0), len(times)).astype(float)
        else:
            t_adv = np.full(len(S["wall"]), float(len(times)))
        V[a] = dict(S=S, T=T, times=times, gt=gt, go=go, oon=oon, r0=r0, clock=clock,
                    tt=tt, t_adv=t_adv, oon_c=oon_c,
                    scorer={ti: SeverityScorer(S["edge_index"], gt[ti], len(S["wall"]),
                                               DEFAULT) for ti in times})
        print("   [prep] %s T=%d" % (a, T), flush=True)
    return V


def node_features(V, a, oofs):
    """Static per-node block: the cached features, the t=0 rate, the ODE onset, and EVERY
    arm's out-of-fold score as its own column (the head can weigh them itself)."""
    v, S = V[a], V[a]["S"]
    cols = [S["X"], np.log1p(np.maximum(v["r0"], 0))[:, None], (v["oon"] / v["T"])[:, None]]
    cols += [oofs[arm][a][:, None] for arm in sorted(oofs)]
    return np.concatenate(cols, axis=1)


#: filled in by the two-stage pass: per anchor, the stage-1 predicted commit fraction of
#: each node's OWNER wall node at each grid time.  Empty -> single-stage behaviour.
OWNER_PRED: dict = {}
#: one-element box so the apply path can see the chosen lag anchor
LAG_ANCHOR = ["pred"]
#: owner-trajectory levels (multiples of crit) offered as the lag anchor; 1.0 = plain ODE
ANCHOR_C = [1.0, 2.0, 4.0, 8.0]
ANCHOR_LEVEL = [1.0]


def time_block(V, a, j, sel=None):
    """The per-(node, time) columns for grid index ``j``, optionally on a node subset.

    v3 had two: the normalised query time and the ODE's fired/not-fired bit.  The rest are
    the time-resolved physics of `scripts/build_temporal_transport.py` -- the node's own
    ODE `Mat(t)`, its owner's, and the advected off-wall field `mat_adv(t)`, all as
    continuous log values.  The advected channel is the first time-varying quantity the
    head has ever been given off the wall.
    """
    v = V[a]
    ti = v["times"][j]
    idx = slice(None) if sel is None else np.asarray(sel, dtype=bool)
    ode_now = (v["oon"][idx] <= ti).astype(np.float32).reshape(-1, 1)
    cols = [np.full((len(ode_now), 1), ti / v["T"], dtype=np.float32), ode_now]
    if v["tt"] is not None:
        for k in ("mat_self_t", "mat_owner_t", "mat_adv_t"):
            cols.append(v["tt"][k][j][idx].reshape(-1, 1))
    # --- the OWNER'S OWN PREDICTED STATE ----------------------------------------------
    # The head sees `mat_owner_t`, the owner's ODE `Mat(t)` -- but the ODE is biased low,
    # which is exactly why PHASE9 12.2's owner-threshold timing rule collapsed ("crit/att is
    # simply unreachable").  The MODEL's estimate of when the owner commits is much better
    # than the ODE's, and it is available: run the head once, read off the wall onsets, feed
    # them back.  This is the timing analogue of the `log_mat_owner` channel.
    if a in OWNER_PRED:
        op = OWNER_PRED[a]
        cols.append(op[j][idx].reshape(-1, 1))
        cols.append(op[-1][idx].reshape(-1, 1))
    # --- the PER-VESSEL PHYSICS CLOCK -------------------------------------------------
    # `ti / T` is a wall-clock fraction and says nothing about how far THIS vessel has got.
    # PHASE9 13.5 measured exactly this: replacing the ODE's per-vessel onset histogram with
    # a pooled cohort growth curve costs -0.081 wall, and concluded "the ODE's contribution
    # is not its ordering -- it is its per-vessel time CALIBRATION".  13.6/13.9 then showed a
    # known per-vessel schedule is worth +0.036 wall / +0.125 off over the ODE's.  These two
    # scalars are the deployable form of that calibration: how far along its OWN growth this
    # vessel is at `ti`, by the ODE and by the advected field, broadcast to every node.
    for c in v["clock"]:
        cols.append(np.full((len(ode_now), 1), c[j], dtype=np.float32))
    return np.concatenate(cols, axis=1)


def fit_head(V, anchors, oofs, set_th, seeds=1):
    """Fit P(clot at t) on the (node, time) table of ``anchors``.

    ``seeds > 1`` returns a list of heads whose probabilities are averaged.  Seed averaging
    is the one variance-reduction lever that has reliably paid on this cohort at every level
    it has been tried (`docs/PHASE9_ML.md` 4 for the GNN); the temporal head was the last
    place still fitting a single model.
    """
    Xs, ys = [], []
    for a in anchors:
        v, S = V[a], V[a]["S"]
        Xn = node_features(V, a, oofs)
        cand = candidate_mask(S, arm_scores(oofs, a), set_th, a) | (v["go"] < v["T"])
        Xc = Xn[cand]
        for j, ti in enumerate(v["times"]):
            Xs.append(np.concatenate([Xc, time_block(V, a, j, cand)], axis=1))
            ys.append(v["gt"][ti][cand])
    X, y = np.concatenate(Xs), np.concatenate(ys)
    out = []
    for s in range(max(int(seeds), 1)):
        cfg = dict(HEAD)
        cfg["random_state"] = s
        out.append(HistGradientBoostingClassifier(**cfg).fit(X, y))
    return out


def arm_scores(oofs, a):
    """``{arm: score array}`` for one anchor -- candidate_mask works per vessel."""
    return {arm: oofs[arm][a] for arm in oofs}


#: optional externally-supplied committed set, keyed by anchor.  When present it REPLACES
#: the family/threshold machinery below -- the set then comes from whichever readout won
#: the per-domain in-fold selection in `scripts/eval_expected_score_readout.py`, including
#: the expected-score readout, which this script cannot express as a family.
EXTERNAL_SET: dict = {}


def candidate_mask(S, sc_by_arm, set_th, anchor=None):
    if anchor is not None and anchor in EXTERNAL_SET:
        return EXTERNAL_SET[anchor]
    return _candidate_mask_family(S, sc_by_arm, set_th)


def _candidate_mask_family(S, sc_by_arm, set_th):
    """The committed SET -- which nodes ever clot.  ``set_th`` is ``{domain: (family, th)}``.

    The set is a statement about the FINAL mask, so it is tuned against final-time GT by
    the same code `scripts/eval_strict.py` uses, and both readout families are offered.

    The family is chosen **per domain**, which is legitimate because the metric is itself
    domain-restricted (`docs/PHASE9_ML.md` 0 already reports a domain-specialised ensemble)
    and necessary because the two domains disagree about it: measured strictly-nested, the
    wall prefers a plain cut (FIN 0.9167 against 0.9046) and off-wall prefers the
    physics-conditioned one (FIN 0.7075 against 0.6431).  On `patient032` a plain cut
    commits **nothing** off-wall (0.000 against 0.432) because its score is uniformly low
    there and the physics mask is the only thing separating its 120 off-wall nodes.
    """
    from eval_strict import FAMILIES
    w = S["wall"]
    aw, fw, tw = set_th["wall"]
    ao, fo, to = set_th["off"]
    return (w & FAMILIES[fw][1](S, sc_by_arm[aw], tw)) | (~w & FAMILIES[fo][1](S, sc_by_arm[ao], to))


def predict_series(V, a, m, oofs):
    """Monotone-in-time probability field ``[n_times, N]``.

    The cumulative maximum is the production law's own property: `J0_Mat >= 0` and there is
    no sink (PHASE7 12.1 measured wall `Mat` to be the exact time-integral of its own nodal
    derivative, rank 0.999), so clot does not un-clot.  A per-time classifier does not get
    that for free the way the onset formulation did.
    """
    v = V[a]
    ms = m if isinstance(m, list) else [m]
    Xn = node_features(V, a, oofs)
    P = np.zeros((len(v["times"]), Xn.shape[0]), dtype=np.float32)
    for j in range(len(v["times"])):
        row = np.concatenate([Xn, time_block(V, a, j)], axis=1)
        P[j] = np.mean([mm.predict_proba(row)[:, 1] for mm in ms], axis=0)
    return np.maximum.accumulate(P, axis=0)


LAG_GRID = list(range(0, 9))
#: predicted off-wall burden (committed nodes) above which the LEARNED lag is trusted.
#: 0 = always, a large value = never; both ends are in the grid so the in-fold tuner can
#: fall back to either pure rule.
BURDEN_GRID = [0, 8, 15, 25, 40, 60, 90, 10 ** 9]
#: committed WALL nodes above which the learned ODE-onset residual is trusted.  `None` is
#: "never" -- i.e. keep the probability rule, which is the arm this must beat.
WBURDEN_GRID = [None, 0, 40, 80, 150, 300]


def lag_features(V, a, oofs):
    """Node features for the lag regression, plus the PHYSICS' own predicted lag.

    The transport solve already answers the question the lag model is asking.  For each node
    it gives the grid step at which the advected field crosses `crit`, and for that node's
    owner the step at which the wall ODE does; their difference is the lag the physics
    predicts, with no fitting at all.  Handing the regression that residual target is much
    easier than making it rediscover the boundary-layer filling time from static features.
    """
    X = node_features(V, a, oofs)
    v, S = V[a], V[a]["S"]
    if v["tt"] is None:
        return X
    thr = float(np.log(2.0))
    n_t = len(v["times"])

    def first_cross(F):
        hot = F >= thr
        return np.where(hot.any(0), hot.argmax(0), n_t).astype(np.float32)

    t_adv = first_cross(v["tt"]["mat_adv_t"])          # when transport says THIS node fires
    t_own = first_cross(v["tt"]["mat_self_t"])[S["owner"]]   # ... and when its owner does
    return np.concatenate([X, t_adv[:, None], t_own[:, None],
                           (t_adv - t_own)[:, None]], axis=1)


def fit_lag_model(V, anchors, oofs, seeds=3, anchor="pred"):
    """Regress the per-node lag behind the owner, on off-wall GT nodes.

    `scripts/diag_offwall_structure.py` measures the lag distribution (median +4 of 11 grid
    steps, p25 +3, p75 +6, 88% >= 2) and a single cohort constant loses to the probability
    rule because of that spread.  But the lag is a PER-NODE quantity with **584 labelled
    examples** across the cohort -- two orders of magnitude more training signal than the 19
    vessel-level samples every other schedule idea in this project has had to work with.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    Xs, ys = [], []
    for a in anchors:
        v, S = V[a], V[a]["S"]
        off = (~S["wall"]) & (v["go"] < v["T"])
        if not off.any():
            continue
        # the GT lag in GRID steps: where each node sits relative to its owner
        gi = np.searchsorted(np.asarray(v["times"]), v["go"], side="left")
        lag = (gi - gi[S["owner"]])[off]
        Xs.append(lag_features(V, a, oofs)[off])
        ys.append(lag)
    if not Xs:
        return None
    X, y = np.concatenate(Xs), np.concatenate(ys)
    ms = [HistGradientBoostingRegressor(max_iter=200, max_depth=4, learning_rate=0.06,
                                        l2_regularization=1.0,
                                        random_state=s).fit(X, y)
          for s in range(max(int(seeds), 1))]
    return _LagEnsemble(ms)


def fit_wall_residual(V, anchors, oofs, seeds=3):
    """Regress GT wall onset MINUS the ODE's onset, in grid steps, on committed wall nodes.

    The off-wall gain came from a residual on a physics anchor (the owner's predicted onset)
    rather than from predicting an absolute time; this is the same construction on the wall,
    where the anchor is the zero-parameter ODE's own crossing.  `PHASE9` 13.5 measured that
    the ODE's contribution is its **per-vessel time calibration**, not its ordering, which is
    exactly what a residual keeps and an absolute prediction throws away.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    Xs, ys = [], []
    for a in anchors:
        v, S = V[a], V[a]["S"]
        on = S["wall"] & (v["go"] < v["T"])
        if not on.any():
            continue
        g = np.asarray(v["times"])
        gi = np.searchsorted(g, v["go"], side="left")
        oi = np.searchsorted(g, v["oon"], side="left")
        Xs.append(lag_features(V, a, oofs)[on])
        ys.append((gi - oi)[on])   # wall residual stays in grid steps (it is not de-quantised)
    if not Xs:
        return None
    X, y = np.concatenate(Xs), np.concatenate(ys)
    return _LagEnsemble([HistGradientBoostingRegressor(
        max_iter=200, max_depth=4, learning_rate=0.06, l2_regularization=1.0,
        random_state=s).fit(X, y) for s in range(max(int(seeds), 1))])


def wall_by_residual(V, a, gm, resid, n_t):
    """Wall mask series with onset = the ODE's grid onset + the learned residual."""
    v, S = V[a], V[a]["S"]
    oi = np.searchsorted(np.asarray(v["times"]), v["oon"], side="left")
    on = np.clip(oi + np.rint(resid).astype(int), 0, n_t)
    M = np.zeros((n_t, len(S["wall"])), dtype=bool)
    for j in range(n_t):
        M[j] = gm & S["wall"] & (on <= j)
    M[-1] = gm & S["wall"]
    return np.maximum.accumulate(M, axis=0)


class _LagEnsemble:
    """Seed-averaged lag regressor -- variance reduction is what this cohort rewards."""

    def __init__(self, models):
        self.models = models

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)


def offwall_by_learned_lag(M_wall, gm, owner, wall, lag_per_node, commit_final=True,
                           times=None, horizon=None):
    """As :func:`offwall_by_lag` but with a per-node lag instead of a cohort constant.

    ``times``/``horizon`` switch the lag from WHOLE GRID STEPS to a continuous fraction of
    the run.  The quantised form is why refining the regression was measured EXACTLY neutral
    to four decimals: the lag is rounded to one of eleven steps, so a better prediction never
    crosses a step boundary and the mask does not change.  Continuous lags are added to the
    owner's commit TIME and compared against the real grid times, so a predicted 0.36 T and a
    0.44 T land on different steps where 4-vs-4 steps could not.
    """
    T, N = M_wall.shape
    won = np.full(N, T, dtype=int)
    for j in range(T - 1, -1, -1):
        won[M_wall[j]] = j
    if times is None or horizon is None:
        on_idx = np.clip(won[owner] + np.rint(lag_per_node).astype(int), 0, T)
        M = np.zeros((T, N), dtype=bool)
        for j in range(T):
            M[j] = gm & ~wall & (on_idx <= j)
    else:
        tt = np.asarray(times, dtype=float)
        big = float(tt[-1]) + 10.0 * float(horizon)
        own_t = np.where(won[owner] < T, tt[np.clip(won[owner], 0, T - 1)], big)
        on_t = own_t + np.asarray(lag_per_node, dtype=float) * float(horizon)
        M = np.zeros((T, N), dtype=bool)
        for j in range(T):
            M[j] = gm & ~wall & (on_t <= tt[j])
    if commit_final:
        M[-1] = gm & ~wall
    return np.maximum.accumulate(M, axis=0)


def offwall_from_adv(V, a, gm, resid, n_t, commit_final=True):
    """Off-wall onset = the advected field's own crossing + a learned residual.

    The most direct physics anchor available: `mat_adv(t)` is COMSOL's transport operator
    solved on the actual mesh, so its crossing of `crit` is the equation's own answer to
    "when does this node clot".  Everything learned is the correction to it.
    """
    S = V[a]["S"]
    on = V[a]["t_adv"] + np.rint(np.asarray(resid, float))
    M = np.zeros((n_t, len(S["wall"])), dtype=bool)
    for j in range(n_t):
        M[j] = gm & ~S["wall"] & (on <= j)
    if commit_final:
        M[-1] = gm & ~S["wall"]
    return np.maximum.accumulate(M, axis=0)


def ode_wall_series(V, a, gm, n_t):
    """Wall commit series taken from the ODE's own crossing -- the physics anchor."""
    v, S = V[a], V[a]["S"]
    oi = np.searchsorted(np.asarray(v["times"]), v["oon_c"][ANCHOR_LEVEL[0]], side="left")
    M = np.zeros((n_t, len(S["wall"])), dtype=bool)
    for j in range(n_t):
        M[j] = gm & S["wall"] & (oi <= j)
    return M


def offwall_by_lag(M_wall, gm, owner, wall, lag, commit_final=True):
    """Off-wall nodes commit `lag` grid steps after their OWNER wall node does.

    `scripts/diag_offwall_structure.py` measures the actual structure and it is not subtle:
    pooled over 584 off-wall GT nodes, the lag behind the owner has median **+4 grid steps
    of 11**, p25 +3, p75 +6, and **88% lag by 2 or more**.  Only 8.4% commit at or before
    their owner -- which is why the owner-precedence constraint is nearly vacuous (it binds
    on 8% of nodes) while the LAG carries almost all the timing information.

    Physically this is the boundary layer filling: the wall node accumulates `Mat` from its
    own flux immediately, and the off-wall node has to wait for enough of it to be advected
    and to build past `crit` at ~0.16 of the owner's level (PHASE7 3.2), which takes most of
    the horizon.  Every previous off-wall timing arm tried to predict an absolute onset or a
    threshold crossing; none of them expressed "later than my owner, by about this much".
    """
    T, N = M_wall.shape
    won = np.full(N, T, dtype=int)
    for j in range(T - 1, -1, -1):
        won[M_wall[j]] = j
    on = np.clip(won[owner] + int(lag), 0, T)
    M = np.zeros((T, N), dtype=bool)
    for j in range(T):
        M[j] = gm & ~wall & (on <= j)
    if commit_final:
        M[-1] = gm & ~wall
    return np.maximum.accumulate(M, axis=0)


def series_masks(gm, P, th, commit_final=True, owner=None, wall=None):
    """Committed mask at each time, with the two constraints the production law implies.

    MONOTONE: `P` is already a cumulative maximum, so a node never un-clots -- `J0_Mat >= 0`
    and there is no sink (PHASE7 12.1 measured wall `Mat` to be the exact time-integral of
    its own nodal derivative, rank 0.999).

    COMMIT BY THE END: every node in the committed set is clot at the last timestep.  This
    is a coherence constraint, not a new model -- the set *is* the prediction of the final
    mask, so a node that is in the set but still below the time cut at `t_final` is the
    readout contradicting itself.  v3 had no such constraint and paid for it: its extra
    probability filter DELETES correct nodes at the last step, which is exactly why the
    time-conditioned arm reads FINAL off-wall 0.6514 against the frozen set's 0.7075 on
    the same set.  With this, the final mask equals the set by construction, so the
    temporal arm can no longer lose to frozen at `t_final`.

    Whether to enforce it is chosen per domain inside the fold, because the probability
    filter is not purely a cost: on the wall it also removes low-confidence set members and
    measured +0.014 FINAL there, while off-wall it deletes real clot and costs -0.064.

    OWNER PRECEDENCE: an off-wall node is fed by its nearest wall node -- PHASE7 3.1
    measured that an off-wall GT node's owner is itself GT-committed **99.9%** of the time,
    and PHASE7 1.1 says why (the wall flux is the only source, `Mat` is advected from it).
    So an off-wall node cannot be clot before its owner is.  `src/clot_ml/locked.py`'s
    shipped `enforce_owner_and_monotone` applies this and the strict evaluator did not.
    """
    M = gm[None, :] & (P >= th)
    if commit_final:
        M[-1] = gm
    M = np.maximum.accumulate(M, axis=0)
    if owner is not None and wall is not None:
        keep = np.zeros(M.shape[1], dtype=bool)
        for j in range(M.shape[0]):
            m = M[j] | keep
            m &= (wall | m[owner])
            M[j] = m
            keep = m
    return M


def score_vessel(V, a, P, oofs, set_th, time_th, prefix="", lag=None,
                 wall_resid=None, owner_cut=None):
    """-> (mean-over-time, final) per domain."""
    v, S = V[a], V[a]["S"]
    gm = candidate_mask(S, arm_scores(oofs, a), set_th, a)
    out = {}
    M_wall = (wall_by_residual(V, a, gm, wall_resid, len(v["times"]))
              if wall_resid is not None
              else series_masks(gm, P, time_th[0][0], time_th[0][1], S["owner"], S["wall"]))
    for key, dom in (("wall", S["wall"]), ("off", ~S["wall"])):
        th, cf = time_th[0] if key == "wall" else time_th[1]
        if key == "wall" and wall_resid is not None:
            M = M_wall
        elif key == "off" and lag is not None:
            Mw = M_wall & S["wall"]
            if isinstance(lag, tuple) and LAG_ANCHOR[0] == "adv":
                M = offwall_from_adv(V, a, gm, lag[1], len(v["times"]), cf)
            elif isinstance(lag, tuple):        # ("learned", per-node prediction)
                M = offwall_by_learned_lag(Mw, gm, S["owner"], S["wall"], lag[1], cf)
            else:
                M = offwall_by_lag(Mw, gm, S["owner"], S["wall"], lag, cf)
        else:
            M = series_masks(gm, P, th, cf, S["owner"], S["wall"])
        vals = []
        for j, ti in enumerate(v["times"]):
            vals.append(v["scorer"][ti].score(M[j] & dom, dom))
        vals = np.asarray(vals, dtype=float)
        with np.errstate(invalid="ignore"):
            out[prefix + key] = (float(np.nanmean(vals)) if np.any(~np.isnan(vals))
                                 else float("nan"))
        out[prefix + key + "_final"] = float(vals[-1])
    return out


def tune_set(cache, V, anchors, oofs):
    """Pick the readout family + scalars for the committed SET, against FINAL-time GT.

    Factorised from the time cut deliberately: the set answers *where* and is a property of
    the final mask, the time cut answers *when*.  v3 tuned one joint grid over a plain cut
    only, which could not express the physics-conditioned readout at all.
    """
    from eval_strict import FAMILIES, GRID

    vs = {a: SeverityScorer(cache[a]["edge_index"], cache[a]["y"] > 0.5,
                            len(cache[a]["wall"]), DEFAULT) for a in anchors}
    out = {}
    for key, dom_of in (("wall", lambda S: S["wall"]), ("off", lambda S: ~S["wall"])):
        best = None
        for arm in sorted(oofs):
            sub = {a: oofs[arm][a] for a in anchors}
            for fam, (tune, apply_) in FAMILIES.items():
                th = tune(cache, vs, anchors, sub, GRID)
                vals = []
                for a in anchors:
                    S = cache[a]
                    d = dom_of(S)
                    x = vs[a].score(apply_(S, sub[a], th) & d, d)
                    if x == x:
                        vals.append(x)
                q = float(np.mean(vals)) if vals else -1e9
                if best is None or q > best[0]:
                    best = (q, (arm, fam, th))
        out[key] = best[1]
    return out


def offwall_burden(V, a, oofs, set_th):
    """How many off-wall nodes this vessel's committed set holds -- label-free."""
    S = V[a]["S"]
    return int((candidate_mask(S, arm_scores(oofs, a), set_th, a) & ~S["wall"]).sum())


def _lag_quality(V, anchors, Pin, oofs, set_th, time_th, lag_pred, lag):
    """Mean off-wall score of a given (anchor level, lag rule) on the selection vessels."""
    vals = []
    for a in anchors:
        v, S = V[a], V[a]["S"]
        dom = ~S["wall"]
        gm = candidate_mask(S, arm_scores(oofs, a), set_th, a)
        th, cf = time_th[1]
        use = isinstance(lag, tuple) and a in lag_pred and             offwall_burden(V, a, oofs, set_th) >= lag[1]
        if use:
            Mw = ode_wall_series(V, a, gm, len(v["times"]))
            M = offwall_by_learned_lag(Mw, gm, S["owner"], S["wall"], lag_pred[a], cf)
        else:
            M = series_masks(gm, Pin[a], th, cf, S["owner"], S["wall"])
        for j, ti in enumerate(v["times"]):
            x = v["scorer"][ti].score(M[j] & dom, dom)
            if x == x:
                vals.append(x)
    return float(np.mean(vals)) if vals else -1e9


def tune_owner_cut(V, anchors, Pin, oofs, set_th, time_th, lag_pred):
    """The wall cut used ONLY to date the owner for the off-wall lag rule.

    The off-wall arm needs "when did my owner commit"; it has been reusing the wall cut that
    maximises the WALL score, which is a different objective.  A cut that is slightly early
    or late can be better for the wall's own mask and worse as a clock.  One scalar, chosen
    in-fold against the OFF-WALL score.
    """
    best = None
    for t_o in TIME_GRID:
        vals = []
        for a in anchors:
            v, S = V[a], V[a]["S"]
            dom = ~S["wall"]
            gm = candidate_mask(S, arm_scores(oofs, a), set_th, a)
            Mw = series_masks(gm, Pin[a], t_o, time_th[0][1], S["owner"], S["wall"]) & S["wall"]
            M = offwall_by_learned_lag(Mw, gm, S["owner"], S["wall"], lag_pred[a],
                                       time_th[1][1])
            for j, ti in enumerate(v["times"]):
                x = v["scorer"][ti].score(M[j] & dom, dom)
                if x == x:
                    vals.append(x)
        q = float(np.mean(vals)) if vals else -1e9
        if best is None or q > best[0]:
            best = (q, float(t_o))
    return best[1]


def tune_lag(V, anchors, Pin, oofs, set_th, time_th, lag_pred=None):
    """Cohort / learned / burden-gated lag, on inner OOF predictions.

    The learned per-node lag regression wins the in-fold selection every time and, held out,
    gains **+0.056 on the priority class** while losing 0.023 on the low-burden baseline
    vessels (docs/PHASE10_V4.md 12.2b).  That split is not mysterious: the stenoses carry 84
    and 122 off-wall GT nodes and `patient005` carries 4, so on a low-burden vessel the
    regression is extrapolating and a single mistimed node is most of the score.

    So the lag rule is GATED on the predicted off-wall burden, which needs no label -- it is
    the size of the committed set.  Below the gate the probability rule is used instead.
    """
    best = None
    opts = [None] + LAG_GRID
    if lag_pred:
        opts += [("learned", B) for B in BURDEN_GRID]
    for lag in opts:
        vals = []
        for a in anchors:
            v, S = V[a], V[a]["S"]
            dom = ~S["wall"]
            gm = candidate_mask(S, arm_scores(oofs, a), set_th, a)
            th, cf = time_th[1]
            use_learned = (isinstance(lag, tuple)
                           and offwall_burden(V, a, oofs, set_th) >= lag[1])
            if lag is None or (isinstance(lag, tuple) and not use_learned):
                M = series_masks(gm, Pin[a], th, cf, S["owner"], S["wall"])
            else:
                Mw = (ode_wall_series(V, a, gm, len(v["times"])) if LAG_ANCHOR[0] == "ode"
                      else series_masks(gm, Pin[a], time_th[0][0], time_th[0][1],
                                        S["owner"], S["wall"]) & S["wall"])
                if use_learned and LAG_ANCHOR[0] == "adv":
                    M = offwall_from_adv(V, a, gm, lag_pred[a], len(v["times"]), cf)
                elif use_learned:
                    M = offwall_by_learned_lag(Mw, gm, S["owner"], S["wall"],
                                               lag_pred[a], cf)
                else:
                    M = offwall_by_lag(Mw, gm, S["owner"], S["wall"], lag, cf)
            for j, ti in enumerate(v["times"]):
                x = v["scorer"][ti].score(M[j] & dom, dom)
                if x == x:
                    vals.append(x)
        q = float(np.mean(vals)) if vals else -1e9
        if best is None or q > best[0]:
            best = (q, lag)
    return best[1]


def tune_time(V, anchors, Pin, oofs, set_th):
    """Per-domain (time cut, commit-final flag), given the set, on inner OOF predictions.

    Selected against mean-over-time, which is the metric the temporal arm exists to move;
    the final-time score is then whatever that choice implies, and is reported separately
    rather than tuned for.
    """
    out = []
    for di in (0, 1):
        top, pick = -1e9, (0.5, True)
        for t_th in TIME_GRID:
            for cf in (True, False):
                vals = []
                for a in anchors:
                    v, S = V[a], V[a]["S"]
                    dom = S["wall"] if di == 0 else ~S["wall"]
                    M = series_masks(candidate_mask(S, arm_scores(oofs, a), set_th, a),
                                     Pin[a], t_th, cf, S["owner"], S["wall"])
                    for j, ti in enumerate(v["times"]):
                        x = v["scorer"][ti].score(M[j] & dom, dom)
                        if x == x:
                            vals.append(x)
                if vals and np.mean(vals) > top:
                    top, pick = float(np.mean(vals)), (float(t_th), bool(cf))
        out.append(pick)
    return tuple(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True,
                    help="each arm is a comma-separated tag list")
    ap.add_argument("--cache", default="gt")
    ap.add_argument("--n-times", type=int, default=11)
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--save", default="")
    ap.add_argument("--wall-resid", action="store_true",
                    help="wall onset = the ODE's grid onset + a learned residual")
    ap.add_argument("--lag-seeds", type=int, default=3,
                    help="seed-average the lag regression")
    ap.add_argument("--lag-anchor", default="pred", choices=["pred", "ode", "adv"],
                    help="date the owner by the head's prediction, or by the ODE crossing")
    ap.add_argument("--owner-cut", action="store_true",
                    help="tune a separate wall cut used only to date the owner off-wall")
    ap.add_argument("--oracle-lag", action="store_true",
                    help="ORACLE per-node lag with the PREDICTED wall onset (a probe)")
    ap.add_argument("--learn-lag", action="store_true",
                    help="offer a PER-NODE learned lag alongside the cohort constants")
    ap.add_argument("--owner-lag", action="store_true",
                    help="off-wall onset = owner's predicted onset + a fitted lag")
    ap.add_argument("--two-stage", action="store_true",
                    help="give off-wall nodes their owner's PREDICTED commit trajectory")
    ap.add_argument("--set-masks", default="",
                    help="npz of committed masks per vessel; overrides the set readout")
    ap.add_argument("--head-seeds", type=int, default=1,
                    help="average this many gradient-boosted heads (variance reduction)")
    ap.add_argument("--clock", action="store_true",
                    help=("add the per-vessel physics clock (fraction of the vessel's own "
                          "ODE/advection nodes fired by t).  MEASURED NEGATIVE and off by "
                          "default: mean-over-time wall +0.005 but off-wall -0.038, because "
                          "a vessel-level scalar lets the head fit the schedule of 14 "
                          "training vessels rather than learn a transferable one."))
    ap.add_argument("--no-tt", action="store_true",
                    help="ablate the time-resolved transport channels")
    args = ap.parse_args()

    from eval_strict import load_scores

    cache = attach_physics(load_cache(args.cache))
    oofs, pool, folds = {}, None, None
    for arm in args.arms:
        p_, f_, sc_ = load_scores(arm.split(","))
        if pool is None:
            pool, folds = [a for a in p_ if a in cache], f_
        fo = {a: k for k, held in f_.items() for a in held}
        oofs[arm] = {a: sc_[(fo[a], a)] for a in pool}
    classes = classes_for(pool, PACKS)

    if args.set_masks:
        z = np.load(REPO / args.set_masks)
        EXTERNAL_SET.update({a: z[a].astype(bool) for a in z.files})
        print("[i] committed set taken from %s (%d vessels)"
              % (args.set_masks, len(EXTERNAL_SET)), flush=True)
    LAG_ANCHOR[0] = args.lag_anchor
    print("[i] precomputing %d vessels ..." % len(pool), flush=True)
    V = precompute(pool, cache, args.n_times)
    if args.no_tt:
        for v in V.values():
            v["tt"] = None
    if not args.clock:
        for v in V.values():
            v["clock"] = []

    rows, t0 = {}, time.time()
    for k, held in sorted(folds.items()):
        sel = [a for a in pool if a not in held]
        # --- inner CV over the selection set, to get honest predictions for tuning ------
        set_th = tune_set(cache, V, sel, oofs)
        inner = [sel[i::args.inner] for i in range(args.inner)]

        def inner_oof():
            """Stage-appropriate out-of-fold predictions for every selection vessel.

            Inner folds use the SAME seed count as the final head.  Dropping them to one
            seed was tried and costs mean-over-time off-wall 0.6833 -> 0.6706: the inner
            predictions are what the time thresholds are tuned on, so their variance lands
            straight in the chosen cut.
            """
            out = {}
            for iv in inner:
                itr = [a for a in sel if a not in iv]
                m_i = fit_head(V, itr, oofs, set_th, args.head_seeds)
                for a in iv:
                    out[a] = predict_series(V, a, m_i, oofs)
            return out

        OWNER_PRED.clear()
        Pin = inner_oof()                                   # STAGE 1, out-of-fold on `sel`
        m_k = fit_head(V, sel, oofs, set_th, args.head_seeds)

        if args.two_stage:
            # Feed each node its OWNER wall node's stage-1 predicted trajectory.  The
            # selection vessels get their INNER out-of-fold stage-1 prediction and the
            # held-out vessels get the stage-1 head fitted on all of `sel`, so the feature
            # is out-of-sample on both sides -- the same discipline the `score` feature
            # already gets, and the reason v3's head was quietly flattered without it.
            for a in sel:
                OWNER_PRED[a] = Pin[a][:, V[a]["S"]["owner"]]
            for a in held:
                OWNER_PRED[a] = predict_series(V, a, m_k, oofs)[:, V[a]["S"]["owner"]]
            Pin = inner_oof()                               # STAGE 2, with the new feature
            m_k = fit_head(V, sel, oofs, set_th, args.head_seeds)

        time_th = tune_time(V, sel, Pin, oofs, set_th)
        owner_cut = None
        wres, wm, wres_pred = None, None, {}
        if args.wall_resid:
            for iv in inner:
                itr = [a for a in sel if a not in iv]
                m_w = fit_wall_residual(V, itr, oofs, args.lag_seeds)
                if m_w is None:
                    continue
                for a in iv:
                    wres_pred[a] = m_w.predict(lag_features(V, a, oofs))
            wm = fit_wall_residual(V, sel, oofs, args.lag_seeds)
            # Choose against the probability rule on the inner out-of-fold predictions, and
            # GATE on wall burden -- the same construction that made the off-wall lag work
            # (12.2b).  A residual fitted across the cohort is only trustworthy on vessels
            # with enough committed wall nodes to have contributed to it.
            n_t = len(V[sel[0]]["times"])

            def wburden(a):
                S = V[a]["S"]
                return int((candidate_mask(S, arm_scores(oofs, a), set_th, a)
                            & S["wall"]).sum())

            best = None
            for B in WBURDEN_GRID:
                vals = []
                for a in sel:
                    v, S = V[a], V[a]["S"]
                    gm = candidate_mask(S, arm_scores(oofs, a), set_th, a)
                    th, cf = time_th[0]
                    use = (B is not None) and (a in wres_pred) and (wburden(a) >= B)
                    M = (wall_by_residual(V, a, gm, wres_pred[a], n_t) if use
                         else series_masks(gm, Pin[a], th, cf, S["owner"], S["wall"]))
                    for j, ti in enumerate(v["times"]):
                        x = v["scorer"][ti].score(M[j] & S["wall"], S["wall"])
                        if x == x:
                            vals.append(x)
                q = float(np.mean(vals)) if vals else -1e9
                if best is None or q > best[0]:
                    best = (q, B)
            wres = best[1]
        # ORACLE-LAG probe: the true per-node lag with our OWN predicted wall onset.  This
        # splits the remaining timing gap into "we mis-predict the lag" and "we mis-predict
        # when the owner commits", which need opposite work.
        if args.oracle_lag:
            for a in pool:
                v_, S_ = V[a], V[a]["S"]
                g_ = np.asarray(v_["times"])
                gi = np.searchsorted(g_, v_["go"], side="left")
                lag_pred_o = (gi - gi[S_["owner"]]).astype(float)
                OWNER_PRED.setdefault("__orc__", {})[a] = lag_pred_o
        lag, lm, lag_pred = None, None, {}
        if args.oracle_lag:
            lag = ("learned", 0)
            lag_pred = {a: OWNER_PRED["__orc__"][a] for a in pool}
            lm = None
        elif args.owner_lag:
            if args.learn_lag:
                # OUT-OF-FOLD lag predictions for the selection vessels, on the same inner
                # split the head uses.  Fitting the lag model on `sel` and then judging it
                # on `sel` is the leak this whole evaluator exists to remove: the regression
                # would be reading its own training labels, and the in-fold tuner would then
                # always prefer it -- which is exactly what it did before this was fixed.
                for iv in inner:
                    itr = [a for a in sel if a not in iv]
                    m_l = fit_lag_model(V, itr, oofs, args.lag_seeds, args.lag_anchor)
                    if m_l is None:
                        continue
                    for a in iv:
                        lag_pred[a] = m_l.predict(lag_features(V, a, oofs))
                lm = fit_lag_model(V, sel, oofs, args.lag_seeds, args.lag_anchor)
            if args.lag_anchor == "ode" and len(ANCHOR_C) > 1:
                bestc = None
                for c in ANCHOR_C:
                    ANCHOR_LEVEL[0] = c
                    lp = {}
                    for iv in inner:
                        m_c = fit_lag_model(V, [a for a in sel if a not in iv], oofs,
                                            args.lag_seeds, "ode")
                        if m_c is None:
                            continue
                        for a in iv:
                            lp[a] = m_c.predict(lag_features(V, a, oofs))
                    lg = tune_lag(V, sel, Pin, oofs, set_th, time_th, lp)
                    q = _lag_quality(V, sel, Pin, oofs, set_th, time_th, lp, lg)
                    if bestc is None or q > bestc[0]:
                        bestc = (q, c, lp, lg)
                ANCHOR_LEVEL[0] = bestc[1]
                lag_pred, lag = bestc[2], bestc[3]
                lm = fit_lag_model(V, sel, oofs, args.lag_seeds, "ode")
            else:
                lag = tune_lag(V, sel, Pin, oofs, set_th, time_th, lag_pred)
            if args.owner_cut and isinstance(lag, tuple) and lag_pred:
                owner_cut = tune_owner_cut(V, sel, Pin, oofs, set_th, time_th, lag_pred)
        for a in held:
            P = predict_series(V, a, m_k, oofs)
            lag_a = lag
            if isinstance(lag, tuple):
                pl = (lag_pred[a] if lm is None else lm.predict(lag_features(V, a, oofs)))
                lag_a = (("learned", pl)
                         if offwall_burden(V, a, oofs, set_th) >= lag[1] else None)
            wr = None
            if wres is not None and wm is not None:
                S_ = V[a]["S"]
                nb = int((candidate_mask(S_, arm_scores(oofs, a), set_th, a)
                          & S_["wall"]).sum())
                if nb >= wres:
                    wr = wm.predict(lag_features(V, a, oofs))
            r = score_vessel(V, a, P, oofs, set_th, time_th, lag=lag_a, wall_resid=wr,
                             owner_cut=owner_cut)
            # the STATIC readout on the same set, as the reference the temporal arm must
            # beat: frozen mask, replayed at every timestep
            r.update(score_vessel(V, a, np.ones_like(P), oofs, set_th,
                                  ((0.0, True), (0.0, True)), prefix="frozen_"))
            # ORACLE TIMING on our OWN committed set: perfect onset, same mask.  This is
            # the ceiling the temporal arm is actually chasing -- not the global oracle,
            # which also has a perfect set.
            Po = np.stack([(V[a]["go"] <= ti).astype(np.float32) for ti in V[a]["times"]])
            r.update(score_vessel(V, a, Po, oofs, set_th,
                                  ((0.5, True), (0.5, True)), prefix="oracle_"))
            r["cls"] = classes.get(a, "?")
            rows[a] = r
        desc = " ".join("%s:%s/%s" % (d, set_th[d][0][:10], set_th[d][1])
                        for d in ("wall", "off"))
        tdesc = (" ".join("%.2f%s" % (t, "C" if c else "-") for t, c in time_th)
                 + ("" if lag is None else " lag=%s" % (lag,))
                 + ("" if wres is None else " wres>=%d" % wres)
                 + ("" if owner_cut is None else " ocut=%.2f" % owner_cut))
        print("  fold %d %s time=%s  %s  (%.0fs)"
              % (k, desc, tdesc,
                 " ".join("%s m%.3f f%.3f" % (a[-3:], rows[a]["wall"], rows[a]["wall_final"])
                          for a in held), time.time() - t0), flush=True)

    groups = [("ALL", pool),
              ("baseline", [a for a in pool if not is_priority(classes.get(a, ""))]),
              ("PRIORITY", [a for a in pool if is_priority(classes.get(a, ""))])]
    print("\nSTRICTLY NESTED (tags=%s, cache=%s)\n" % (",".join(args.arms), args.cache))
    print("%-10s %-8s %3s | %9s %9s | %9s %9s"
          % ("group", "arm", "n", "mean wall", "mean off", "FIN wall", "FIN off"))
    for name, sub in groups:
        if not sub:
            continue
        for arm, pre in (("frozen", "frozen_"), ("temporal", ""), ("oracleT", "oracle_")):
            print("%-10s %-8s %3d | %9.4f %9.4f | %9.4f %9.4f"
                  % (name, arm, len(sub),
                     np.nanmean([rows[a][pre + "wall"] for a in sub]),
                     np.nanmean([rows[a][pre + "off"] for a in sub]),
                     np.nanmean([rows[a][pre + "wall_final"] for a in sub]),
                     np.nanmean([rows[a][pre + "off_final"] for a in sub])))
    print("\nper vessel (mean-over-time / final)")
    for a in sorted(rows):
        r = rows[a]
        print("  %-11s %-9s wall %.3f/%.3f   off %s/%s"
              % (a, r["cls"][:9], r["wall"], r["wall_final"],
                 ("%.3f" % r["off"]) if r["off"] == r["off"] else "  n/a",
                 ("%.3f" % r["off_final"]) if r["off_final"] == r["off_final"] else "  n/a"))
    if args.save:
        Path(args.save).write_text(json.dumps(rows, indent=2, default=float))
        print("\nwrote %s" % args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
