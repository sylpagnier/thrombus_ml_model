"""Turning a per-node score into a mask.  This is where per-vessel calibration lives.

PHASE7 7.2 measured that **calibration alone is 53% of the score gap** -- remapping the
model's Mat onto the GT distribution, with rank order untouched, was worth more than every
mechanism in that phase.  A single global probability cut cannot express that: clot burden
runs from 11 to 313 nodes across this cohort, and the score is F0.5-weighted, so the same
cut is far too loose on a light vessel and far too tight on a heavy one.

Three readouts, in increasing amounts of per-vessel adaptation:

  ``thresh``   one cut per domain.  No adaptation.
  ``resid``    keep / add cuts per domain against the physics mask.  Wall error is two
               opposite failure modes (weak-sep FP, ungated FN) so one cut cannot fix both.
  ``topk``     per-vessel budget: take the top ``a * |physics wall mask|`` nodes in each
               domain.  The physics mask size is a deploy-legal burden estimate, so this
               calibrates each vessel to its own scale without seeing the label.
"""
from __future__ import annotations

import itertools

import numpy as np


def _topk_mask(score: np.ndarray, domain: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros_like(domain)
    if k <= 0:
        return out
    idx = np.flatnonzero(domain)
    if len(idx) == 0:
        return out
    k = min(int(k), len(idx))
    sel = idx[np.argpartition(-score[idx], k - 1)[:k]]
    out[sel] = True
    return out


def apply_thresh(S, score, p):
    w = S["wall"]
    return ((score >= p[0]) & w) | ((score >= p[1]) & ~w)


def apply_resid(S, score, p):
    w, ph = S["wall"], S["phys_mask"]
    kw, aw, ko, ao = p
    return (((w & ph & (score >= kw)) | (w & ~ph & (score >= aw)))
            | ((~w & ph & (score >= ko)) | (~w & ~ph & (score >= ao))))


def apply_topk(S, score, p):
    """``p = (a_wall, a_off)`` as multiples of the physics WALL mask size."""
    w, ph = S["wall"], S["phys_mask"]
    base = max(int((ph & w).sum()), 1)
    return (_topk_mask(score, w, round(p[0] * base))
            | _topk_mask(score, ~w, round(p[1] * base)))


def apply_topk_resid(S, score, p):
    """Physics-anchored budget: the physics mask, plus/minus a budgeted edit.

    ``p = (a_wall, a_off, drop_frac)``.  Nodes are ranked by score; the wall budget is
    ``a_wall * |physics wall mask|`` and the weakest ``drop_frac`` of physics-positive wall
    nodes are removed.  Never strays far from a mask that already scores 0.86.
    """
    w, ph = S["wall"], S["phys_mask"]
    base = max(int((ph & w).sum()), 1)
    keep = ph & w
    if p[2] > 0 and keep.sum() > 3:
        idx = np.flatnonzero(keep)
        n_drop = int(round(p[2] * len(idx)))
        if n_drop > 0:
            worst = idx[np.argsort(score[idx])[:n_drop]]
            keep = keep.copy()
            keep[worst] = False
    add = _topk_mask(np.where(w & ~ph, score, -np.inf), w & ~ph,
                     max(round(p[0] * base) - int(keep.sum()), 0))
    off = _topk_mask(score, ~w, round(p[1] * base))
    return keep | add | off


def apply_expected(S, score, p):
    """Per-vessel budget from the model's OWN confidence mass: ``k = a * sum(p)``.

    The physics-mask size is a poor burden proxy off-wall (it barely tracks the GT count),
    but ``sum(p)`` over a domain IS the model's expected number of positives, so this is
    self-calibrating and needs no external estimate.  It matters because the off-wall score
    is dominated by vessels with 4-14 GT nodes, where F0.5 collapses on a single false
    positive, while the vessels with 90-122 nodes need recall -- one global cut cannot serve
    both and an expected-count budget can.
    """
    w = S["wall"]
    kw = int(round(p[0] * float(score[w].sum())))
    ko = int(round(p[1] * float(score[~w].sum())))
    return _topk_mask(score, w, kw) | _topk_mask(score, ~w, ko)


def apply_expected_resid(S, score, p):
    """Expected-count budget, but the wall starts from the physics mask.

    ``p = (a_wall, a_off, drop_frac)``.
    """
    w, ph = S["wall"], S["phys_mask"]
    keep = ph & w
    if p[2] > 0 and keep.sum() > 3:
        idx = np.flatnonzero(keep)
        nd = int(round(p[2] * len(idx)))
        if nd > 0:
            keep = keep.copy()
            keep[idx[np.argsort(score[idx])[:nd]]] = False
    kw = int(round(p[0] * float(score[w].sum())))
    add = _topk_mask(np.where(w & ~ph, score, -np.inf), w & ~ph, max(kw - int(keep.sum()), 0))
    ko = int(round(p[1] * float(score[~w].sum())))
    return keep | add | _topk_mask(score, ~w, ko)


def apply_thresh_shell(S, score, p):
    """One cut per domain, but off-wall predictions are confined to the species shell.

    99.9% of off-wall GT clot sits on the first corner row (PHASE7 8.4), and the shell is a
    purely topological object -- no length constant, so it transfers across meshes.  Nodes
    outside it are structurally unable to carry Mat, so predicting there can only cost
    precision, and the score is F0.5-weighted.
    """
    w, sh = S["wall"], S["shell"].astype(bool)
    return ((score >= p[0]) & w) | ((score >= p[1]) & ~w & sh)


def apply_expected_shell(S, score, p):
    w, sh = S["wall"], S["shell"].astype(bool)
    off = ~w & sh
    kw = int(round(p[0] * float(score[w].sum())))
    ko = int(round(p[1] * float(score[off].sum())))
    return _topk_mask(score, w, kw) | _topk_mask(score, off, ko)


def apply_blend(S, score, p):
    """Blend the physics mask into the score as a prior, then one cut per domain.

    ``p = (lam, t_wall, t_off)``.  The backbone scores 0.86 wall on its own and beats the
    network on individual vessels (p035 0.94 vs 0.71), so where the network is unsure the
    prior should win.  This is the cheapest form of "do not lose to the thing you started
    from" -- lam=0 recovers the pure network, lam=1 the pure physics mask.
    """
    lam, tw, to = p
    sc = (1.0 - lam) * score + lam * S["phys_mask"].astype(np.float32)
    w = S["wall"]
    return ((sc >= tw) & w) | ((sc >= to) & ~w)


REGISTRY = {
    "thresh": (apply_thresh, [np.linspace(0.02, 0.998, 60)] * 2),
    "resid": (apply_resid, [np.linspace(0.02, 0.995, 14)] * 4),
    "topk": (apply_topk, [np.round(np.arange(0.4, 2.61, 0.1), 3),
                          np.round(np.arange(0.0, 1.81, 0.1), 3)]),
    "topk_resid": (apply_topk_resid, [np.round(np.arange(0.6, 2.41, 0.15), 3),
                                      np.round(np.arange(0.0, 1.61, 0.15), 3),
                                      np.array([0.0, 0.05, 0.1, 0.2, 0.3])]),
    "expected": (apply_expected, [np.round(np.arange(0.2, 2.21, 0.1), 3),
                                  np.round(np.arange(0.1, 2.21, 0.1), 3)]),
    "blend": (apply_blend, [np.round(np.arange(0.0, 0.85, 0.1), 3),
                            np.linspace(0.05, 0.98, 16), np.linspace(0.05, 0.98, 16)]),
    "thresh_shell": (apply_thresh_shell, [np.linspace(0.02, 0.995, 24)] * 2),
    "expected_shell": (apply_expected_shell, [np.round(np.arange(0.2, 2.21, 0.1), 3),
                                              np.round(np.arange(0.1, 2.61, 0.1), 3)]),
    "expected_resid": (apply_expected_resid, [np.round(np.arange(0.3, 2.01, 0.15), 3),
                                              np.round(np.arange(0.1, 2.01, 0.15), 3),
                                              np.array([0.0, 0.1, 0.2, 0.3])]),
}


def tune(name, bench, scores, anchors):
    """Grid-search the readout parameters on ``anchors``; domains scored independently
    where the parameterisation allows it, jointly otherwise."""
    fn, grids = REGISTRY[name]
    best, best_p = -1e9, None
    for p in itertools.product(*grids):
        vw, vo = [], []
        for a in anchors:
            S = bench.cache[a]
            pr = fn(S, scores[a], p)
            w = bench.vs[a].score(pr, S["wall"])
            o = bench.vs[a].score(pr, ~S["wall"])
            if w == w:
                vw.append(w)
            if o == o:
                vo.append(o)
        obj = (np.mean(vw) if vw else 0.0) + (np.mean(vo) if vo else 0.0)
        if obj > best:
            best, best_p = obj, p
    return best_p, best


# Readouts whose parameters split cleanly by domain: index 0 controls the wall decision
# and index 1 the off-wall one, with no interaction.  The metric of record is computed
# separately per domain, so tuning them JOINTLY (on the sum) needlessly couples them --
# a better off-wall model then drags the wall threshold off its own optimum.
SEPARABLE = {"thresh": (0, 1), "expected": (0, 1),
             "thresh_shell": (0, 1), "expected_shell": (0, 1)}


def tune_separable(name, bench, scores, anchors):
    """Tune the wall parameter on the wall score and the off parameter on the off score."""
    fn, grids = REGISTRY[name]
    iw, io_ = SEPARABLE[name]
    base = [g[0] for g in grids]

    def best(i, domain_of, invert):
        bv, bp = -1e9, grids[i][0]
        for val in grids[i]:
            p = list(base)
            p[i] = val
            vals = []
            for a in anchors:
                S = bench.cache[a]
                d = (~S["wall"]) if invert else S["wall"]
                v = bench.vs[a].score(fn(S, scores[a], tuple(p)), d)
                if v == v:
                    vals.append(v)
            if vals and np.mean(vals) > bv:
                bv, bp = float(np.mean(vals)), val
        return bp, bv

    pw, vw = best(iw, None, False)
    base[iw] = pw
    po, vo = best(io_, None, True)
    base[io_] = po
    return tuple(base), vw + vo


def apply(name, S, score, p):
    return REGISTRY[name][0](S, score, p)
