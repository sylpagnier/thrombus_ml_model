"""Per-vessel readout calibration -- turning a score field into a mask without labels.

THE MEASUREMENT THAT MOTIVATES THIS.  `scripts/diag_readout_ceiling.py`, strictly nested,
same score field, final time point:

    cohort-wide absolute cut     wall 0.9024    off 0.7075
    per-vessel ORACLE cut        wall 0.9447    off 0.8275

**+0.042 wall and +0.120 off-wall sit in the threshold, not in the model.**  Individually:
`patient035` wall 0.656 -> 0.974, `patient028` 0.699 -> 0.854, `patient019` 0.851 -> 0.971,
`patient020` off 0.509 -> 0.829, `patient005` off 0.240 -> 0.621, `patient032` off
0.432 -> 0.742.  The network already separates those vessels; one cohort constant does not
sit in the right place on any of them.

`docs/PHASE9_ML.md` 4 records two failed attempts at per-vessel adaptivity -- a budget from
the physics mask size, and one from the model's own confidence mass.  Both predicted a
**count** (how many nodes to commit).  None of them measured the ceiling first, and none
tried to make the *cut* scale-free instead of predicting a budget.  The rules here are the
latter kind: each replaces the absolute constant with a quantity computed from this
vessel's own score distribution, so a vessel whose scores are uniformly shifted gets a
correspondingly shifted cut for free.

Every rule takes cohort-wide parameters, which are still fitted inside the fold; what
changes is what those parameters *mean*.  ``absolute`` reproduces the shipped readout
exactly, so it is the control.
"""
from __future__ import annotations

import numpy as np

__all__ = ["RULES", "apply_rule", "rule_grid"]


def _dom(score, domain):
    return np.asarray(score, dtype=np.float64)[np.asarray(domain, dtype=bool)]


def absolute(score, domain, phys, p):
    """``score >= t`` -- the shipped readout, and the control for every other rule."""
    return score >= p[0]


def quantile(score, domain, phys, p):
    """Commit the top ``1 - q`` fraction of THIS domain's nodes.

    Scale-free in the score: a vessel whose whole field is shifted up or down keeps the
    same committed fraction.  This is a budget rule, and `PHASE9_ML` 4 killed budget rules
    derived from the *physics mask size* -- but a fixed fraction of the mesh is a different
    quantity, and the mesh size does track vessel size where the physics mask does not.
    """
    v = _dom(score, domain)
    if v.size == 0:
        return np.zeros_like(score, dtype=bool)
    return score >= np.quantile(v, np.clip(p[0], 0.0, 1.0))


def rel_max(score, domain, phys, p):
    """``score >= t * max(score)`` within the domain -- relative to the field's own top."""
    v = _dom(score, domain)
    if v.size == 0:
        return np.zeros_like(score, dtype=bool)
    return score >= p[0] * float(v.max())


def phys_anchored(score, domain, phys, p):
    """Put the cut at the score quantile the PHYSICS mask's own size implies, scaled.

    The physics backbone commits `n_phys` nodes in this domain with zero free parameters.
    That count is a bad *answer* off-wall (`PHASE9_ML` 4) but it is a serviceable *unit*:
    committing `p[0] * n_phys` nodes adapts to the vessel while letting the cohort fit the
    one ratio that the backbone systematically gets wrong.
    """
    d = np.asarray(domain, dtype=bool)
    v = _dom(score, domain)
    n_phys = int((np.asarray(phys, dtype=bool) & d).sum())
    k = int(round(p[0] * max(n_phys, 1)))
    k = int(np.clip(k, 1, v.size))
    if v.size == 0:
        return np.zeros_like(score, dtype=bool)
    return score >= np.sort(v)[::-1][k - 1]


def gap(score, domain, phys, p):
    """Cut at the widest gap in the top of the sorted score -- an Otsu-like separator.

    If the field genuinely separates two populations there is an empty band between them,
    and its location is a property of this vessel alone.  ``p[0]`` bounds the search to the
    top fraction so the (very large) bulk-vs-everything gap is not selected.
    """
    d = np.asarray(domain, dtype=bool)
    v = np.sort(_dom(score, domain))[::-1]
    if v.size < 4:
        return np.zeros_like(score, dtype=bool)
    m = max(2, min(v.size - 1, int(p[0] * v.size)))
    seg = v[:m + 1]
    i = int(np.argmax(seg[:-1] - seg[1:]))
    return score >= float(seg[i])


RULES = {"absolute": absolute, "quantile": quantile, "rel_max": rel_max,
         "phys_anchored": phys_anchored, "gap": gap}

#: parameter grid per rule, searched inside the fold
_GRIDS = {
    "absolute": np.round(np.linspace(0.02, 0.98, 33), 4),
    "quantile": np.concatenate([1.0 - np.geomspace(0.30, 2e-4, 28)]),
    "rel_max": np.round(np.linspace(0.05, 0.99, 25), 4),
    "phys_anchored": np.round(np.geomspace(0.15, 6.0, 22), 4),
    "gap": np.round(np.geomspace(2e-4, 0.20, 18), 5),
}


def rule_grid(name: str) -> np.ndarray:
    return _GRIDS[name]


def apply_rule(name: str, score, domain, phys, p) -> np.ndarray:
    """Committed mask, always intersected with the domain it was calibrated on."""
    d = np.asarray(domain, dtype=bool)
    return RULES[name](np.asarray(score, dtype=np.float64), d, phys, np.atleast_1d(p)) & d
