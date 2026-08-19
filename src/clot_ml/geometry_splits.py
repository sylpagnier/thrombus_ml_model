"""Geometry-stratified protocol for the wall cohort.

WHY THE OLD CUT HAD TO GO.  `src/core_physics/wall_cohort_splits.py` puts 040/041/044 in DEV
and everything else in FIT.  Measured (`geometry_class.py`), that is **exactly** the
stenosis/aneurysm set against an all-baseline FIT, so every FIT-vs-DEV number in
`docs/PHASE9_ML.md` is confounded with geometry class: the model reads DEV off-wall 0.80 and
FIT 0.64, and that is a comparison of three pathological vessels against ten normal ones,
not evidence of generalisation.

WHY A FIXED RE-CUT CANNOT FIX IT.  Of the 19 eligible vessels outside SEALED
(T >= 150, non-empty GT) exactly **three** are priority class:

    patient040  aneurysm      patient041  stenosis      patient044  stenosis

`patient039` is also an aneurysm but T = 92, and a truncated run is a different quantity
(PHASE6_RESULTS 6.2), so it is excluded everywhere.  `patient042` (stenosis) and
`patient043` (aneurysm) are in SEALED and stay there.

**With one non-SEALED aneurysm, no fixed FIT/DEV cut can put an aneurysm on both sides.**
Putting 040 in FIT means aneurysm generalisation is never measured; putting it in DEV means
the model never trains on an aneurysm.  That is a property of the data, not of the split.

WHAT THIS MODULE DOES INSTEAD.  Geometry-stratified K-fold over the whole eligible
non-SEALED pool.  Every vessel is held out exactly once, so:

  * every vessel has an honest out-of-fold score, including all three priority vessels;
  * 040 is *trained on* in K-1 folds and *measured* in one -- both, rather than neither;
  * priority vessels land in different folds by construction, so each fold's training set
    contains at least two of them.

The one thing it still cannot do is train on an aneurysm while measuring a different
aneurysm.  That needs either `patient039` re-run to full horizon, or SEALED opened.  Until
then **aneurysm performance is an n=1 out-of-fold number and must be quoted as such.**
"""
from __future__ import annotations

from collections import defaultdict

import torch

from src.core_physics.wall_cohort_splits import DEV as OLD_DEV, FIT as OLD_FIT, MIN_T, SEALED

PRIORITY_CLASSES = ("aneurysm", "stenosis", "stenosis+aneurysm")


def eligible_pool() -> list[str]:
    """Non-SEALED vessels with a full horizon and non-empty GT, in a stable order."""
    return sorted(set(OLD_FIT) | set(OLD_DEV))


def classes_for(anchors, pack_dir) -> dict[str, str]:
    """anchor -> geometry class, using the measured classifier with its documented abstain."""
    from src.clot_ml.geometry_class import USER_DESIGNATED, classify, width_stats

    out = {}
    for a in anchors:
        p = pack_dir / f"{a}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        if int(d.y.shape[0]) < MIN_T:
            continue
        s = width_stats(d)
        cls = classify(s, a)
        if cls == "unknown":
            # width_nd is unusable here; fall back to the human designation, and treat an
            # unlabelled vessel as baseline for STRATIFICATION only (never for reporting).
            cls = USER_DESIGNATED.get(a, "unknown")
        out[a] = cls
    return out


def is_priority(cls: str) -> bool:
    return cls in PRIORITY_CLASSES


def stratified_folds(classes: dict[str, str], k: int = 5) -> list[list[str]]:
    """K held-out sets, dealing each geometry class round-robin so priority vessels spread.

    Deterministic: vessels are dealt in sorted order within class, and the classes are
    processed rarest-first so the scarce priority vessels choose their folds before the
    plentiful baseline ones fill the space.
    """
    by_cls: dict[str, list[str]] = defaultdict(list)
    for a, c in classes.items():
        by_cls[c].append(a)
    folds: list[list[str]] = [[] for _ in range(k)]
    order = sorted(by_cls, key=lambda c: (len(by_cls[c]), c))
    slot = 0
    for c in order:
        for a in sorted(by_cls[c]):
            folds[slot % k].append(a)
            slot += 1
    return [sorted(f) for f in folds]


def describe(classes: dict[str, str], folds: list[list[str]]) -> str:
    lines = []
    for i, f in enumerate(folds):
        tag = ", ".join("%s[%s]" % (a, classes.get(a, "?")[:4]) for a in f)
        n_prio = sum(is_priority(classes.get(a, "")) for a in f)
        lines.append("  fold %d (n=%d, priority=%d): %s" % (i, len(f), n_prio, tag))
    return "\n".join(lines)
