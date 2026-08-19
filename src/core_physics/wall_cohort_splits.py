"""FIT / DEV / SEALED splits for the wall-cohort physics model.

This is the protocol ``scripts/sweep_ml_clean_protocol.py`` and
``docs/WALL_MODEL_PLAN.md`` §21.1 already use.  Phase-7/8 evals that average
``WALL_COHORT_V2_TRAIN`` (27, or the 19 eligible full-horizon clot-carrying subset)
are mixing FIT with DEV, and ``patient020`` is a FIT vessel -- not a holdout.

    FIT     TRAIN minus DEV-train (039/040/041/044)
    DEV     039, 040, 041, 044 -- selection only, never fitted
    SEALED  WALL_COHORT_V2_GENERALIZATION -- never tune, spend once

Truncated (T<150) and empty-GT vessels are a different *quantity*, not a different
split: drop them everywhere (PHASE6_RESULTS 6.2).  The wall-gen small cohort in
AGENTS.md (train 005/006/010/023/002, val=020) is a different stack; do not mix it
into these numbers.
"""
from __future__ import annotations

from collections import defaultdict

from src.biochem_gnn.mat_growth_simple import (
    WALL_COHORT_V2_DEV,
    WALL_COHORT_V2_DEV_HOLDOUT,
    WALL_COHORT_V2_DEV_TRAIN,
    WALL_COHORT_V2_FIT,
    WALL_COHORT_V2_GENERALIZATION,
    WALL_COHORT_V2_TRAIN,
)

MIN_T = 150

FIT = WALL_COHORT_V2_FIT
DEV = WALL_COHORT_V2_DEV_TRAIN
SEALED = WALL_COHORT_V2_GENERALIZATION


def split_of(anchor: str) -> str:
    if anchor in SEALED:
        return "sealed"
    if anchor in DEV:
        return "dev"
    if anchor in WALL_COHORT_V2_TRAIN:
        return "fit"
    return "other"


def assert_disjoint() -> None:
    fit, dev, sealed = set(FIT), set(DEV), set(SEALED)
    assert not (fit & dev), fit & dev
    assert not (fit & sealed), fit & sealed
    assert not (dev & sealed), dev & sealed
    assert set(DEV) == set(WALL_COHORT_V2_DEV) - set(WALL_COHORT_V2_DEV_HOLDOUT)
    assert set(WALL_COHORT_V2_DEV_HOLDOUT) <= set(SEALED)
    assert set(FIT) | set(DEV) == set(WALL_COHORT_V2_TRAIN)


def bucket(anchors) -> dict[str, list[str]]:
    out = defaultdict(list)
    for a in anchors:
        out[split_of(a)].append(a)
    return dict(out)


def mean_by_split(scores: dict[str, float]) -> dict[str, dict]:
    """``scores`` maps anchor -> scalar.  Sealed is returned but must not drive selection."""
    acc: dict[str, list[float]] = {"fit": [], "dev": [], "sealed": [], "other": []}
    for a, s in scores.items():
        if s != s:  # nan
            continue
        acc[split_of(a)].append(float(s))
    out = {}
    for k, vs in acc.items():
        out[k] = dict(n=len(vs), mean=(float(sum(vs) / len(vs)) if vs else None))
    return out


def format_split_means(scores: dict[str, float], *, width: int = 44) -> str:
    m = mean_by_split(scores)
    parts = []
    for k in ("fit", "dev"):
        row = m[k]
        if row["n"] == 0 or row["mean"] is None:
            parts.append("%s n=0" % k.upper())
        else:
            parts.append("%s n=%d %.4f" % (k.upper(), row["n"], row["mean"]))
    if m["sealed"]["n"]:
        parts.append("SEALED n=%d (do not select)" % m["sealed"]["n"])
    return "  ".join(parts)
