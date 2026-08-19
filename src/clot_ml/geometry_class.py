"""Geometry classes: aneurysm / stenosis / baseline, with an explicit abstain.

Stenoses and aneurysms are the class that matters most, and `patient039`-`patient044` are
in it.  Rather than hard-code that list, the class is **measured** from the mesh's own lumen
width and then checked against it -- so it transfers to an unlabelled vessel.

TWO SCALARS, both dimensionless (normalised by the vessel's own median width, so calibre
does not matter), measured over wall nodes **more than 12 hops from the inlet and outlet**
so the cut ends cannot masquerade as pathology, and locally averaged along the wall so a
single bad node cannot:

    bulge      p98(smoothed width) / median      a local dilatation
    narrowing  p2 (smoothed width) / median      a local constriction

THRESHOLDS AND WHAT THEY SELECT (measured over all 34 vessels):

    aneurysm   bulge >= 2.0      -> 039 (3.48), 043 (2.83), 040 (2.57)
    stenosis   narrowing <= 0.40 -> 041 (0.281), 042 (0.292), 044 (0.323)
    baseline   everything else; the closest baseline vessels are 016 (bulge 1.68) and
               012 (narrowing 0.441), so both cuts sit in a real gap.

That is exactly the user-designated set, split into the two classes, with margin.

THE ABSTAIN MATTERS.  `width_nd` is **unusable on 9 of 34 vessels**: 001/010/011 read a
constant 1.000 with a 10x spike, and 003/004/005/006/007/008 read ~0.12, neither of which
is anatomy.  Those return ``"unknown"`` rather than a confident wrong label, and
``USER_DESIGNATED`` is consulted as the override.  Fix the channel and the abstain goes away.

CONSEQUENCE FOR THE PROTOCOL, worth stating plainly: **DEV (040/041/044) is entirely
priority-class and FIT is entirely baseline.**  Every FIT-vs-DEV difference in
`docs/PHASE9_ML.md` is therefore confounded with a geometry-class difference, and neither
split can certify the other.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

BULGE_ANEURYSM = 2.0
NARROWING_STENOSIS = 0.40
BOUNDARY_HOPS = 12
# Usable-width guard: outside this band the channel is not measuring anatomy.
WIDTH_OK_LO, WIDTH_OK_HI = 0.40, 5.0

USER_DESIGNATED = {
    "patient039": "aneurysm", "patient040": "aneurysm", "patient043": "aneurysm",
    "patient041": "stenosis", "patient042": "stenosis", "patient044": "stenosis",
}
PRIORITY = ("aneurysm", "stenosis", "stenosis+aneurysm")


def _adj(ei, n):
    A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
    return ((A + A.T) > 0).astype(np.int8)


def width_stats(data) -> dict:
    ch = {c: i for i, c in enumerate(data.x_channel_names.split(","))}
    nan = dict(bulge=float("nan"), narrowing=float("nan"), usable=False)
    if "width_nd" not in ch:
        return nan
    x = data.x.detach().cpu().numpy()
    w = x[:, ch["width_nd"]].astype(np.float64)
    wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
    n = len(wall)
    A = _adj(data.edge_index.detach().cpu().numpy(), n)

    io_ = np.zeros(n, bool)
    for k in ("mask_inlet", "mask_outlet"):
        m = getattr(data, k, None)
        if m is not None:
            io_ |= m.reshape(-1).bool().cpu().numpy()
    far = np.ones(n, bool)
    if io_.any():
        cur, d = io_.copy(), np.full(n, 99, np.int16)
        d[cur] = 0
        for h in range(1, BOUNDARY_HOPS + 1):
            nxt = ((A @ cur.astype(np.int8)) > 0) & ~cur
            if not nxt.any():
                break
            d[nxt] = h
            cur = cur | nxt
        far = d > BOUNDARY_HOPS

    sel = wall & far & (w > 0)
    if int(sel.sum()) < 30:
        return nan
    cnt = np.asarray(A @ sel.astype(np.float64)).reshape(-1)
    sm = np.asarray(A @ np.where(sel, w, 0.0)).reshape(-1) / np.maximum(cnt, 1.0)
    ws, sml = w[sel], sm[sel]
    med = float(np.median(ws))
    if med <= 0:
        return nan
    lo, hi = float(np.percentile(ws, 5) / med), float(np.percentile(ws, 95) / med)
    usable = (WIDTH_OK_LO <= lo) and (hi <= WIDTH_OK_HI)
    return dict(bulge=float(np.percentile(sml, 98) / med),
                narrowing=float(np.percentile(sml, 2) / med),
                width_median=med, usable=bool(usable))


def classify(stats: dict, anchor: str | None = None) -> str:
    if not stats.get("usable", False):
        return USER_DESIGNATED.get(anchor or "", "unknown")
    b, nr = stats.get("bulge", np.nan), stats.get("narrowing", np.nan)
    an = b == b and b >= BULGE_ANEURYSM
    st = nr == nr and nr <= NARROWING_STENOSIS
    if an and st:
        return "stenosis+aneurysm"
    if an:
        return "aneurysm"
    if st:
        return "stenosis"
    return "baseline"


def is_priority(cls: str) -> bool:
    return cls in PRIORITY


def classify_cohort(anchors, load) -> dict:
    out = {}
    for a in anchors:
        d = load(a)
        if d is None:
            continue
        s = width_stats(d)
        out[a] = dict(cls=classify(s, a), **s)
    return out
