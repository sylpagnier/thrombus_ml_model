"""Cache loading, splits, and the shared readout from a per-node score to a mask."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.core_physics.wall_cohort_splits import DEV, FIT

REPO = Path(__file__).resolve().parents[2]


def load_cache(flow: str = "gt") -> dict[str, dict]:
    root = REPO / f"outputs/clot_ml_cache_{flow}"
    out = {}
    for p in sorted(root.glob("*.npz")):
        z = np.load(p, allow_pickle=True)
        out[p.stem] = {k: z[k] for k in z.files}
    return out


def splits(cache: dict) -> tuple[list[str], list[str]]:
    fit = [a for a in FIT if a in cache]
    dev = [a for a in DEV if a in cache]
    return fit, dev


def standardiser(cache: dict, anchors: list[str]):
    X = np.concatenate([cache[a]["X"] for a in anchors], axis=0)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def mask_from_score(score: np.ndarray, thresh: float) -> np.ndarray:
    return score >= thresh


def physics_mask(S: dict) -> np.ndarray:
    """The shipped zero-parameter backbone's own full-mesh mask, from cached fields."""
    from src.config import BiochemConfig
    from src.core_physics.physics_lumen_model import adjacency, grow_into_lumen
    bio = BiochemConfig(phase="biochem")
    wall, ei = S["wall"], S["edge_index"]
    A = adjacency(ei, len(wall)).astype(np.int8)
    cur = (S["gate"] > 0) & wall
    adm = (S["sr"] < float(bio.lss) * 2.0) & wall
    for _ in range(20):
        cur = cur | (((A @ cur.astype(np.int8)) > 0) & adm)
    off = grow_into_lumen(cur, wall, A, S["spd"], S["sr"], lumen_hops=2, speed_thresh=0.2)
    return cur | off


def attach_physics(cache: dict) -> dict:
    """Add the backbone mask, and expose it to the models as an extra feature column."""
    for S in cache.values():
        if "phys_mask" in S:
            continue
        m = physics_mask(S)
        S["phys_mask"] = m
        cols = [str(c) for c in S["cols"]]
        S["sdf"] = S["X"][:, cols.index("sdf_nd")].astype(np.float64)
        S["X"] = np.concatenate([S["X"], m.astype(np.float32).reshape(-1, 1)], axis=1)
        S["cols"] = np.array([str(c) for c in S["cols"]] + ["phys_mask"])
    return cache
