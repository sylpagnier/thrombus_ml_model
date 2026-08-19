"""Extract the wall-node state trajectories every Phase-6 fit needs, once per vessel.

WHY THIS EXISTS.  The AP closure (PHASE6_HANDOFF 2.1) and the Damkohler ratio (1) were
both measured on ``outputs/comsol_p007_wall.npz`` -- COMSOL's raw export for **patient007,
which is SEALED**.  Fitting a constant there is a protocol violation (6.1).

The packs turn out to carry everything the export does for the surface problem:
``M_log1p_nd``/``Mas_log1p_nd``/``Mat_log1p_nd`` at all 201 timesteps alongside
``RP``/``AP``.  Verified node-by-node against the export on patient007 (matched by
position, 583/583 wall nodes):

    Mas   corr 0.999978   median(pack/export) 1.0000
    Mat   corr 0.999995   median(pack/export) 1.0000
    ap    corr 1.000000   median(pack/export) 1.0000

with ``surface = expm1(nd) * 7e10`` -> [plt/cm^2] and ``bulk = expm1(nd) * 2.5e14`` ->
[plt/cm^3] (the latter is exactly what ``species_fields.gt_species_trajectory`` does).
So the whole cohort is available and patient007 need never be fit against.

Each pack is 50-350 MB and only ~600-2000 of its ~20k nodes are wall; this writes the
wall slice to a small npz so the fits can sweep the cohort in seconds.

    python scripts/build_wall_species_cache.py            # train + dev + sealed
    python scripts/build_wall_species_cache.py --only patient020
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

from src.biochem_gnn.mat_growth_simple import (  # noqa: E402
    WALL_COHORT_V2_GENERALIZATION, WALL_COHORT_V2_TRAIN,
)
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.mls_gradient import (  # noqa: E402
    build_mls_gradient, node_positions, shear_rate_2d,
)
from src.core_physics.physics_wall_model import M_TO_CM  # noqa: E402
from src.core_physics.species_fields import gt_species_trajectory  # noqa: E402
from src.core_physics.temporal_metrics import gt_onset_index  # noqa: E402

DIR = Path("data/processed/graphs_biochem_anchors")
OUT = Path("outputs/wall_species_cache")
SURFACE_ND_TO_CGS = 7.0e10        # expm1(Mas_log1p_nd) * this = [plt/cm^2]
MIN_T = 150
HOPS = 3


def _surface(data, name: str) -> np.ndarray:
    col = data.y_channel_names.split(",").index(name)
    return torch.expm1(data.y[:, :, col].clamp(-10, 8)).numpy().astype(np.float64) * SURFACE_ND_TO_CGS


def extract(name: str, bio, phys, *, with_sr_t: bool = True) -> dict | None:
    p = DIR / f"{name}.pt"
    if not p.exists():
        return None
    d = torch.load(p, map_location="cpu", weights_only=False)
    T = int(d.y.shape[0])
    if T < MIN_T:
        return None
    w = d.mask_wall.reshape(-1).bool().numpy()
    if w.sum() < 16:
        return None

    pos = node_positions(d)
    ei = d.edge_index.detach().cpu().numpy()
    u_ref = float(d.u_ref.reshape(-1)[0])
    d_bar = float(d.d_bar.reshape(-1)[0])
    Dx, Dy = build_mls_gradient(pos, ei, hops=HOPS)

    def sr_at(ti: int) -> np.ndarray:
        u = d.y[ti, :, 0].numpy().astype(np.float64)
        v = d.y[ti, :, 1].numpy().astype(np.float64)
        return shear_rate_2d(Dx @ u, Dy @ u, Dx @ v, Dy @ v) * (u_ref / d_bar)

    sr0 = sr_at(0)
    dsrx0 = (Dx @ sr0) / (d_bar * M_TO_CM)
    sr_t = dsrx_t = None
    if with_sr_t:
        # GT shear AND its x-gradient at every step.  ORACLE input -- for diagnostics only,
        # never a deployable arm.  Both branches of the gate need both fields, so caching
        # only ``sr`` would make the time-varying-GATE oracle uncomputable -- and that gate
        # is the lever PHASE6_HANDOFF 9 says carries the most ordering (rho 0.795 for
        # perfect flow, against 0.866 for perfect flow AND species).
        full = np.stack([sr_at(ti) for ti in range(T)])
        sr_t = full[:, w]
        dsrx_t = np.stack([(Dx @ full[ti]) / (d_bar * M_TO_CM) for ti in range(T)])[:, w]

    sr0_pred = dsrx0_pred = None
    if getattr(d, "u0_pred", None) is not None:
        up = d.u0_pred.reshape(-1).numpy().astype(np.float64)
        vp = d.v0_pred.reshape(-1).numpy().astype(np.float64)
        sr0_pred = shear_rate_2d(Dx @ up, Dy @ up, Dx @ vp, Dy @ vp) * (u_ref / d_bar)
        dsrx0_pred = (Dx @ sr0_pred) / (d_bar * M_TO_CM)

    # wall-restricted adjacency, as (2, E) local indices.  A KD-tree on wall positions
    # would be simpler but jumps the lumen wherever the vessel is narrow, which is exactly
    # where the interesting nodes are -- so smoothing has to follow the MESH.
    loc = -np.ones(len(w), dtype=np.int64)
    loc[w] = np.arange(int(w.sum()))
    keep = w[ei[0]] & w[ei[1]]
    wall_edges = np.stack([loc[ei[0][keep]], loc[ei[1][keep]]])

    rp, ap = gt_species_trajectory(d, bio)
    out = dict(
        wall_edges=wall_edges,
        name=name, t=d.t.reshape(-1).numpy().astype(np.float64), wall_idx=np.where(w)[0],
        n_nodes=np.int64(len(w)), pos=pos[w], u_ref=np.float64(u_ref), d_bar=np.float64(d_bar),
        sr0=sr0[w], dsrx0=dsrx0[w],
        ap=ap[:, w], rp=rp[:, w],
        mas=_surface(d, "Mas_log1p_nd")[:, w], mat=_surface(d, "Mat_log1p_nd")[:, w],
        m_tot=_surface(d, "M_log1p_nd")[:, w],
        gt_onset=gt_onset_index(d, phys, w)[w],
        sealed=np.bool_(name in WALL_COHORT_V2_GENERALIZATION),
    )
    if sr_t is not None:
        out["sr_t"] = sr_t
        out["dsrx_t"] = dsrx_t
    if sr0_pred is not None:
        out["sr0_pred"] = sr0_pred[w]
        out["dsrx0_pred"] = dsrx0_pred[w]
    return out


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--only", default="")
    ap_.add_argument("--no-sr-t", action="store_true")
    ap_.add_argument("--force", action="store_true")
    args = ap_.parse_args()
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    OUT.mkdir(parents=True, exist_ok=True)

    names = ([args.only] if args.only
             else sorted(set(WALL_COHORT_V2_TRAIN) | set(WALL_COHORT_V2_GENERALIZATION)))
    print("%-12s %6s %6s %8s %8s" % ("vessel", "T", "n_wall", "gt_hot", "secs"))
    n_ok = 0
    for n in names:
        dst = OUT / f"{n}.npz"
        if dst.exists() and not args.force:
            print("%-12s   (cached)" % n)
            n_ok += 1
            continue
        t0 = time.time()
        try:
            r = extract(n, bio, phys, with_sr_t=not args.no_sr_t)
        except Exception as exc:                                   # noqa: BLE001
            print("%-12s   FAILED: %s" % (n, exc))
            continue
        if r is None:
            print("%-12s   skipped (missing or T<%d)" % (n, MIN_T))
            continue
        np.savez_compressed(dst, **r)
        print("%-12s %6d %6d %8d %8.1f"
              % (n, len(r["t"]), r["ap"].shape[1], int((r["gt_onset"] >= 0).sum()),
                 time.time() - t0))
        n_ok += 1
    print("\n%d vessels cached in %s" % (n_ok, OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
