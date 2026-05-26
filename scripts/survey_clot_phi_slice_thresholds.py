"""Clot vs non-clot feature separation and mask-slice sweep (patient007 focus).

Usage::

    python scripts/survey_clot_phi_slice_thresholds.py
    python scripts/survey_clot_phi_slice_thresholds.py --anchor patient007
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import torch

from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND, VesselConfig
from src.core_physics.clot_phi_simple import (
    _wall_mask_from_data,
    cap_mu_eff_si,
    clot_phi_thresh_si,
    dgamma_dx_slice_mask,
    gt_gamma_dot_nd,
    gt_neg_dgamma_dx_phys,
    supervision_region_mask,
)
from src.core_physics.clot_kinematics_fields import compute_clot_kinematics_fields
from src.utils.paths import get_project_root


def _stats(name: str, pos: np.ndarray, neg: np.ndarray) -> None:
    if pos.size == 0:
        return
    print(
        f"    {name:14s}  clot med={np.median(pos):8.2f} p90={np.percentile(pos, 90):8.2f}"
        f"  |  non med={np.median(neg):8.2f} p90={np.percentile(neg, 90):8.2f}"
    )


def _metrics(mask: torch.Tensor, clot: torch.Tensor) -> tuple[int, int, float, float]:
    m = mask.bool()
    tp = int((m & clot).sum())
    n = int(m.sum())
    n_clot = int(clot.sum())
    prec = tp / max(n, 1)
    rec = tp / max(n_clot, 1)
    return n, tp, prec, rec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", default="", help="Single stem (default: all)")
    args = parser.parse_args()

    os.environ.setdefault("CLOT_PHI_MASK_MODE", "neighbor")
    os.environ.setdefault("CLOT_PHI_CENTER_EXCLUDE_FRAC", "0.10")
    os.environ.setdefault("CLOT_PHI_SHEAR_MIN_FRAC", "0")
    os.environ.setdefault("CLOT_PHI_DGAMMA_SLICE", "0")

    root = get_project_root()
    anchor_dir = root / VesselConfig(phase="biochem_anchors").graph_output_dir
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    dev = torch.device("cpu")

    paths = sorted(anchor_dir.glob("*.pt"))
    if args.anchor.strip():
        paths = [anchor_dir / f"{args.anchor.strip()}.pt"]

    for path in paths:
        if not path.is_file():
            continue
        data = torch.load(path, map_location="cpu", weights_only=False)
        ti_f = int(data.y.shape[0]) - 1
        mu_f = cap_mu_eff_si(phys.viscosity_nd_to_si(data.y[ti_f][:, STATE_CHANNEL_MU_EFF_ND]))
        clot = mu_f >= clot_phi_thresh_si(phys)
        base = supervision_region_mask(data, dev, mu_f, phys)
        wall = _wall_mask_from_data(data, dev, int(data.num_nodes))
        pos_m = clot & base
        neg_m = (~clot) & base

        props = {"u_ref": data.u_ref, "d_bar": data.d_bar}
        y0 = data.y[0]
        f0 = compute_clot_kinematics_fields(data, y0[:, 0], y0[:, 1], bio, props)
        neg_dx0 = (-f0.dgamma_dx_phys).clamp(min=0.0)
        g0 = gt_gamma_dot_nd(data, 0, dev)

        print(f"\n=== {path.stem}  mask n={int(base.sum())}  clot+={int(pos_m.sum())}  ===")
        _stats("neg_dx_t0", neg_dx0[pos_m].numpy(), neg_dx0[neg_m].numpy())
        _stats("gamma_nd_t0", g0[pos_m].numpy(), g0[neg_m].numpy())

        off = base & ~wall
        print(f"    off-wall only: n={int(off.sum())} clot={int((clot & off).sum())}")

        n0, tp0, p0, r0 = _metrics(base, clot)
        print(f"    baseline           n={n0:4d}  clot={tp0:3d}  rec={r0:.3f}  prec={p0:.3f}")

        sliced = dgamma_dx_slice_mask(data, dev, base, clot, bio)
        n1, tp1, p1, r1 = _metrics(sliced, clot)
        print(f"    dgamma slice         n={n1:4d}  clot={tp1:3d}  rec={r1:.3f}  prec={p1:.3f}")

        os.environ["CLOT_PHI_SHEAR_MIN_FRAC"] = "0.5"
        os.environ["CLOT_PHI_SHEAR_WALL_EXEMPT"] = "1"
        shear = supervision_region_mask(data, dev, mu_f, phys)
        os.environ["CLOT_PHI_SHEAR_MIN_FRAC"] = "0"
        n2, tp2, p2, r2 = _metrics(shear, clot)
        print(f"    shear 0.5 wall ex.   n={n2:4d}  clot={tp2:3d}  rec={r2:.3f}  prec={p2:.3f}")


if __name__ == "__main__":
    main()
