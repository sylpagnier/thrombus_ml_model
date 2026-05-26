"""Print clot-phi mask sizes and GT clot coverage for filter combinations.

Usage (repo root)::

    python scripts/survey_clot_phi_mask_filters.py
    python scripts/survey_clot_phi_mask_filters.py --anchor patient007
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch

from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND, VesselConfig
from src.core_physics.clot_phi_simple import (
    _wall_mask_from_data,
    cap_mu_eff_si,
    clot_phi_thresh_si,
    gt_gamma_dot_nd,
    neighbor_supervision_mask,
    supervision_region_mask,
)
from src.utils.paths import get_project_root


def _mask_stats(data, region: torch.Tensor, clot: torch.Tensor, wall: torch.Tensor) -> str:
    m = region.bool()
    n = int(m.sum())
    pos = int((clot & m).sum())
    n_clot = int(clot.sum())
    cov = 100.0 * pos / max(n_clot, 1)
    pf = pos / max(n, 1)
    wk = int((wall & m).sum())
    wtot = int((wall).sum())
    return f"n={n:4d}  clot={pos:3d}/{n_clot} ({cov:5.1f}%)  pos_frac={pf:.3f}  wall={wk}/{wtot}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Survey clot-phi supervision masks")
    parser.add_argument("--anchor", default="", help="Single stem (default: all anchors)")
    args = parser.parse_args()

    root = get_project_root()
    anchor_dir = root / VesselConfig(phase="biochem_anchors").graph_output_dir
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    dev = torch.device("cpu")

    paths = sorted(anchor_dir.glob("*.pt"))
    if args.anchor.strip():
        paths = [anchor_dir / f"{args.anchor.strip()}.pt"]

    presets = [
        ("neighbor (baseline)", {"CLOT_PHI_SHEAR_MIN_FRAC": "0"}),
        ("+ center 10% (default)", {"CLOT_PHI_SHEAR_MIN_FRAC": "0", "CLOT_PHI_CENTER_EXCLUDE_FRAC": "0.10"}),
        ("+ shear 0.5 global, wall exempt", {"CLOT_PHI_SHEAR_MIN_FRAC": "0.5", "CLOT_PHI_SHEAR_WALL_EXEMPT": "1"}),
        ("+ shear 0.5 global, all nodes", {"CLOT_PHI_SHEAR_MIN_FRAC": "0.5", "CLOT_PHI_SHEAR_WALL_EXEMPT": "0"}),
        ("+ shear 0.25 global, wall exempt", {"CLOT_PHI_SHEAR_MIN_FRAC": "0.25", "CLOT_PHI_SHEAR_WALL_EXEMPT": "1"}),
        ("+ shear 0.5 region-max, wall exempt", {
            "CLOT_PHI_SHEAR_MIN_FRAC": "0.5",
            "CLOT_PHI_SHEAR_WALL_EXEMPT": "1",
            "CLOT_PHI_SHEAR_MAX_SCOPE": "region",
        }),
    ]

    for path in paths:
        if not path.is_file():
            print(f"[skip] missing {path}")
            continue
        data = torch.load(path, map_location="cpu", weights_only=False)
        ti = int(data.y.shape[0]) - 1
        mu = cap_mu_eff_si(phys.viscosity_nd_to_si(data.y[ti][:, STATE_CHANNEL_MU_EFF_ND]))
        clot = mu >= clot_phi_thresh_si(phys)
        wall = _wall_mask_from_data(data, dev, int(data.num_nodes))
        g = gt_gamma_dot_nd(data, ti, dev)
        print(f"\n=== {path.stem}  t={ti}  gamma_max={float(g.max()):.2f} (ND) ===")
        for label, env_extra in presets:
            for k, v in {
                "CLOT_PHI_MASK_MODE": "neighbor",
                "CLOT_PHI_CENTER_EXCLUDE_FRAC": "0.10",
                **env_extra,
            }.items():
                os.environ[k] = v
            region = supervision_region_mask(data, dev, mu, phys)
            print(f"  {label:36s}  {_mask_stats(data, region, clot, wall)}")


if __name__ == "__main__":
    main()
