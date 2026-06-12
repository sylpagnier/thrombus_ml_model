"""Quick probe: ceiling-masked clot_shape on a few rule architectures."""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ.update(
    {
        "BIOCHEM_PRIOR_COMSOL_ALIGNED": "1",
        "CLOT_PHI_HARD_SUPPORT_PROJECTION": "1",
        "CLOT_PHI_SUPPORT_BAND": "ceiling_growth",
        "CLOT_SHAPE_MU_THRESH_SI": "0.055",
        "CLOT_SHAPE_EVAL_MASK": "ceiling",
        "CLOT_PHI_FIXED_MU_FROM_PHI": "1",
    }
)

import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_temporal_growth_rules import (
    comprehensive_rule_architecture_grid,
    eval_temporal_rule_on_anchor,
)
from src.utils.channel_schema import infer_missing_schema

phys = PhysicsConfig(phase="biochem")
bio = BiochemConfig(phase="biochem")
device = torch.device("cpu")
data = torch.load(
    "data/processed/graphs_biochem_anchors/patient007.pt",
    map_location="cpu",
    weights_only=False,
)
data = infer_missing_schema(data, phase_hint="biochem")

names = (
    "static_global",
    "ranked_onset_global",
    "loc_rank_both_t25_s15_ndx45",
    "loc_rank_lower_t25_s15_ndx45",
    "hop_growth_global",
)
rules = [r for r in comprehensive_rule_architecture_grid() if r.name in names]
for cfg in rules:
    r = eval_temporal_rule_on_anchor(
        data, cfg, stem="patient007", device=device, phys_cfg=phys, bio_cfg=bio
    )
    print(
        f"{cfg.name:35s} bandF1={r['tfinal_band_f1']:.3f} "
        f"shape={r['tfinal_clot_shape']:.3f} pred+={r['tfinal_clot_pred_frac']:.3f}"
    )
