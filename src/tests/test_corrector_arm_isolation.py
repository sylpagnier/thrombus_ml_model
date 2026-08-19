"""The flow-coupled arm must be ISOLATED: nothing it adds may move an existing arm.

``mode='corrector'`` was added to ``physics_wall_model.predict_phi`` as a research arm while
the shipped prediction path (``scripts/predict_wall_clot.py``) stays on the frozen t=0 gate.
These pin that separation, plus the one duplication risk in the promotion: the arm seeds
itself from ``predicted_seed_mask``, which re-implements the shipped mask construction.  If
those two drift, the arm is seeded with a clot the model does not actually predict and every
number it produces is measuring something else.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.physics_wall_model import (
    CorrectorArm, predict_phi, predicted_seed_mask, t0_flow_fields,
)

ANCHORS = Path("data/processed/graphs_biochem_anchors")


@pytest.fixture(scope="module")
def anchor():
    paths = [p for p in sorted(ANCHORS.glob("patient0*.pt")) if "mirror" not in p.name]
    if not paths:
        pytest.skip("no biochem anchor graphs")
    return torch.load(str(paths[0]), map_location="cpu", weights_only=False)


def test_seed_mask_matches_the_shipped_predictor_exactly(anchor):
    """``predicted_seed_mask`` must equal ``predict_wall_clot``'s wall mask, node for node."""
    import sys
    repo = Path(__file__).resolve().parents[2]
    if str(repo / "scripts") not in sys.path:
        sys.path.insert(0, str(repo / "scripts"))
    from predict_wall_clot import GROW_HOPS, RELAX, STENCIL, predict_wall_clot

    bio = BiochemConfig(phase="biochem")
    shipped, _ = predict_wall_clot(anchor, bio, flow="gt", lumen=False)
    fields = t0_flow_fields(anchor, bio, hops=STENCIL["gt"], flow_source="gt")
    seed, _, _ = predicted_seed_mask(anchor, bio, fields, relax=RELAX, grow_hops=GROW_HOPS)
    assert np.array_equal(seed, shipped), (
        "the arm's seed mask has drifted from the shipped prediction: "
        f"{int(seed.sum())} vs {int(shipped.sum())} nodes"
    )


def test_existing_modes_do_not_require_the_arm(anchor):
    """``gate`` and ``ode`` must run with no corrector present and stay deterministic."""
    bio = BiochemConfig(phase="biochem")
    a1, _, _ = predict_phi(anchor, bio, mode="gate")
    a2, _, _ = predict_phi(anchor, bio, mode="gate")
    b1, _, m1 = predict_phi(anchor, bio, mode="ode", da_scale=100.0)
    b2, _, m2 = predict_phi(anchor, bio, mode="ode", da_scale=100.0)
    assert torch.equal(a1, a2) and torch.equal(b1, b2)
    assert np.array_equal(m1, m2)
    # the ODE arm at a saturated da_scale collapses onto the bare gate (PHASE3_RESULTS 3)
    assert torch.equal(a1, b1)


def test_corrector_mode_refuses_to_run_without_an_arm(anchor):
    bio = BiochemConfig(phase="biochem")
    with pytest.raises(ValueError, match="CorrectorArm"):
        predict_phi(anchor, bio, mode="corrector")


def test_seed_ramp_zero_is_the_unseeded_loop(anchor):
    """``seed_ramp=0`` must reproduce the occlusion-driven loop, i.e. seed nothing early."""
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    fields = t0_flow_fields(anchor, bio, hops=3, flow_source="gt")
    from src.core_physics.physics_wall_model import corrector_blockage

    arm = CorrectorArm(corrector=None, phys_cfg=phys, device=torch.device("cpu"),
                       seed_ramp=0.0)
    blk = corrector_blockage(anchor, bio, fields, arm, hops=3)
    n = int(anchor.num_nodes)
    # with no committed mass and no seeding there is nothing to occlude, so the corrector
    # is never invoked and the gate is handed straight back
    g = blk(np.zeros(n), fields.gate, 0)
    assert np.array_equal(g, fields.gate)
    assert blk.state["calls"] == 0
