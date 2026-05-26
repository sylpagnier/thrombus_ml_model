"""ClotPhiHybrid MLP depth builds and runs forward."""

import os

import torch

from src.core_physics.clot_phi_simple import ClotPhiHybrid, build_clot_phi_model


def test_clot_phi_hybrid_depth2_forward():
    os.environ["CLOT_PHI_HYBRID"] = "1"
    os.environ["CLOT_PHI_MODEL"] = "mlp"
    os.environ["CLOT_PHI_MLP_DEPTH"] = "2"
    os.environ["CLOT_PHI_DROPOUT"] = "0.1"
    m = build_clot_phi_model(in_dim=3, hidden=8)
    assert isinstance(m, ClotPhiHybrid)
    x = torch.randn(20, 3)
    logits = m.forward_logits(x)
    dlog = m.forward_delta_log_mu(x)
    assert logits.shape == (20,)
    assert dlog.shape == (20,)
