"""Biochem phase: localized L2 velocity error in clotting zones (truth mask)."""

import torch

from src.utils.biochem_masks import biochem_truth_node_mask


def test_localized_l2_velocity_in_clot_zone():
    """Masked L2 between predicted and reference velocity on anchor (clot-truth) nodes only."""
    n = 12
    device = torch.device("cpu")
    pred_uv = torch.randn(n, 2, device=device)
    true_uv = torch.randn(n, 2, device=device)

    class _B:
        batch = None
        is_anchor = torch.zeros(n, dtype=torch.bool)
        is_anchor[4:9] = True

    m = biochem_truth_node_mask(_B(), n, device)
    assert m[4:9].all() and not m[0:4].any()

    diff = (pred_uv - true_uv)[m]
    l2 = torch.mean(diff ** 2)
    assert l2.ndim == 0 and torch.isfinite(l2)


def test_empty_mask_yields_zero_norm_convention():
    """When no anchor nodes exist, localized error should be handled explicitly by callers."""
    n = 5
    device = torch.device("cpu")

    class _B:
        batch = None
        is_anchor = torch.tensor(False)

    m = biochem_truth_node_mask(_B(), n, device)
    assert not m.any()
