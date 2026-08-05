"""`clot_score_true` must punish under- and over-prediction symmetrically.

The legacy `clot_guiding` score is `0.5*dilation_iou(3 hop) + 0.5*f_beta(beta=0.5, 3 hop)`:
3-hop tolerant on both terms, and beta=0.5 weights precision 4x over recall. A model that
commits half the required mass very precisely therefore outranks one that commits the right
amount -- the drift observed in every WC leg (docs/GENERALIZATION_PLAN.md s2b-quinquies).
"""

from __future__ import annotations

import torch

from src.evaluation.clot_relaxed_metrics import (
    clot_score_true,
    compute_clot_relaxed_metrics,
)


def _chain(n: int) -> torch.Tensor:
    """Path graph 0-1-2-...-(n-1) as an undirected edge_index."""
    src = torch.arange(n - 1, dtype=torch.long)
    dst = src + 1
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)


def _mask(n: int, idx: range | list[int]) -> torch.Tensor:
    m = torch.zeros(n, dtype=torch.float32)
    m[list(idx)] = 1.0
    return m


def test_exact_match_scores_one() -> None:
    n = 60
    ei = _chain(n)
    gt = _mask(n, range(20, 40))
    m = compute_clot_relaxed_metrics(gt.clone(), gt, ei)
    assert m["clot_score_true"] == 1.0
    assert m["clot_mass_ratio"] == 1.0


def test_under_and_over_prediction_penalised_symmetrically() -> None:
    """Half the mass and double the mass should score comparably -- neither is preferred."""
    n = 200
    ei = _chain(n)
    gt = _mask(n, range(80, 120))          # 40 nodes
    under = _mask(n, range(80, 100))       # 20 nodes, all correct
    over = _mask(n, range(80, 160))        # 80 nodes, covers GT plus 40 spurious

    s_under = compute_clot_relaxed_metrics(under, gt, ei)
    s_over = compute_clot_relaxed_metrics(over, gt, ei)

    assert s_under["clot_mass_ratio"] < 1.0 < s_over["clot_mass_ratio"]
    # Symmetric within a small tolerance (1-hop dilation helps the contiguous cases equally).
    assert abs(s_under["clot_score_true"] - s_over["clot_score_true"]) < 0.15


def test_true_score_prefers_correct_mass_over_precise_sliver() -> None:
    """The key regression: legacy guiding prefers the sliver, true score must not."""
    n = 200
    ei = _chain(n)
    gt = _mask(n, range(80, 120))
    sliver = _mask(n, range(80, 86))        # 6 of 40 nodes, perfect precision
    honest = _mask(n, range(78, 122))       # right mass, slightly offset ends

    s_sliver = compute_clot_relaxed_metrics(sliver, gt, ei)
    s_honest = compute_clot_relaxed_metrics(honest, gt, ei)

    assert s_honest["clot_score_true"] > s_sliver["clot_score_true"]
    # And the sliver's precision-flattering numbers are exactly what misled us.
    assert s_sliver["clot_relaxed_prec"] == 1.0


def test_empty_gt_uses_graded_restraint() -> None:
    n = 60
    ei = _chain(n)
    gt = torch.zeros(n, dtype=torch.float32)
    quiet = compute_clot_relaxed_metrics(_mask(n, [10]), gt, ei)
    spray = compute_clot_relaxed_metrics(_mask(n, range(10, 50)), gt, ei)
    silent = compute_clot_relaxed_metrics(gt.clone(), gt, ei)

    assert silent["clot_score_true"] == 1.0
    assert quiet["clot_score_true"] > spray["clot_score_true"]


def test_score_true_is_mean_of_components() -> None:
    assert clot_score_true(0.2, 0.6) == 0.4
    assert clot_score_true(1.0, 1.0) == 1.0
    assert clot_score_true(0.0, 0.0) == 0.0
