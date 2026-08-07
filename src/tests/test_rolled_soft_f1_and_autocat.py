"""Tests for the s12.6 objective/growth-law changes.

Covers the two things that made v3-v6 uninterpretable, so they cannot regress silently:
  * the soft-occupancy scale bug (terms numerically dead at the shipped soft_k),
  * the soft-F_beta rolled-state surrogate being monotone in the metric,
  * the autocatalytic growth term actually amplifying, and surviving freeze_backbone.
"""
from __future__ import annotations

import pytest
import torch

from src.architecture.pushforward_config import PushforwardConfig, use_pushforward_config
from src.core_physics.species_pushforward_continuous import (
    SpeciesDualHeadContinuousGNN,
    _sd,
    rolled_final_mass_fp_penalty,
    rolled_soft_f1_loss,
    soft_commit_prob,
)
from src.training.biochem_species_scope import MAT_CHANNEL, pushforward_state_bulk_indices
from src.training.train_offwall_growth import freeze_growth_backbone

THR = 1e-4


def _state(n_committed: int, n: int = 2000) -> torch.Tensor:
    bulk = pushforward_state_bulk_indices()
    mi = bulk.index(MAT_CHANNEL)
    s = torch.zeros(n, len(bulk))
    s[:n_committed, mi] = 2.0 * THR
    return s


def _gt(n_pos: int = 50, n: int = 2000) -> torch.Tensor:
    bulk = pushforward_state_bulk_indices()
    mi = bulk.index(MAT_CHANNEL)
    g = torch.zeros(n, len(bulk))
    g[:n_pos, mi] = 1.0
    return g


def test_absolute_soft_k_is_numerically_dead():
    """The shipped soft_k=40 cannot separate committed from empty at thr=1e-4.

    This is the measured root cause of s9.12's "the brake moved the rollout ~1%".
    Pinned as a regression guard on the DIAGNOSIS, not as desired behaviour.
    """
    cfg = PushforwardConfig(species_scope="mat", rolled_soft_k_relative=False)
    with use_pushforward_config(cfg):
        empty = soft_commit_prob(torch.tensor(0.0), THR, 40.0)
        committed = soft_commit_prob(torch.tensor(10 * THR), THR, 40.0)
        assert abs(float(empty) - 0.5) < 0.002
        assert abs(float(committed) - 0.5) < 0.02
        assert float(committed) - float(empty) < 0.02  # no usable separation


def test_relative_soft_k_separates():
    cfg = PushforwardConfig(species_scope="mat", rolled_soft_k_relative=True)
    with use_pushforward_config(cfg):
        empty = float(soft_commit_prob(torch.tensor(0.0), THR, 10.0))
        committed = float(soft_commit_prob(torch.tensor(2 * THR), THR, 10.0))
        assert empty < 0.01
        assert committed > 0.99


def test_brake_regains_dynamic_range():
    """0.12% -> ~98% across empty..12x over-painted (s12.6.1)."""
    base = dict(species_scope="mat", final_mass_penalty=1.5, final_mass_target=1.2,
                final_prec_fp_penalty=1.0, loss_scale=1.0)
    gt, mask = _gt(), torch.ones(2000, dtype=torch.bool)
    spans = {}
    for rel in (False, True):
        with use_pushforward_config(PushforwardConfig(**base, rolled_soft_k_relative=rel,
                                                      rolled_soft_f1_k=10.0)):
            vals = [float(rolled_final_mass_fp_penalty(_state(k), gt, mask))
                    for k in (0, 50, 292, 600)]
            spans[rel] = (max(vals) - min(vals)) / max(vals)
    assert spans[False] < 0.01, "absolute mode should be (bug-compatibly) inert"
    assert spans[True] > 0.5, "relative mode must respond to the rollout"


def test_soft_f1_is_monotone_in_the_metric():
    """Ranks the ACTUAL observed basins: fp=110 (v2 ep5) must beat fp=292 (saturated)."""
    cfg = PushforwardConfig(species_scope="mat", rolled_soft_f1_weight=1.0,
                            rolled_soft_f1_beta=1.0, rolled_soft_f1_k=10.0,
                            rolled_soft_k_relative=True, loss_scale=1.0)
    # NB: state/GT must be built INSIDE the config context -- the channel layout comes from
    # the active species scope, so building them outside silently mismatches the Mat column.
    with use_pushforward_config(cfg):
        gt, mask = _gt(), torch.ones(2000, dtype=torch.bool)
        perfect = float(rolled_soft_f1_loss(_state(50), gt, mask))
        good = float(rolled_soft_f1_loss(_state(160), gt, mask))   # fp=110
        bad = float(rolled_soft_f1_loss(_state(342), gt, mask))    # fp=292
        empty = float(rolled_soft_f1_loss(_state(0), gt, mask))
        assert perfect < good < bad < empty
        assert perfect < 0.05 and empty > 0.95


def test_soft_f1_has_gradient_in_the_saturated_basin():
    """fp_frac saturates toward 1.0 at fp=292; the surrogate must not."""
    cfg = PushforwardConfig(species_scope="mat", rolled_soft_f1_weight=1.0,
                            rolled_soft_f1_k=10.0, rolled_soft_k_relative=True, loss_scale=1.0)
    with use_pushforward_config(cfg):
        gt, mask = _gt(), torch.ones(2000, dtype=torch.bool)
        p = _state(342).requires_grad_(True)
        rolled_soft_f1_loss(p, gt, mask).backward()
        assert p.grad is not None and float(p.grad.norm()) > 0.0


def test_soft_f1_beta_tilts_toward_recall():
    cfg = dict(species_scope="mat", rolled_soft_f1_weight=1.0, rolled_soft_f1_k=10.0,
               rolled_soft_k_relative=True, loss_scale=1.0)
    with use_pushforward_config(PushforwardConfig(**cfg, rolled_soft_f1_beta=2.0)):
        gt, mask = _gt(), torch.ones(2000, dtype=torch.bool)
        under, over = _state(20), _state(120)
        assert float(rolled_soft_f1_loss(under, gt, mask)) > float(rolled_soft_f1_loss(over, gt, mask))
    with use_pushforward_config(PushforwardConfig(**cfg, rolled_soft_f1_beta=0.5)):
        gt, mask = _gt(), torch.ones(2000, dtype=torch.bool)
        under, over = _state(20), _state(120)
        assert float(rolled_soft_f1_loss(under, gt, mask)) < float(rolled_soft_f1_loss(over, gt, mask))


def test_disabled_terms_are_exactly_zero():
    """Default-off must not perturb any historical leg."""
    with use_pushforward_config(PushforwardConfig(species_scope="mat")):
        gt, mask = _gt(), torch.ones(2000, dtype=torch.bool)
        assert float(rolled_soft_f1_loss(_state(342), gt, mask)) == 0.0


def _tiny_model(autocat: bool) -> tuple[SpeciesDualHeadContinuousGNN, torch.Tensor, torch.Tensor]:
    n, in_dim = 60, 32
    m = SpeciesDualHeadContinuousGNN(in_dim, hidden=16)
    x = torch.randn(n, in_dim)
    ei = torch.stack([torch.arange(n).repeat_interleave(2), torch.randint(0, n, (2 * n,))])
    return m, x, ei


def test_autocatalytic_amplifies_near_committed_material():
    cfg = PushforwardConfig(species_scope="mat", dual_head=True, autocatalytic_growth=True,
                            autocat_k_dep_init=1.0, autocat_k_auto_init=1.0, autocat_alpha=0.8)
    with use_pushforward_config(cfg):
        torch.manual_seed(0)
        m, x, ei = _tiny_model(True)
        seeded = torch.zeros(60, _sd())
        seeded[:10, :] = 1e-3
        bare = torch.zeros(60, _sd())
        d_seeded, _, _ = m.forward_decoupled(x, ei, seeded)
        d_bare, _, _ = m.forward_decoupled(x, ei, bare)
        mi = pushforward_state_bulk_indices().index(MAT_CHANNEL)
        near = ei[1][torch.isin(ei[0], torch.arange(10))].unique()
        assert float(d_seeded[near, mi].mean()) > float(d_bare[near, mi].mean())


def test_autocatalytic_params_survive_freeze_backbone():
    """log_k_* ARE the growth law; freezing them silently disables change D."""
    cfg = PushforwardConfig(species_scope="mat", dual_head=True, autocatalytic_growth=True)
    with use_pushforward_config(cfg):
        m, _, _ = _tiny_model(True)
        freeze_growth_backbone(m)
        trainable = {k for k, p in m.named_parameters() if p.requires_grad}
        assert "log_k_dep" in trainable and "log_k_auto" in trainable


def test_autocatalytic_off_by_default():
    with use_pushforward_config(PushforwardConfig(species_scope="mat", dual_head=True)):
        m, x, ei = _tiny_model(False)
        assert m.log_k_dep is None and m.log_k_auto is None


# --- Per-vessel label threshold (WALL_MODEL_PLAN.md 20.3 / 21) -------------------------------

def test_label_thresh_defaults_to_absolute_and_is_leak_safe():
    """Default must be the historical absolute threshold, and rel_max must no-op when unbound."""
    from src.core_physics.species_pushforward_continuous import (
        continuous_mat_commit_thresh, mat_label_thresh,
    )
    with use_pushforward_config(PushforwardConfig(species_scope="mat")):
        assert mat_label_thresh() == continuous_mat_commit_thresh()
    # rel_max with no vessel bound must fall back rather than guess
    with use_pushforward_config(
        PushforwardConfig(species_scope="mat", mat_label_thresh_mode="rel_max")
    ):
        assert mat_label_thresh() == continuous_mat_commit_thresh()


def test_label_thresh_scales_per_vessel():
    from src.core_physics.species_pushforward_continuous import (
        mat_label_thresh, use_vessel_mat_max,
    )
    cfg = PushforwardConfig(species_scope="mat", mat_label_thresh_mode="rel_max",
                            mat_label_rel_frac=0.10)
    with use_pushforward_config(cfg):
        with use_vessel_mat_max(4.31e-3):          # patient041
            assert mat_label_thresh() == pytest.approx(4.31e-4)
        with use_vessel_mat_max(3.40e-4):          # patient024, 12.7x smaller peak
            assert mat_label_thresh() == pytest.approx(3.40e-5)


def test_prediction_side_threshold_never_uses_vessel_max():
    """The model's own commit threshold must stay absolute -- using GT max would be a leak."""
    from src.core_physics.species_pushforward_continuous import (
        continuous_mat_commit_thresh, use_vessel_mat_max,
    )
    cfg = PushforwardConfig(species_scope="mat", mat_label_thresh_mode="rel_max")
    with use_pushforward_config(cfg):
        before = continuous_mat_commit_thresh()
        with use_vessel_mat_max(1.5e-2):
            assert continuous_mat_commit_thresh() == before
