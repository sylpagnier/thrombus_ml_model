"""Unit tests for WC_v7 compound two-model frontier routing."""

from __future__ import annotations

import os

import torch

from src.core_physics.species_pushforward_continuous import (
    _two_model_blend_mask,
    clear_offwall_model_cache,
    two_model_frontier_hops,
    two_model_route,
)


def test_two_model_route_aliases(monkeypatch):
    monkeypatch.setenv("SPECIES_TWO_MODEL_ROUTE", "frontier")
    assert two_model_route() == "frontier"
    monkeypatch.setenv("SPECIES_TWO_MODEL_ROUTE", "growth")
    assert two_model_route() == "frontier"
    monkeypatch.setenv("SPECIES_TWO_MODEL_ROUTE", "wall")
    assert two_model_route() == "wall"
    monkeypatch.delenv("SPECIES_TWO_MODEL_ROUTE", raising=False)
    assert two_model_route() == "wall"


def test_two_model_frontier_hops(monkeypatch):
    monkeypatch.setenv("SPECIES_TWO_MODEL_FRONTIER_HOPS", "3")
    assert two_model_frontier_hops() == 3
    monkeypatch.setenv("SPECIES_TWO_MODEL_FRONTIER_HOPS", "bad")
    assert two_model_frontier_hops() == 2


def test_blend_mask_wall_route_keeps_wall():
    wall = torch.tensor([True, True, False, False])
    log_state = torch.zeros(4, 1)
    # trivial chain edges 0-1-2-3
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    keep = _two_model_blend_mask(
        route="wall", wall_mask=wall, log_state=log_state, edge_index=edge_index
    )
    assert keep.tolist() == [True, True, False, False]


def test_blend_mask_frontier_empty_clot_keeps_all_for_nucleation(monkeypatch):
    monkeypatch.setenv("BIOCHEM_PUSHFORWARD_SPECIES_SCOPE", "mat")
    monkeypatch.setenv("SPECIES_TWO_MODEL_FRONTIER_HOPS", "1")
    wall = torch.tensor([True, True, False, False])
    log_state = torch.zeros(4, 1)  # no committed Mat
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    keep = _two_model_blend_mask(
        route="frontier", wall_mask=wall, log_state=log_state, edge_index=edge_index
    )
    assert bool(keep.all().item())


def test_blend_mask_frontier_hands_growth_zone_to_specialist(monkeypatch):
    monkeypatch.setenv("BIOCHEM_PUSHFORWARD_SPECIES_SCOPE", "mat")
    monkeypatch.setenv("SPECIES_TWO_MODEL_FRONTIER_HOPS", "1")
    # Lower commit thresh so a moderate Mat value counts as committed.
    monkeypatch.setenv("SPECIES_CONTINUOUS_MAT_COMMIT_THRESH", "0.1")
    wall = torch.tensor([True, True, False, False])
    log_state = torch.tensor([[0.0], [1.0], [0.0], [0.0]])  # node 1 committed
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    keep = _two_model_blend_mask(
        route="frontier", wall_mask=wall, log_state=log_state, edge_index=edge_index
    )
    # Growth zone = {0,1,2}; keep_wall = ~growth => only node 3 True
    assert keep.tolist() == [False, False, False, True]


def test_clear_offwall_cache_resets():
    clear_offwall_model_cache()
    # Idempotent; just ensure no exception and env can be empty.
    os.environ.pop("SPECIES_OFFWALL_MODEL_CKPT", None)
    clear_offwall_model_cache()
