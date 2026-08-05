"""Unit tests for WC_v7 compound two-model frontier routing."""

from __future__ import annotations

import os

import torch

from src.architecture.pushforward_config import PushforwardConfig, use_pushforward_config
from src.architecture.runtime_config import BiochemRuntimeConfig, use_biochem_runtime
from src.core_physics.species_pushforward_continuous import (
    _frontier_growth_zone,
    _frontier_offwall_growth_zone,
    _two_model_blend_mask,
    clear_offwall_model_cache,
    parse_two_model_frontier_hops_map,
    two_model_frontier_hops,
    two_model_route,
)


def test_two_model_route_aliases():
    cases = [
        ("frontier", "frontier"),
        ("frontier_offwall", "frontier_offwall"),
        ("frontier_lumen_only", "frontier_offwall"),
        ("growth", "frontier"),
        ("wall", "wall"),
    ]
    for raw, expected in cases:
        rt = BiochemRuntimeConfig.from_kwargs({"two_model_route": raw})
        with use_biochem_runtime(rt):
            assert two_model_route() == expected
    with use_biochem_runtime(BiochemRuntimeConfig.from_kwargs({"two_model_route": "wall"})):
        assert two_model_route() == "wall"


def test_two_model_frontier_hops():
    with use_biochem_runtime(BiochemRuntimeConfig.from_kwargs({"two_model_frontier_hops": 3})):
        assert two_model_frontier_hops() == 3.0
    with use_biochem_runtime(BiochemRuntimeConfig.from_kwargs({"two_model_frontier_hops": 0.5})):
        assert two_model_frontier_hops() == 0.5
    # Invalid env falls back when no active runtime override is present.
    with use_biochem_runtime(None):
        os.environ["SPECIES_TWO_MODEL_FRONTIER_HOPS"] = "bad"
        try:
            assert two_model_frontier_hops() == 2.0
        finally:
            os.environ.pop("SPECIES_TWO_MODEL_FRONTIER_HOPS", None)


def test_parse_frontier_hops_map():
    m = parse_two_model_frontier_hops_map("patient010:0.5,default:1")
    assert m["patient010"] == 0.5
    assert m["default"] == 1.0


def test_frontier_growth_zone_fractional_tighter_than_one():
    wall = torch.tensor([True, True, False, False])
    committed = torch.tensor([False, True, False, False])
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    full_one = _frontier_growth_zone(
        committed=committed, edge_index=edge_index, wall_mask=wall, hops=1.0
    )
    tight = _frontier_growth_zone(
        committed=committed, edge_index=edge_index, wall_mask=wall, hops=0.5
    )
    assert bool(tight[1].item())
    assert int(tight.sum().item()) <= int(full_one.sum().item())


def test_frontier_offwall_growth_zone_strips_wall_nodes():
    wall = torch.tensor([True, True, False, False])
    committed = torch.tensor([False, True, False, False])
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    offwall = _frontier_offwall_growth_zone(
        committed=committed, edge_index=edge_index, wall_mask=wall, hops=0.5
    )
    assert offwall.tolist() == [False, False, True, False]


def test_blend_mask_wall_route_keeps_wall():
    wall = torch.tensor([True, True, False, False])
    log_state = torch.zeros(4, 1)
    # trivial chain edges 0-1-2-3
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    keep = _two_model_blend_mask(
        route="wall", wall_mask=wall, log_state=log_state, edge_index=edge_index
    )
    assert keep.tolist() == [True, True, False, False]


def test_blend_mask_frontier_empty_clot_keeps_all_for_nucleation():
    pf = PushforwardConfig(species_scope="mat", channels=(11,), mat_commit_thresh=0.002)
    rt = BiochemRuntimeConfig.from_kwargs({"two_model_frontier_hops": 1})
    wall = torch.tensor([True, True, False, False])
    log_state = torch.zeros(4, 1)  # no committed Mat
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    with use_pushforward_config(pf), use_biochem_runtime(rt):
        keep = _two_model_blend_mask(
            route="frontier", wall_mask=wall, log_state=log_state, edge_index=edge_index
        )
    assert bool(keep.all().item())


def test_blend_mask_frontier_hands_growth_zone_to_specialist():
    pf = PushforwardConfig(species_scope="mat", channels=(11,), mat_commit_thresh=0.1)
    rt = BiochemRuntimeConfig.from_kwargs({"two_model_frontier_hops": 1})
    wall = torch.tensor([True, True, False, False])
    log_state = torch.tensor([[0.0], [1.0], [0.0], [0.0]])  # node 1 committed
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    with use_pushforward_config(pf), use_biochem_runtime(rt):
        keep = _two_model_blend_mask(
            route="frontier", wall_mask=wall, log_state=log_state, edge_index=edge_index
        )
    # Growth zone = {0,1,2}; keep_wall = ~growth => only node 3 True
    assert keep.tolist() == [False, False, False, True]


def test_blend_mask_frontier_offwall_keeps_committed_wall():
    pf = PushforwardConfig(species_scope="mat", channels=(11,), mat_commit_thresh=0.1)
    rt = BiochemRuntimeConfig.from_kwargs({"two_model_frontier_hops": 0.5})
    wall = torch.tensor([True, True, False, False])
    log_state = torch.tensor([[0.0], [1.0], [0.0], [0.0]])  # node 1 committed
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    with use_pushforward_config(pf), use_biochem_runtime(rt):
        keep = _two_model_blend_mask(
            route="frontier_offwall", wall_mask=wall, log_state=log_state, edge_index=edge_index
        )
    assert keep.tolist() == [True, True, False, True]


def test_clear_offwall_cache_resets():
    clear_offwall_model_cache()
    # Idempotent; just ensure no exception and env can be empty.
    os.environ.pop("SPECIES_OFFWALL_MODEL_CKPT", None)
    clear_offwall_model_cache()
