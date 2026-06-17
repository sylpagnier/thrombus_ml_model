"""Biochem anchor discovery and per-vessel full-horizon caps."""

from __future__ import annotations

import os

import pytest

from src.core_physics.species_pushforward_continuous import (
    BIOCHEM_ANCHORS_6,
    deploy_eval_time_index,
    default_deploy_metric_times,
    discover_biochem_anchors,
    graph_last_time_index,
    legacy_capped_deploy_time_index,
    parse_biochem_train_anchors,
    pushforward_train_t0_per_vessel,
    resolve_train_t0_max,
    train_t0_max_for_n_times,
)
from src.utils.paths import get_project_root


def test_discover_biochem_anchors_includes_legacy_six():
    anchors = discover_biochem_anchors(get_project_root())
    for anc in BIOCHEM_ANCHORS_6:
        assert anc in anchors


def test_discover_biochem_anchors_sorted_unique():
    anchors = discover_biochem_anchors(get_project_root())
    assert anchors == sorted(set(anchors))
    assert len(anchors) >= len(BIOCHEM_ANCHORS_6)


def test_parse_biochem_train_anchors_all_on_disk():
    anchors = parse_biochem_train_anchors("", all_anchors=True, root=get_project_root())
    assert anchors == discover_biochem_anchors(get_project_root())


def test_train_t0_max_scales_with_graph_length():
    short = train_t0_max_for_n_times(54)
    long = train_t0_max_for_n_times(201)
    assert short == legacy_capped_deploy_time_index(54)
    assert long > short
    assert long == 132
    assert resolve_train_t0_max(201) == long


def test_default_deploy_metric_times_uses_per_graph_last():
    capped = default_deploy_metric_times(54)
    full = default_deploy_metric_times(201)
    assert capped[-1] == graph_last_time_index(54)
    assert full[-1] == graph_last_time_index(201)
    assert legacy_capped_deploy_time_index(201) in full


def test_deploy_eval_time_index_defaults_to_graph_last(monkeypatch):
    monkeypatch.delenv("SPECIES_CONTINUOUS_DEPLOY_EVAL_FULL", raising=False)
    monkeypatch.setenv("SPECIES_CONTINUOUS_DEPLOY_HORIZON", "0")
    assert deploy_eval_time_index(54) == 53
    assert deploy_eval_time_index(201) == 200


def test_resolve_train_t0_max_honors_global_override_when_per_vessel_off(monkeypatch):
    monkeypatch.setenv("SPECIES_PUSHFORWARD_TRAIN_T0_PER_VESSEL", "0")
    monkeypatch.setenv("SPECIES_PUSHFORWARD_TRAIN_T0_MAX", "22")
    assert resolve_train_t0_max(201) == 22


def test_pushforward_train_t0_per_vessel_default_on():
    old = os.environ.pop("SPECIES_PUSHFORWARD_TRAIN_T0_PER_VESSEL", None)
    try:
        assert pushforward_train_t0_per_vessel() is True
    finally:
        if old is not None:
            os.environ["SPECIES_PUSHFORWARD_TRAIN_T0_PER_VESSEL"] = old


def test_deploy_eval_full_timeline_last_index(monkeypatch):
    monkeypatch.setenv("SPECIES_CONTINUOUS_DEPLOY_EVAL_FULL", "1")
    assert deploy_eval_time_index(201) == 200
