"""Tests for canonical SciML model nomenclature."""

from __future__ import annotations

from src.model_nomenclature import (
    BIOCHEM_DEPLOY_STACK,
    GINO_DEQ_KINE,
    PMGP_DEQ_KINE,
    SPECIES_GRAPHSAGE,
    is_legacy_kine_id,
    is_legacy_stack_id,
    pmgp_deq_feature_lines,
    resolve_model_id,
    stack_display_line,
)


def test_pmgp_deq_is_canonical_kine_id() -> None:
    assert PMGP_DEQ_KINE.id == "pmgp_deq_kine"
    assert PMGP_DEQ_KINE.acronym == "RGP-DEQ"
    assert GINO_DEQ_KINE is PMGP_DEQ_KINE
    assert len(pmgp_deq_feature_lines()) == 3


def test_resolve_stack_legacy_aliases() -> None:
    assert resolve_model_id("biochem_gnn") == BIOCHEM_DEPLOY_STACK.id
    assert resolve_model_id("biochem_deploy") == BIOCHEM_DEPLOY_STACK.id
    assert is_legacy_stack_id("biochem_gnn")
    assert not is_legacy_stack_id("biochem_deploy")


def test_resolve_kine_legacy_aliases() -> None:
    assert resolve_model_id("gino_deq_kine") == PMGP_DEQ_KINE.id
    assert resolve_model_id("gino-deq-kine") == PMGP_DEQ_KINE.id
    assert resolve_model_id("pmgp-deq-kine") == PMGP_DEQ_KINE.id
    assert resolve_model_id("PMGP-DEQ") == PMGP_DEQ_KINE.id
    assert resolve_model_id("RGP-DEQ") == PMGP_DEQ_KINE.id
    assert is_legacy_kine_id("gino_deq_kine")
    assert not is_legacy_kine_id("pmgp_deq_kine")
    assert PMGP_DEQ_KINE.code_class == "GINO_DEQ"


def test_resolve_species_legacy_aliases() -> None:
    assert resolve_model_id("species_gnn") == SPECIES_GRAPHSAGE.id


def test_stack_display_line_mentions_pmgp() -> None:
    line = stack_display_line()
    assert "pmgp_deq_kine" in line
    assert "RGP-DEQ" in line
