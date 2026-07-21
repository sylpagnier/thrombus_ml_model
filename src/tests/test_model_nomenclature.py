"""Tests for canonical SciML model nomenclature (HemoRGP)."""

from __future__ import annotations

from src.model_nomenclature import (
    BIOCHEM_DEPLOY_STACK,
    BIOCHEM_GNN_STACK,
    GINO_DEQ_KINE,
    PMGP_DEQ_KINE,
    RGP_DEQ_KINE,
    SPECIES_GRAPHSAGE,
    is_legacy_kine_id,
    is_legacy_stack_id,
    pmgp_deq_feature_lines,
    resolve_model_id,
    rgp_deq_feature_lines,
    stack_display_line,
)


def test_rgp_deq_is_canonical_kine_id() -> None:
    assert RGP_DEQ_KINE.id == "rgp_deq_kine"
    assert RGP_DEQ_KINE.acronym == "RGP-DEQ"
    assert RGP_DEQ_KINE.code_class == "RGP_DEQ"
    assert GINO_DEQ_KINE is RGP_DEQ_KINE
    assert PMGP_DEQ_KINE is RGP_DEQ_KINE
    assert len(rgp_deq_feature_lines()) == 3
    assert pmgp_deq_feature_lines() == rgp_deq_feature_lines()


def test_resolve_stack_legacy_aliases() -> None:
    assert resolve_model_id("biochem_gnn") == BIOCHEM_GNN_STACK.id
    assert resolve_model_id("biochem_deploy") == BIOCHEM_GNN_STACK.id
    assert BIOCHEM_DEPLOY_STACK is BIOCHEM_GNN_STACK
    assert is_legacy_stack_id("biochem_deploy")
    assert not is_legacy_stack_id("biochem_gnn")


def test_resolve_kine_legacy_aliases() -> None:
    assert resolve_model_id("gino_deq_kine") == RGP_DEQ_KINE.id
    assert resolve_model_id("gino-deq-kine") == RGP_DEQ_KINE.id
    assert resolve_model_id("pmgp-deq-kine") == RGP_DEQ_KINE.id
    assert resolve_model_id("pmgp_deq_kine") == RGP_DEQ_KINE.id
    assert resolve_model_id("PMGP-DEQ") == RGP_DEQ_KINE.id
    assert resolve_model_id("RGP-DEQ") == RGP_DEQ_KINE.id
    assert resolve_model_id("rgp-deq-kine") == RGP_DEQ_KINE.id
    assert is_legacy_kine_id("gino_deq_kine")
    assert is_legacy_kine_id("pmgp_deq_kine")
    assert not is_legacy_kine_id("rgp_deq_kine")


def test_resolve_local_corrector_aliases() -> None:
    from src.model_nomenclature import LOCAL_KINEMATIC_CORRECTOR

    assert resolve_model_id("local_corrector") == LOCAL_KINEMATIC_CORRECTOR.id
    assert resolve_model_id("local_kinematic_corrector") == LOCAL_KINEMATIC_CORRECTOR.id
    assert LOCAL_KINEMATIC_CORRECTOR.code_class == "LocalKinematicCorrector"


def test_stack_display_line_mentions_rgp() -> None:
    line = stack_display_line()
    assert "rgp_deq_kine" in line
    assert "RGP-DEQ" in line
    assert "biochem_gnn" in line
