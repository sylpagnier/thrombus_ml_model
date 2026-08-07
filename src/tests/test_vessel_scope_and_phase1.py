"""Guards for the Phase-1 wiring (WALL_MODEL_PLAN.md 21-22).

Every assertion here corresponds to a way the wiring could silently become a no-op. That is
this project's dominant failure mode: v4 and v5 both looked like real experiments and changed
nothing (12.3), and the 20.0 test leak and 20.1 scoring split were the same shape.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from src.architecture.pushforward_config import PushforwardConfig, use_pushforward_config
from src.core_physics.vessel_scope import (
    prepare_vessel_data,
    prior_source_cache_tag,
    resolve_vessel_mat_max,
)

TRAINER = Path("src/training/train_species_pushforward_continuous.py")
EVALER = Path("scripts/eval_mat_growth_simple.py")


# --- vessel_scope primitives ----------------------------------------------------------------

def test_prior_cache_tag_separates_prior_sources():
    """A cache built with leaked priors must not be reusable by an analytic run.

    The DEQ latent is baked into the cached pack, so reusing it would preserve the leak while
    appearing to have switched (16.1c).
    """
    assert prior_source_cache_tag("stored") == ""          # pre-existing caches stay valid
    assert prior_source_cache_tag("analytic") == "_prior-analytic"
    assert prior_source_cache_tag("zero") == "_prior-zero"
    assert prior_source_cache_tag("analytic") != prior_source_cache_tag("zero")


def test_vessel_mat_max_returns_none_for_clot_free():
    """None, not 0.0 -- a zero scale would label every node committed."""
    class _D:
        y = None
    assert resolve_vessel_mat_max(_D()) is None

    class _E:
        y = torch.zeros(4, 10, 16)
    assert resolve_vessel_mat_max(_E()) is None


def test_vessel_mat_max_reads_the_MAT_COLUMN_OF_Y_not_the_species_index():
    """Regression: MAT_CHANNEL is species-block-relative (11), not a column of `y` (15).

    Indexing `y` with MAT_CHANNEL directly reads FG_log1p_nd, whose peak is ~160x Mat's, which
    silently inflates every relative label threshold. The original version of this test seeded
    and read using the SAME wrong index, so it passed while the function was broken -- hence
    the explicit column arithmetic here.
    """
    from src.training.biochem_species_scope import MAT_CHANNEL
    from src.utils import species_channels as sc

    mat_col = int(sc.SPECIES_BLOCK.start) + int(MAT_CHANNEL)
    assert mat_col == 15, "y layout changed; revisit resolve_vessel_mat_max"

    class _D:
        y = torch.zeros(5, 10, 16)
    _D.y[3, 7, mat_col] = 4.2e-3
    _D.y[2, 4, int(MAT_CHANNEL)] = 0.69      # decoy in the FG column, ~160x larger
    assert resolve_vessel_mat_max(_D()) == pytest.approx(4.2e-3)


def test_vessel_mat_max_matches_real_packs():
    """Against real data, not synthetic -- the layout assumption must hold on disk."""
    path = Path("data/processed/graphs_biochem_anchors/patient041.pt")
    if not path.exists():
        pytest.skip("anchor pack not available")
    d = torch.load(path, map_location="cpu", weights_only=False)
    names = d.y_channel_names.split(",")
    col = names.index("Mat_log1p_nd")
    assert resolve_vessel_mat_max(d) == pytest.approx(float(d.y[:, :, col].max()))


def test_prepare_vessel_data_is_a_noop_for_stored():
    class _D:
        y = torch.zeros(2, 3, 16)
    d = _D()
    out, _ = prepare_vessel_data(d, prior_source="stored")
    assert out is d, "stored must not copy or rewrite"


# --- trainer wiring -------------------------------------------------------------------------

def test_trainer_applies_priors_before_the_kinematics_solve():
    """Order matters: the DEQ consumes UV_PRIOR/MU_PRIOR, so rewriting after the solve is a
    no-op that still looks like a change."""
    src = TRAINER.read_text(encoding="utf-8")
    i_prep = src.find("prepare_vessel_data(")
    i_solve = src.find("predict_kinematics_and_latent(")
    assert i_prep > 0, "trainer never applies the prior source"
    assert i_solve > 0
    assert i_prep < i_solve, (
        "trainer rewrites priors AFTER the kinematics solve; z_kin would stay conditioned on "
        "the leaked CFD field and prior_source would be a silent no-op"
    )


def test_trainer_cache_key_includes_prior_source():
    src = TRAINER.read_text(encoding="utf-8")
    assert "prior_source_cache_tag()" in src, (
        "pack cache key omits the prior source; an analytic run would reuse leaked packs"
    )


def test_pack_carries_its_label_scale():
    src = TRAINER.read_text(encoding="utf-8")
    assert '"mat_max": resolve_vessel_mat_max(data)' in src


@pytest.mark.parametrize("scope_expr", [
    'use_vessel_mat_max(pack.get("mat_max"))',      # main training loop
    'use_vessel_mat_max(vpack.get("mat_max"))',     # deploy-horizon aux
    'use_vessel_mat_max(val_pack.get("mat_max"))',  # reported deploy metrics
])
def test_every_loss_and_metric_site_binds_the_vessel_scale(scope_expr):
    """If any site forgets, mat_label_thresh() silently falls back to absolute there."""
    assert scope_expr in TRAINER.read_text(encoding="utf-8"), f"missing binding: {scope_expr}"


def test_eval_script_uses_the_same_vessel_scope():
    src = EVALER.read_text(encoding="utf-8")
    assert "prepare_vessel_data(" in src
    assert "use_vessel_mat_max(" in src


# --- the Phase 1 leg ------------------------------------------------------------------------

def test_phase1_leg_switches_on_the_whole_foundation():
    from src.biochem_gnn.mat_growth_simple import (
        get_mat_growth_config_kwargs as C,
        get_mat_growth_runtime_kwargs as R,
    )
    c, r = C("WG_phase1_baseline"), R("WG_phase1_baseline")
    assert c["mat_label_thresh_mode"] == "rel_max"
    assert c["mat_label_rel_frac"] == pytest.approx(0.10)
    assert c["rolled_soft_k_relative"] is True
    assert r["prior_source"] == "analytic"
    # selection window must bracket the MEASURED optimum of 3.04x (20.2)
    assert r["select_mass_hard_min"] < 3.04 < r["select_mass_hard_max"]
    assert c["final_mass_target"] == pytest.approx(3.0)


def test_phase1_trains_on_cohort_v2_and_never_on_the_sealed_set():
    from src.biochem_gnn.mat_growth_simple import (
        WALL_COHORT_V2_GENERALIZATION,
        WALL_COHORT_V2_TRAIN,
    )
    assert len(WALL_COHORT_V2_TRAIN) == 26
    assert len(WALL_COHORT_V2_GENERALIZATION) == 8
    assert not set(WALL_COHORT_V2_TRAIN) & set(WALL_COHORT_V2_GENERALIZATION)
    # patient002/023 are the project's existing data-quality exclusions; the launcher forbids
    # them, so a cohort containing one would fail at launch rather than at analysis time.
    assert "patient002" not in WALL_COHORT_V2_TRAIN
    assert "patient023" not in WALL_COHORT_V2_TRAIN


def test_historical_legs_are_untouched_by_phase1_defaults():
    """v3-v10 must stay bit-reproducible: absolute labels, stored priors, old mass window."""
    from src.biochem_gnn.mat_growth_simple import (
        get_mat_growth_config_kwargs as C,
        get_mat_growth_runtime_kwargs as R,
    )
    for leg in ("WG_stenosis_subcohort_ft_v3", "WG_stenosis_subcohort_ft_v10"):
        c, r = C(leg), R(leg)
        assert c.get("mat_label_thresh_mode") in (None, "absolute")
        assert r.get("prior_source") in (None, "stored")
        assert r["select_mass_hard_max"] == pytest.approx(1.5)


def test_label_and_prediction_thresholds_stay_separate_in_source():
    """The GT side may be vessel-relative; the prediction side must never be."""
    from src.core_physics import species_pushforward_continuous as spc

    for fn in (spc.rolled_soft_f1_loss, spc.rolled_final_mass_fp_penalty):
        src = inspect.getsource(fn)
        assert "mat_label_thresh()" in src, f"{fn.__name__} does not use the label threshold"
    commit_src = inspect.getsource(spc.continuous_mat_commit_thresh)
    assert "_VESSEL_MAT_MAX" not in commit_src and "mat_label" not in commit_src, (
        "prediction-side threshold must not consult the vessel max -- that is a deploy leak"
    )
