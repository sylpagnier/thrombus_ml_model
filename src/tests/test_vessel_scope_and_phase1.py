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


# --- s24 fixes ------------------------------------------------------------------------------

def test_latent_ablate_is_symmetric_across_train_and_eval():
    """Hard ablation must apply at BOTH train and eval.

    `latent_dropout` is stochastic and a no-op at eval; using it as an ablation would train and
    deploy on different inputs and confound the result.
    """
    from src.core_physics.species_pushforward_continuous import maybe_drop_latent

    class _M:
        kin_latent_dim = 256
        latent_dropout_p = 0.0

    x = torch.randn(40, 287)
    with use_pushforward_config(PushforwardConfig(species_scope="mat", latent_ablate=True)):
        for training in (True, False):
            out = maybe_drop_latent(x, _M(), training)
            assert float(out[:, :256].abs().max()) == 0.0, f"z_kin not ablated (training={training})"
            assert float(out[:, 256:].abs().max()) > 0.0, "non-latent block must be untouched"
    with use_pushforward_config(PushforwardConfig(species_scope="mat", latent_ablate=False)):
        assert float(maybe_drop_latent(x, _M(), True)[:, :256].abs().max()) > 0.0


def test_latent_ablation_reaches_the_DEPLOY_rollout_too():
    """The zeroing being correct is not enough -- the deploy path must call it.

    `test_latent_ablate_is_symmetric_across_train_and_eval` proves `maybe_drop_latent` zeroes
    z_kin at both train and eval, but it calls that helper directly. If the deploy rollout ever
    stopped routing through it, the leg would train on ablated input and deploy on full input --
    exactly the train/deploy asymmetry the hard ablation exists to avoid, and invisible in the
    config fingerprint. The chain is
    `deploy_species_rollout_series` -> `predict_continuous_step_delta` -> `maybe_drop_latent`.
    """
    src = Path("src/core_physics/species_pushforward_continuous.py").read_text(encoding="utf-8")

    def _body(fn: str) -> str:
        start = src.index(f"def {fn}(")
        nxt = src.index("\ndef ", start + 1)
        return src[start:nxt]

    assert "predict_continuous_step_delta(" in _body("deploy_species_rollout_series"), (
        "the deploy rollout no longer routes through predict_continuous_step_delta, so the "
        "z_kin ablation would not apply at deploy"
    )
    assert "maybe_drop_latent(" in _body("predict_continuous_step_delta")
    assert "maybe_drop_latent(" in _body("unroll_continuous_loss")

    # The hard-ablation branch must sit BEFORE the `if not training` short-circuit, or deploy
    # (training=False) would return early with z_kin intact.
    body = _body("maybe_drop_latent")
    assert body.index("continuous_latent_ablate()") < body.index("if not training:"), (
        "hard ablation must precede the eval short-circuit"
    )


def test_eval_bundle_binds_kin_latent_dim_so_ablation_is_not_a_no_op():
    """s26.10: `maybe_drop_latent` no-ops when `model.kin_latent_dim` is 0 or absent.

    Training binds it; the eval bundle loader did not, so `latent_ablate` was a silent no-op in
    the canonical eval and an ablated leg was scored on intact z_kin. The call chain being
    present is not enough -- the width has to be bound too, which is the same class of gap as
    the chain itself.
    """
    src = Path("src/core_physics/species_pushforward_continuous.py").read_text(encoding="utf-8")
    start = src.index("def load_continuous_bundle(")
    body = src[start:src.index("\ndef ", start + 1)]
    assert "model.kin_latent_dim" in body, (
        "the eval bundle must bind kin_latent_dim, or latent_ablate does nothing at eval"
    )

    # And the guard it feeds must actually gate on a positive width.
    drop = src[src.index("def maybe_drop_latent("):]
    drop = drop[:drop.index("\ndef ")]
    assert "ld > 0" in drop, "maybe_drop_latent no longer gates on the latent width"


def test_loss_accounting_is_gated_to_the_training_path():
    """s24 fix 2: three call sites feed the loss; only the main loop may be counted."""
    from src.core_physics.species_pushforward_continuous import (
        get_loss_terms, record_loss_term, reset_loss_terms, set_loss_accounting,
    )
    reset_loss_terms()
    set_loss_accounting(False)
    record_loss_term("t", 1.0)
    assert get_loss_terms() == {}, "terms recorded while accounting was off"
    set_loss_accounting(True)
    record_loss_term("t", 5.0)
    set_loss_accounting(False)
    assert get_loss_terms()["t"] == pytest.approx(5.0)


def test_final_state_value_scaling_defaults_off_and_is_readable():
    from src.core_physics.species_pushforward_continuous import (
        continuous_final_state_value_scaled,
    )
    with use_pushforward_config(PushforwardConfig(species_scope="mat")):
        assert continuous_final_state_value_scaled() is False
    with use_pushforward_config(
        PushforwardConfig(species_scope="mat", final_state_value_scaled=True)
    ):
        assert continuous_final_state_value_scaled() is True


def test_s24_legs_are_single_variable_against_each_other():
    from src.biochem_gnn.mat_growth_simple import get_mat_growth_config_kwargs as C

    a, b = C("WG_phase3a_closedloop"), C("WG_phase3b_zkin_ablate")
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert diff == {"latent_ablate"}, f"3a vs 3b must differ only by latent_ablate, got {diff}"
    # mass is out of the LOSS entirely; deploy_clot_score rewards precision, not mass
    for k in ("step_mass_penalty", "step_prec_fp_penalty",
              "final_mass_penalty", "final_prec_fp_penalty"):
        assert a[k] == 0.0, f"{k} must be 0 -- mass belongs in selection, not the loss"
    # the surrogate must match the metric's precision tilt, not F1's symmetry
    assert a["rolled_soft_f1_beta"] < 1.0
    assert a["closed_loop_init"] == pytest.approx(1.0)
    assert a["final_state_value_scaled"] is True


# --- s26 T1: the single-term ablation pair --------------------------------------------------

def test_t1_legs_are_exact_complements():
    """A and B must cut the objective along ONE seam and land on opposite sides of it.

    If they ever differ by a third knob the result stops being attributable, which is how
    v1-v6 were spent (9.12, 9.14).
    """
    from src.biochem_gnn.mat_growth_simple import (
        get_mat_growth_config_kwargs as C,
        get_mat_growth_runtime_kwargs as R,
    )

    a, b = C("WG_t1a_perstep_only"), C("WG_t1b_rolledf1_only")
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert diff == {"per_step_weight", "rolled_soft_f1_weight"}, (
        f"T1 A vs B must differ only along the per-step/rolled seam, got {diff}"
    )
    # Leg A: the per-step block alone.
    assert a["per_step_weight"] == 1.0
    assert a["rolled_soft_f1_weight"] == 0.0
    # Leg B: the rolled surrogate alone, at 3a's unchanged weight and beta.
    assert b["per_step_weight"] == 0.0
    assert b["rolled_soft_f1_weight"] == 120.0
    assert b["rolled_soft_f1_beta"] == C("WG_phase3a_closedloop")["rolled_soft_f1_beta"]

    # Everything NOT on the seam must be off in both, or "only" is a lie.
    for leg, k in ((a, k) for k in (
        "step_mass_penalty", "step_prec_fp_penalty", "final_mass_penalty",
        "final_prec_fp_penalty", "step_soft_f1_weight", "final_state_weight",
    )):
        assert leg[k] == 0.0, f"{k} must be 0 in leg A"
    for k in ("step_mass_penalty", "step_prec_fp_penalty", "final_mass_penalty",
              "final_prec_fp_penalty", "step_soft_f1_weight", "final_state_weight"):
        assert b[k] == 0.0, f"{k} must be 0 in leg B"
    # The three families s23.7 never recorded (per-step phi/mu, final phi/mu, speed-FP bleed).
    for leg, name in ((a, "A"), (b, "B")):
        assert leg["physics_readout"] is False, f"phi/mu readout must be off in leg {name}"
    for leg, name in ((R("WG_t1a_perstep_only"), "A"), (R("WG_t1b_rolledf1_only"), "B")):
        assert leg["speed_fp_weight"] == 0.0, f"speed-FP bleed must be off in leg {name}"


def test_per_step_weight_is_wired_and_scales_the_block():
    """`per_step_weight` must reach the summation point, not just the dataclass.

    The dominant failure mode here is a knob that resolves correctly and never multiplies
    anything -- nine dead constants on record.
    """
    from src.core_physics.species_pushforward_continuous import continuous_per_step_weight

    with use_pushforward_config(PushforwardConfig(species_scope="mat")):
        assert continuous_per_step_weight() == 1.0  # historical behaviour unchanged
    with use_pushforward_config(PushforwardConfig(species_scope="mat", per_step_weight=0.0)):
        assert continuous_per_step_weight() == 0.0

    src = Path("src/core_physics/species_pushforward_continuous.py").read_text(encoding="utf-8")
    body = src[src.index('record_loss_term("per_step_block", step_loss)'):]
    body = body[:body.index("fw = continuous_final_state_weight()")]
    assert "step_loss = step_loss * psw" in body, (
        "per_step_weight must multiply the assembled block, immediately after it is recorded"
    )


def test_fp_branch_selection_is_counted_and_is_empty_at_realistic_deltas():
    """s26 T4(c): does the growth Huber's FP branch select any node at all?

    `fp = (~gt_active) & (p_raw > max(fp_thresh=2e-5, delta_thresh=5e-6))`, while logged
    `val_pred_delta` runs 4.5e-8 to 5.2e-7 across every Phase-1 and Phase-3a epoch. s12.6.2
    inferred emptiness from a MEAN, which a mean cannot establish -- this asserts the
    selection directly, and the same counters now report it in situ from the training log.
    """
    from src.training.biochem_loss_policy import ActiveGrowthHuberLoss
    from src.core_physics.species_pushforward_continuous import (
        get_loss_terms,
        reset_loss_terms,
        set_loss_accounting,
    )

    n = 64
    loss = ActiveGrowthHuberLoss(
        delta_threshold=5e-6, beta=0.5, fp_weight=6.0, fp_threshold=2e-5,
        value_scale=1.5e5, underpred_weight=3.0,
    )
    band = torch.ones(n, dtype=torch.bool)
    gt = torch.zeros(n, 2)  # every node GT-inactive: the FP branch's whole domain

    # Predicted deltas at the magnitude the model actually produces.
    reset_loss_terms()
    set_loss_accounting(True)
    try:
        loss(torch.full((n, 2), 3e-7), gt, band)
        terms = get_loss_terms()
    finally:
        set_loss_accounting(False)
    assert terms["diag_fp_thresh"] == pytest.approx(2e-5)
    for ch in (0, 1):
        assert terms[f"diag_fp_nodes_ch{ch}"] == 0.0, (
            "at ~3e-7 the FP branch selects no nodes, so fp_weight multiplies an empty set"
        )

    # Two orders of magnitude up, the branch does fire -- the counter is not stuck at 0.
    reset_loss_terms()
    set_loss_accounting(True)
    try:
        loss(torch.full((n, 2), 5e-5), gt, band)
        terms = get_loss_terms()
    finally:
        set_loss_accounting(False)
    for ch in (0, 1):
        assert terms[f"diag_fp_nodes_ch{ch}"] == float(n)


# --- s26 T5: the loss_scale asymmetry -------------------------------------------------------

def test_loss_scale_reaches_rolled_terms_but_not_the_dual_head_block():
    """Pins the asymmetry itself, so a future edit cannot quietly move a term across it.

    `loss_scale`=0.1 multiplies every rolled term at its own site. The per-step path applies
    it in `continuous_delta_loss` (single head) and NOT in `dual_head_step_loss` -- and every
    cohort leg runs `dual_head=True`. So the rolled terms have always been /10 against the
    only term measured to move the model.
    """
    src = Path("src/core_physics/species_pushforward_continuous.py").read_text(encoding="utf-8")

    def _body(fn: str) -> str:
        start = src.index(f"def {fn}(")
        nxt = src.index("\ndef ", start + 1)
        return src[start:nxt]

    # The rolled-term sites scale; the dual-head per-step path does not.
    for fn in ("rolled_soft_f1_loss", "rolled_final_mass_fp_penalty", "continuous_delta_loss"):
        assert "continuous_loss_scale()" in _body(fn), f"{fn} lost its loss_scale"
    assert "continuous_loss_scale()" not in _body("dual_head_step_loss"), (
        "if dual_head_step_loss ever scales itself, s26 T5's premise changes and the "
        "surrogate weight must be re-derived again"
    )


def test_loss_scale_unified_is_off_by_default_and_cancels_the_scale_when_on():
    from src.core_physics.species_pushforward_continuous import rolled_scale_gain

    with use_pushforward_config(PushforwardConfig(species_scope="mat", loss_scale=0.1)):
        assert rolled_scale_gain() == 1.0, "default must be the historical behaviour exactly"
    with use_pushforward_config(
        PushforwardConfig(species_scope="mat", loss_scale=0.1, loss_scale_unified=True)
    ):
        assert rolled_scale_gain() == pytest.approx(10.0)
    # A leg that never set loss_scale must not be perturbed by turning the flag on.
    with use_pushforward_config(
        PushforwardConfig(species_scope="mat", loss_scale=1.0, loss_scale_unified=True)
    ):
        assert rolled_scale_gain() == pytest.approx(1.0)


def test_live_legs_have_not_silently_adopted_the_unified_scale():
    """T1's two legs must run on the historical scale, or they are not comparable with 3a."""
    from src.biochem_gnn.mat_growth_simple import get_mat_growth_config_kwargs as C

    for leg in ("WG_phase1_baseline", "WG_phase3a_closedloop",
                "WG_t1a_perstep_only", "WG_t1b_rolledf1_only"):
        assert C(leg).get("loss_scale_unified", False) is False, (
            f"{leg} must stay on the historical scale until T5 is run as its own leg"
        )
