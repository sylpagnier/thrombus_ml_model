# Temporal growing rule winner (set after sweep_clot_temporal_growth_rules.py).
# Dot-source after _clot_prior_rule_winner_env.ps1 for spatial base.
#
# growing winner (all anchors): ranked_onset_std -- matches static tfinal on p007
#   but turns on high-risk nodes earlier (time-varying phi for ML target).
# Override kind: progressive_topk | neighbor_ac | hop_growth | static_spatial

$env:CLOT_TEMPORAL_RULE_KIND = "ranked_onset"
$env:CLOT_TEMPORAL_RULE_NAME = "ranked_onset_std"
$env:CLOT_TEMPORAL_START_FRAC = "0.05"
$env:CLOT_TEMPORAL_END_FRAC = "0.22"
$env:CLOT_TEMPORAL_POWER = "1.5"
$env:CLOT_TEMPORAL_SEED_FRAC = "0.08"
$env:CLOT_TEMPORAL_ONSET_SPREAD = "0.55"
$env:CLOT_TEMPORAL_MIN_ONSET = "0.08"
$env:CLOT_TEMPORAL_GLOBAL_ONSET = "0.0"
$env:CLOT_TEMPORAL_NEIGHBOR_RISK_Q = "0.38"
$env:CLOT_TEMPORAL_RISK_FLOOR_Q = "0.42"
