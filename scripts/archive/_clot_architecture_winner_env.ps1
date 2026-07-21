# Auto-promoted from C:\Users\pgssy\thrombus_ml_model\outputs\biochem\diagnostics\clot_rule_shear_risk_sweep.json
# winner rule: loc_prog_both_t20_s0_ndx25_inc40
# Dot-source AFTER _clot_prior_rule_winner_env.ps1

$env:CLOT_LOCALIZED_MODE = "wall_half"
$env:CLOT_LOCALIZED_SKIP_ARC = "0.00"
$env:CLOT_LOCALIZED_TOP_FRAC = "0.20"
$env:CLOT_TEMPORAL_END_FRAC = "0.22"
$env:CLOT_TEMPORAL_GLOBAL_ONSET = "0.40"
$env:CLOT_TEMPORAL_MIN_ONSET = "0.08"
$env:CLOT_TEMPORAL_ONSET_SPREAD = "0.55"
$env:CLOT_TEMPORAL_POWER = "1.5"
$env:CLOT_TEMPORAL_RULE_KIND = "progressive_topk"
$env:CLOT_TEMPORAL_RULE_NAME = "loc_prog_both_t20_s0_ndx25_inc40"
$env:CLOT_TEMPORAL_START_FRAC = "0.05"
Remove-Item Env:CLOT_LOCALIZED_ARC_BINS -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_LOCALIZED_SPECIES_WEIGHT -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_LOCALIZED_SPECIES_GT_Q -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_LOCALIZED_SPECIES_TIME -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_TEMPORAL_PROMOTION_BOOST -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_TEMPORAL_ACCUM_GAIN -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_TEMPORAL_ACCUM_THRESHOLD -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_TEMPORAL_ACCUM_SPLIT_WALL -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_TEMPORAL_ACCUM_SPLIT_LUMEN -ErrorAction SilentlyContinue
