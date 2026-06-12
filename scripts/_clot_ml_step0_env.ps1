# Auto-promoted from C:\Users\pgssy\thrombus_ml_model\outputs\biochem\clot_ml_ladder\step0_coef\best_coef.json
# Step 0 learned rule coefficients (pred GINO-DEQ kine)
# Dot-source AFTER _clot_prior_rule_winner_env.ps1

$env:CLOT_LOCALIZED_MODE = "wall_half"
$env:CLOT_LOCALIZED_NEG_DX_WEIGHT = "0.1000"
$env:CLOT_LOCALIZED_SKIP_ARC = "0.00"
$env:CLOT_LOCALIZED_TOP_FRAC = "0.1837"
$env:CLOT_SHEAR_W_LGRAD = "0.0000"
$env:CLOT_SHEAR_W_NEG_DX = "0.1000"
$env:CLOT_SHEAR_W_SEP = "0.0000"
$env:CLOT_SHEAR_W_STASIS = "0.0000"
$env:CLOT_TEMPORAL_END_FRAC = "0.2200"
$env:CLOT_TEMPORAL_GLOBAL_ONSET = "0.4000"
$env:CLOT_TEMPORAL_MIN_ONSET = "0.08"
$env:CLOT_TEMPORAL_ONSET_SPREAD = "0.55"
$env:CLOT_TEMPORAL_POWER = "1.5000"
$env:CLOT_TEMPORAL_PROMOTION_BOOST = "1.0000"
$env:CLOT_TEMPORAL_RULE_KIND = "progressive_topk"
$env:CLOT_TEMPORAL_RULE_NAME = "ml_step0_coef"
$env:CLOT_TEMPORAL_START_FRAC = "0.0500"
$env:CLOT_TEMPORAL_VEL_SOURCE = "kinematics"
Remove-Item Env:CLOT_LOCALIZED_SPECIES_GT_Q -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_LOCALIZED_SPECIES_WEIGHT -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_LOCALIZED_SPECIES_TIME -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_TEMPORAL_ACCUM_GAIN -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_TEMPORAL_ACCUM_THRESHOLD -ErrorAction SilentlyContinue
