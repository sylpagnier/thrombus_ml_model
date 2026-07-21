# Deploy rule from multistep probe (2026-06): refined winner + skip inlet q25.
# prior_p0.80 | flux_stag_top20 | tie_dx_hop | skip_inlet_q25
# Mean band F1 ~0.54, pred+ ~0.60 (avoids whole-wall flags vs refined-only ~0.83 pred+).
# Dot-source from go_clot_deploy_ladder.ps1 and go_clot_deploy_s0_rule_*.ps1.

$env:CLOT_PHI_PRIOR_RULE_P = "0.80"
$env:CLOT_PHI_PRIOR_RULE_T0_STRIP = "0"
$env:CLOT_PHI_PRIOR_RULE_FLUX_STAG_TOP = "0.20"
$env:CLOT_PHI_PRIOR_RULE_TIE_BREAK = "1"
$env:CLOT_PHI_PRIOR_RULE_SKIP_INLET_Q = "0.25"
Remove-Item Env:CLOT_PHI_PRIOR_RULE_RANK_SDF_MAX -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_PHI_PRIOR_RULE_FLUX_DX_RAW_TOP -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_PHI_PRIOR_RULE_STAG_OFF_WALL_ADJ -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_PHI_PRIOR_RULE_FLUX_STREAM_TOP -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_PHI_PRIOR_RULE_NEG_DGAMMA_TOP -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_PHI_PRIOR_RULE_ON_WALL -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_PHI_PRIOR_RULE_MAX_HOP_WALL -ErrorAction SilentlyContinue
