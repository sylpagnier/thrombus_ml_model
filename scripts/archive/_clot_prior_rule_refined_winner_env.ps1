# Refined sweep winner (2026-06 full grid): prior_p0.80 | flux_stag_top20 | tie_dx_hop
# Dot-source from go_clot_deploy_s0_rule_viz_refined.ps1

$env:CLOT_PHI_PRIOR_RULE_P = "0.80"
$env:CLOT_PHI_PRIOR_RULE_T0_STRIP = "0"
$env:CLOT_PHI_PRIOR_RULE_FLUX_STAG_TOP = "0.20"
$env:CLOT_PHI_PRIOR_RULE_TIE_BREAK = "1"
Remove-Item Env:CLOT_PHI_PRIOR_RULE_FLUX_DX_RAW_TOP -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_PHI_PRIOR_RULE_STAG_OFF_WALL_ADJ -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_PHI_PRIOR_RULE_FLUX_STREAM_TOP -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_PHI_PRIOR_RULE_NEG_DGAMMA_TOP -ErrorAction SilentlyContinue
