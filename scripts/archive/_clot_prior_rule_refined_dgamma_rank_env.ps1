# Refined winner + rank pool = ceiling intersect dgamma adhesion slice @ t_in.
# Dot-source from go_clot_deploy_s0_rule_viz_dgamma_rank.ps1

. (Join-Path $PSScriptRoot "_clot_prior_rule_refined_winner_env.ps1")
$env:CLOT_PHI_PRIOR_RULE_RANK_DGAMMA_SLICE = "1"
