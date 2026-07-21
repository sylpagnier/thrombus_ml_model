# Localized temporal rule winner (segment-wise rank + skip inlet arc 15%).
# Sweep winner (deployable, non-oracle): loc_prog_half_top25_skip15
#   p007 tfinal F1 ~0.53 (vs 0.78 global), pred+ ~0.30 (vs 0.47 whole-wall)
# Dot-source AFTER _clot_prior_rule_winner_env.ps1

$env:CLOT_TEMPORAL_RULE_KIND = "progressive_topk"
$env:CLOT_TEMPORAL_RULE_NAME = "loc_prog_half_top25_skip15"
$env:CLOT_TEMPORAL_START_FRAC = "0.05"
$env:CLOT_TEMPORAL_END_FRAC = "0.22"
$env:CLOT_TEMPORAL_POWER = "1.5"
$env:CLOT_LOCALIZED_MODE = "wall_half"
$env:CLOT_LOCALIZED_TOP_FRAC = "0.25"
$env:CLOT_LOCALIZED_SKIP_ARC = "0.15"
Remove-Item Env:CLOT_LOCALIZED_RECESS -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_LOCALIZED_SPECIES_GT_Q -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_LOCALIZED_SPECIES_WEIGHT -ErrorAction SilentlyContinue
