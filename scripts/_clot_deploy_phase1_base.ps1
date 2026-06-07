# CAVO Stage G1: rolling one-step pairs + optional mu carry warm-up.
# Dot-source from go_clot_deploy_phase1.ps1

. (Join-Path $PSScriptRoot "_clot_deploy_binary_base.ps1")

$env:CLOT_FORECAST_PAIR_SCHEDULE = "rolling"
$env:CLOT_FORECAST_PAIR_STRIDE = "1"
$env:CLOT_FORECAST_MASK = "input"

# Mu carry (Tier A): GT @ t_in during warm-up only; fade via launcher.
$env:CLOT_FORECAST_MU_CARRY = "1"
$env:CLOT_FORECAST_MU_CARRY_DETACH = "1"
$env:CLOT_FORECAST_MU_INIT = "carreau"
$env:CLOT_FORECAST_INPUT_MU = "1"
if (-not $env:CLOT_PHI_CARRY_GT_WARMUP_EPOCHS) { $env:CLOT_PHI_CARRY_GT_WARMUP_EPOCHS = "4" }
if (-not $env:CLOT_PHI_CARRY_GT_WARMUP_STEPS) { $env:CLOT_PHI_CARRY_GT_WARMUP_STEPS = "2" }
if (-not $env:CLOT_PHI_CARRY_GT_FADE_EPOCHS) { $env:CLOT_PHI_CARRY_GT_FADE_EPOCHS = "8" }