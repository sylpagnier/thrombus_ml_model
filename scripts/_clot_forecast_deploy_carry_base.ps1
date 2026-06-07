# Deploy-faithful forecast: one-step phi + carried pred mu (no GT mu @ t_in after warm-up).
# Dot-source from go_clot_forecast_deploy_carry.ps1 / go_lane_a_deploy_carry.ps1

. (Join-Path $PSScriptRoot "_clot_forecast_r2a_plus_base.ps1")

$env:CLOT_FORECAST_MU_CARRY = "1"
$env:CLOT_FORECAST_MU_CARRY_DETACH = "1"
$env:CLOT_FORECAST_MU_INIT = "carreau"
$env:CLOT_FORECAST_MASK = "deploy_pred"
$env:CLOT_PHI_DGAMMA_SLICE = "0"
$env:CLOT_PHI_DGAMMA_FEATURE_TIME = "current"
$env:CLOT_PHI_TIME_STRIDE = "1"
$env:CLOT_PHI_TIME_STRIDE_AUTO = "0"
$env:CLOT_PHI_MESH_AUX_LAMBDA = "0.65"
$env:CLOT_PHI_MESH_BULK_LAMBDA = "0.22"
# GT -> carry curriculum (override per phase in launcher)
if (-not $env:CLOT_PHI_CARRY_GT_WARMUP_EPOCHS) { $env:CLOT_PHI_CARRY_GT_WARMUP_EPOCHS = "4" }
if (-not $env:CLOT_PHI_CARRY_GT_WARMUP_STEPS) { $env:CLOT_PHI_CARRY_GT_WARMUP_STEPS = "1" }
if (-not $env:CLOT_PHI_CARRY_GT_FADE_EPOCHS) { $env:CLOT_PHI_CARRY_GT_FADE_EPOCHS = "12" }
