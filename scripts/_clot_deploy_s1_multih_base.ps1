# Stage S1: multi-horizon from t=0 — (0, t_k) pairs, no carry, anchor times only.

. (Join-Path $PSScriptRoot "_clot_deploy_binary_base.ps1")

$env:CLOT_FORECAST_PAIR_SCHEDULE = "from_t0"
$env:CLOT_FORECAST_PAIR_STRIDE = "1"
