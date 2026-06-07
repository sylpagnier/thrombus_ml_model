# Stage S0: static final shell — geometry + flow @ t=0 -> clot @ T_final (localization gate).

. (Join-Path $PSScriptRoot "_clot_deploy_binary_base.ps1")

$env:CLOT_FORECAST_PAIR_SCHEDULE = "static_final"
$env:CLOT_FORECAST_PAIR_STRIDE = "1"
