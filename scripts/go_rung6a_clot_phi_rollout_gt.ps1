# Rung 6a / forecast R2 alias (GT velocity rollout). Prefer go_clot_forecast_r2.ps1.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_rung6a_clot_phi_rollout_gt.ps1" -Fresh

param(
    [switch] $Fresh,
    [switch] $NoInitFromD,
    [int] $Epochs = 60
)

$ErrorActionPreference = "Stop"
$ScriptDir = $PSScriptRoot
$fwd = @("-File", (Join-Path $ScriptDir "go_clot_forecast_r2.ps1"))
if ($Fresh) { $fwd += "-Fresh" }
if ($NoInitFromD) { $fwd += "-NoInitFromD" }
if ($Epochs -ne 60) { $fwd += "-Epochs"; $fwd += "$Epochs" }
powershell -NoProfile -ExecutionPolicy Bypass @fwd
exit $LASTEXITCODE
