# 8h clean generalization growth retrain (cold init, sealed 009/032).
# Usage:
#   .\scripts\go_generalization_8h.ps1 -Smoke
#   .\scripts\go_generalization_8h.ps1 -Fresh
param(
    [switch]$Smoke,
    [switch]$Fresh,
    [switch]$SkipTrain,
    [switch]$SkipEval,
    [double]$DeadlineHours = 8.0
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$pyArgs = @("-u", "scripts/run_generalization_8h.py", "--deadline-hours", "$DeadlineHours")
if ($Smoke) { $pyArgs += "--smoke" }
if ($Fresh) { $pyArgs += "--fresh" }
if ($SkipTrain) { $pyArgs += "--skip-train" }
if ($SkipEval) { $pyArgs += "--skip-eval" }

Write-Host "[i] GENERALIZATION 8H launcher"
Write-Host ("[i] " + ($pyArgs -join " "))
& python @pyArgs
exit $LASTEXITCODE
