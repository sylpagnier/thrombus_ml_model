# Overnight clot-trigger unblock: T5 deploy teacher + T1 honest trigger + T0 physics sweep.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_trigger_overnight.ps1"
#   powershell ... -Fresh
#   powershell ... -SkipT5 -SkipT1   # resume after partial T5/T1

param(
    [switch] $Fresh,
    [switch] $SkipT5,
    [switch] $SkipT1,
    [switch] $SkipT0
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$env:PYTHONUNBUFFERED = "1"

if (-not $SkipT5) {
    $t5Params = @{ SkipEval = $true }
    if ($Fresh) { $t5Params.Fresh = $true }
    & (Join-Path $RepoRoot "scripts\go_clot_trigger_t5.ps1") @t5Params
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (-not $SkipT1) {
    $t1Params = @{ Fast = $true }
    if ($Fresh) { $t1Params.Fresh = $true }
    & (Join-Path $RepoRoot "scripts\go_clot_trigger_t1.ps1") @t1Params
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (-not $SkipT0) {
    & (Join-Path $RepoRoot "scripts\go_clot_trigger_t0_physics_sweep.ps1")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host ""
Write-Host "[OK] overnight chain done" -ForegroundColor Green
Write-Host "[i]  tomorrow: go_clot_trigger_t2.ps1, go_clot_trigger_t3.ps1" -ForegroundColor DarkGray
