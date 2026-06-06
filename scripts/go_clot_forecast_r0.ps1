# R0: COMSOL label sanity for clot forecast ladder (mu(t) -> mu(t+dt)).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_forecast_r0.ps1"

param(
    [string] $Anchors = "patient003,patient007,patient006",
    [int] $PairStride = 1,
    [double] $MuThrSi = 0.055
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")

Write-Host "[NEW] R0 clot forecast label sanity" -ForegroundColor Cyan
Write-Host "[i]  Checks GT mu growth + |dlog mu| on anchor pairs (no model)" -ForegroundColor DarkGray

$rc = Invoke-PythonRc @(
    (Join-Path $RepoRoot "scripts\check_clot_forecast_r0.py"),
    "--anchors", $Anchors,
    "--pair-stride", "$PairStride",
    "--mu-thr-si", "$MuThrSi"
)
if ($rc -ne 0) { exit $rc }
Write-Host "[OK]  R0 complete" -ForegroundColor Green
