# T0 GT baseline: GT u,v,p + species -> clot (no spf.mu / spf.sr sidecar for prediction).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_t0_baseline_gt.ps1"
#   powershell ... -Anchor patient007 -NucleationRow

param(
    [string] $Anchor = "patient007",
    [string] $Times = "",
    [string] $SweepOut = "outputs/biochem/clot_trigger/t0_clot_predictor_sweep.json",
    [string] $VizOut = "",
    [switch] $SkipSweep,
    [switch] $SkipViz,
    [switch] $VizOnly,
    [switch] $NucleationRow,
    [switch] $NoNucleationRow
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")

$env:PYTHONUNBUFFERED = "1"

if (-not $VizOnly -and -not $SkipSweep) {
    Write-Host "[NEW] T0 clot predictor sweep ($Anchor) -- no spf.sr sidecar" -ForegroundColor Cyan
    $sweepArgs = @(
        "scripts/diagnose_t0_clot_predictor.py",
        "--anchor", $Anchor,
        "--out", $SweepOut
    )
    if ($Times) { $sweepArgs += @("--times", $Times) }
    Invoke-PythonRcCheck -Label "t0 clot predictor sweep" -PyArgs $sweepArgs
}

if (-not $SkipViz) {
    Write-Host "[NEW] T0 GT baseline viz ($Anchor)" -ForegroundColor Cyan
    $vizArgs = @(
        "scripts/viz_t0_baseline_gt.py",
        "--anchor", $Anchor,
        "--sweep-json", $SweepOut
    )
    if ($VizOut) { $vizArgs += @("--out", $VizOut) }
    if ($NucleationRow -or -not $NoNucleationRow) { $vizArgs += "--nucleation-row" }
    Invoke-PythonRcCheck -Label "t0 baseline gt viz" -PyArgs $vizArgs
}

Write-Host "[OK] sweep=$SweepOut viz=outputs/biochem/viz/clot_trigger/t0_baseline_gt_$Anchor.png" -ForegroundColor Green
