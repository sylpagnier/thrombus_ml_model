# T0 mu physics: GT flow + species -> viscosity (no GT mu input).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_t0_mu_physics.ps1"
#   powershell ... -Anchor patient007 -SweepGamma

param(
    [string] $Anchor = "patient007",
    [string] $Times = "",
    [string] $GammaMode = "auto",
    [string] $Out = "outputs/biochem/clot_trigger/t0_mu_physics.json",
    [string] $VizOut = "",
    [switch] $SkipViz,
    [switch] $SweepGamma,
    [switch] $VizOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")

$env:PYTHONUNBUFFERED = "1"

$srTxt = Join-Path $RepoRoot "data/processed/cfd_results_biochem/${Anchor}_sr.txt"
if (Test-Path $srTxt) {
    Write-Host "[NEW] Build COMSOL spf.sr sidecar ($Anchor)" -ForegroundColor Cyan
    Invoke-PythonRcCheck -Label "build comsol sr" -PyArgs @(
        "scripts/build_comsol_sr_sidecar.py",
        "--anchor", $Anchor
    )
}

if (-not $VizOnly) {
    Write-Host "[NEW] T0 mu physics eval ($Anchor)" -ForegroundColor Cyan
    $evalArgs = @(
        "scripts/eval_t0_mu_physics.py",
        "--anchor", $Anchor,
        "--out", $Out,
        "--gamma-mode", $GammaMode
    )
    if ($Times) { $evalArgs += @("--times", $Times) }
    if ($SweepGamma) { $evalArgs += "--sweep-gamma" }
    Invoke-PythonRcCheck -Label "t0 mu eval" -PyArgs $evalArgs
}

if (-not $SkipViz) {
    Write-Host "[NEW] T0 mu physics viz ($Anchor)" -ForegroundColor Cyan
    $vizArgs = @(
        "scripts/viz_t0_mu_physics.py",
        "--anchor", $Anchor,
        "--gamma-mode", $GammaMode
    )
    if ($VizOut) { $vizArgs += @("--out", $VizOut) }
    Invoke-PythonRcCheck -Label "t0 mu viz" -PyArgs $vizArgs
}

Write-Host "[OK] T0 mu -> $Out and outputs/biochem/viz/clot_trigger/t0_mu_$Anchor.png" -ForegroundColor Green
