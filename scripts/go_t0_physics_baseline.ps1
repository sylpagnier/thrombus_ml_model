# T0 baseline physics: diagnose mu/gamma/phi + field viz (COMSOL spf.mu alignment).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_t0_physics_baseline.ps1"
#   powershell ... -Anchor patient007 -VizOnly
#   powershell ... -EvalOracle   # also run deploy T0 F1 eval

param(
    [string] $Anchor = "patient007",
    [string] $Times = "0,17,35,53",
    [string] $DiagOut = "outputs/biochem/clot_trigger/t0_physics_baseline_diag.json",
    [string] $VizOut = "",
    [switch] $SkipDiag,
    [switch] $SkipViz,
    [switch] $VizOnly,
    [switch] $EvalOracle
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")

$env:CLOT_PHI_PHYSICS_MU_BASE = "comsol_carreau"
$env:CLOT_PHI_PHYSICS_GAMMA_MODE = "max"
$env:CLOT_PHI_PHYSICS_MU_RATIO_MAX = "4"
$env:CLOT_PHI_PHYSICS_SUBTRACT_T0_MU = "1"
$env:PYTHONUNBUFFERED = "1"

if (-not $VizOnly -and -not $SkipDiag) {
    $srTxt = Join-Path $RepoRoot "data/processed/cfd_results_biochem/${Anchor}_sr.txt"
    if (Test-Path $srTxt) {
        Write-Host "[NEW] Build COMSOL spf.sr sidecar ($Anchor)" -ForegroundColor Cyan
        Invoke-PythonRcCheck -Label "build comsol sr sidecar" -PyArgs @(
            "scripts/build_comsol_sr_sidecar.py",
            "--anchor", $Anchor
        )
    }
    Write-Host "[NEW] T0 physics baseline diagnostic ($Anchor)" -ForegroundColor Cyan
    Invoke-PythonRcCheck -Label "t0 physics baseline diag" -PyArgs @(
        "scripts/diagnose_t0_physics_baseline.py",
        "--anchor", $Anchor,
        "--times", $Times,
        "--out", $DiagOut
    )
}

if (-not $SkipViz) {
    Write-Host "[NEW] T0 physics baseline viz ($Anchor)" -ForegroundColor Cyan
    $vizArgs = @(
        "scripts/viz_t0_physics_baseline.py",
        "--anchor", $Anchor,
        "--max-frames", "6"
    )
    if ($VizOut) { $vizArgs += @("--out", $VizOut) }
    Invoke-PythonRcCheck -Label "t0 physics baseline viz" -PyArgs $vizArgs
}

if ($EvalOracle -and -not $VizOnly) {
    Write-Host "[NEW] T0 deploy oracle eval" -ForegroundColor Cyan
    & (Join-Path $PSScriptRoot "go_clot_trigger_t0_oracle.ps1") -Val $Anchor -VizAnchor $Anchor
}

Write-Host "[OK] baseline -> $DiagOut and outputs/biochem/viz/clot_trigger/t0_physics_$Anchor.png" -ForegroundColor Green
Write-Host "[i] Optional COMSOL gamma sidecar: data/processed/cfd_results_biochem_diag/${Anchor}_gammat.pt" -ForegroundColor DarkGray
