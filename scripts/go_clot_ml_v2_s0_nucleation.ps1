# V0: nucleation mask audit + viz (no new ML training).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_ml_v2_s0_nucleation.ps1"
#   powershell ... -Anchor patient007 -MaxFrames 12

param(
    [string] $Anchor = "patient007",
    [string] $AnchorDir = "data/processed/graphs_biochem_anchors",
    [string] $Step1Ckpt = "outputs/biochem/sweep_clot_ml_physics_6h/step1_a35/clot_ml_step1_best.pth",
    [int] $MaxFrames = 10,
    [switch] $SkipViz
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")

$OutRoot = "outputs/biochem/clot_ml_ladder_v2/v0_nucleation"
$VizDir = "outputs/biochem/viz/clot_v2"
New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot $OutRoot) | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot $VizDir) | Out-Null

$env:CLOT_V2_NUCLEATION_HOPS = "1"
$env:CLOT_V2_CATALYTIC_HOPS = "1"
$env:CLOT_V2_CATALYTIC_BETA = "1.0"
$env:CLOT_ML_USE_MACRO_TAU = "1"
$env:CLOT_PHI_GROWTH_SEED = "gt"
$env:PYTHONUNBUFFERED = "1"

Write-Host "[NEW] V0 nucleation mask audit (all anchors)" -ForegroundColor Cyan
Invoke-PythonRcCheck -Label "v0 nucleation audit" -PyArgs @(
    "scripts/audit_clot_v2_nucleation_mask.py",
    "--anchor-dir", $AnchorDir,
    "--step1-ckpt", $Step1Ckpt,
    "--compare-pred",
    "--out", "$OutRoot/audit.json"
)

if (-not $SkipViz) {
    Write-Host ""
    Write-Host "[NEW] V0 nucleation viz anchor=$Anchor (GT + pred seed)" -ForegroundColor Cyan
    foreach ($seed in @("gt", "pred")) {
        Invoke-PythonRcCheck -Label "v0 viz $seed" -PyArgs @(
            "scripts/viz_clot_v2_nucleation_mask.py",
            "--anchor", $Anchor,
            "--anchor-dir", $AnchorDir,
            "--max-frames", "$MaxFrames",
            "--growth-seed", $seed
        )
    }
}

Write-Host ""
Write-Host "[OK] audit -> $OutRoot\audit.json" -ForegroundColor Green
Write-Host "[OK] PNG  -> $VizDir\v0_nucleation_${Anchor}_*.png" -ForegroundColor Green
