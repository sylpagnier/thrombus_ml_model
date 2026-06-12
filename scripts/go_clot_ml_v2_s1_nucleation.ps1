# V1: frozen step1_a35 + nucleation projection (vs ceiling baseline).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_ml_v2_s1_nucleation.ps1"
#   powershell ... -Anchor patient007 -MaxFrames 14 -SkipViz

param(
    [string] $Anchor = "patient007",
    [string] $AnchorDir = "data/processed/graphs_biochem_anchors",
    [string] $Step0Json = "outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json",
    [string] $Step1Ckpt = "outputs/biochem/sweep_clot_ml_physics_6h/step1_a35/clot_ml_step1_best.pth",
    [string] $Val = "patient007",
    [int] $MaxFrames = 12,
    [switch] $SkipViz
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_clot_prior_rule_winner_env.ps1")
. (Join-Path $PSScriptRoot "_python_rc.ps1")

$OutRoot = "outputs/biochem/clot_ml_ladder_v2/v1_nucleation"
$VizDir = "outputs/biochem/viz/clot_v2"
New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot $OutRoot) | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot $VizDir) | Out-Null

$env:CLOT_V2_NUCLEATION = "1"
$env:CLOT_V2_NUCLEATION_HOPS = "1"
$env:CLOT_V2_CATALYTIC_HOPS = "1"
$env:CLOT_PHI_GROWTH_SEED = "gt"
$env:CLOT_ML_USE_MACRO_TAU = "1"
$env:PYTHONUNBUFFERED = "1"

Write-Host "[NEW] V1 LOAO eval ceiling vs nucleation (frozen step1)" -ForegroundColor Cyan
Invoke-PythonRcCheck -Label "v1 eval loao" -PyArgs @(
    "scripts/eval_clot_ml_v2_step1_nucleation.py",
    "--anchor-dir", $AnchorDir,
    "--step0-json", $Step0Json,
    "--step1-ckpt", $Step1Ckpt,
    "--val", $Val,
    "--out", "$OutRoot/eval_loao.json"
)

if (-not $SkipViz) {
    Write-Host ""
    Write-Host "[NEW] V1 viz $Anchor ceiling vs nucleation" -ForegroundColor Cyan
    Invoke-PythonRcCheck -Label "v1 viz" -PyArgs @(
        "scripts/viz_clot_ml_v2_step1_nucleation.py",
        "--anchor", $Anchor,
        "--anchor-dir", $AnchorDir,
        "--step0-json", $Step0Json,
        "--step1-ckpt", $Step1Ckpt,
        "--max-frames", "$MaxFrames"
    )
}

Write-Host ""
Write-Host "[OK] eval -> $OutRoot\eval_loao.json" -ForegroundColor Green
Write-Host "[OK] PNG  -> $VizDir\v1_step1_${Anchor}_ceiling_vs_nuc.png" -ForegroundColor Green
