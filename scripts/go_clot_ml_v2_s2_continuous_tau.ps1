# V2: continuous macro tau + extrap indices on V1 nucleation shell (frozen step1).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_ml_v2_s2_continuous_tau.ps1"
#   powershell ... -SimEndScale 5.0 -Anchor patient007 -MaxFrames 16

param(
    [string] $Anchor = "patient007",
    [string] $AnchorDir = "data/processed/graphs_biochem_anchors",
    [string] $Step0Json = "outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json",
    [string] $Step1Ckpt = "outputs/biochem/sweep_clot_ml_physics_6h/step1_a35/clot_ml_step1_best.pth",
    [string] $Val = "patient007",
    [double] $SimEndScale = 2.0,
    [double] $VizSimEndScale = 5.0,
    [int] $MaxFrames = 14,
    [switch] $SkipViz
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_clot_prior_rule_winner_env.ps1")
. (Join-Path $PSScriptRoot "_python_rc.ps1")

$OutRoot = "outputs/biochem/clot_ml_ladder_v2/v2_continuous_tau"
$VizDir = "outputs/biochem/viz/clot_v2"
New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot $OutRoot) | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot $VizDir) | Out-Null

$env:CLOT_V2_NUCLEATION = "1"
$env:CLOT_V2_NUCLEATION_HOPS = "1"
$env:CLOT_V2_CATALYTIC_HOPS = "1"
$env:CLOT_PHI_GROWTH_SEED = "gt"
$env:CLOT_ML_USE_MACRO_TAU = "1"
$env:CLOT_ML_CONTINUOUS_EXTRAP = "1"
$env:CLOT_ML_SIM_END_SCALE = "$SimEndScale"
$env:PYTHONUNBUFFERED = "1"

Write-Host "[NEW] V2 LOAO eval V1 vs continuous tau (frozen step1)" -ForegroundColor Cyan
Invoke-PythonRcCheck -Label "v2 eval loao" -PyArgs @(
    "scripts/eval_clot_ml_v2_continuous_tau.py",
    "--anchor-dir", $AnchorDir,
    "--step0-json", $Step0Json,
    "--step1-ckpt", $Step1Ckpt,
    "--val", $Val,
    "--sim-end-scale", "$SimEndScale",
    "--out", "$OutRoot/eval_loao.json"
)

if (-not $SkipViz) {
    Write-Host ""
    Write-Host "[NEW] V2 viz $Anchor scale=$VizSimEndScale" -ForegroundColor Cyan
    Invoke-PythonRcCheck -Label "v2 viz" -PyArgs @(
        "scripts/viz_clot_ml_v2_continuous_tau.py",
        "--anchor", $Anchor,
        "--anchor-dir", $AnchorDir,
        "--step0-json", $Step0Json,
        "--step1-ckpt", $Step1Ckpt,
        "--sim-end-scale", "$VizSimEndScale",
        "--max-frames", "$MaxFrames"
    )
}

Write-Host ""
Write-Host "[OK] eval -> $OutRoot\eval_loao.json" -ForegroundColor Green
Write-Host "[OK] PNG  -> $VizDir\v2_tau_${Anchor}_s$([int]$VizSimEndScale).png" -ForegroundColor Green
