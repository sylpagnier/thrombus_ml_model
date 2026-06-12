# V3.1: spatial growth loss + soft commits (init from V3 ckpt).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_ml_v2_s31_growth_gnn.ps1"
#   powershell ... -Fast -SkipTrain

param(
    [string] $Anchor = "patient007",
    [string] $AnchorDir = "data/processed/graphs_biochem_anchors",
    [string] $Step0Json = "outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json",
    [string] $InitCkpt = "outputs/biochem/clot_ml_ladder_v2/v3_growth_gnn/clot_ml_v3_growth_gnn_best.pth",
    [string] $Val = "patient007",
    [int] $Epochs = 32,
    [int] $MaxFrames = 12,
    [switch] $Fast,
    [switch] $SkipTrain,
    [switch] $SkipViz,
    [switch] $NoInit
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_clot_prior_rule_winner_env.ps1")
. (Join-Path $PSScriptRoot "_python_rc.ps1")

$OutRoot = "outputs/biochem/clot_ml_ladder_v2/v31_growth_gnn"
$VizDir = "outputs/biochem/viz/clot_v2"
$V31Ckpt = "$OutRoot/clot_ml_v31_growth_gnn_best.pth"
New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot $OutRoot) | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot $VizDir) | Out-Null

if ($Fast) { $Epochs = 16 }

$env:CLOT_V31_RECIPE = "1"
$env:CLOT_V2_NUCLEATION = "1"
$env:CLOT_V3_RATE_SCALE = "2.5"
$env:CLOT_V31_MAX_STEP_DELTA = "0.10"
$env:CLOT_ML_USE_MACRO_TAU = "1"
$env:CLOT_V31_HARD_COMMIT = "1"
$env:PYTHONUNBUFFERED = "1"

if (-not $SkipTrain) {
    Write-Host "[NEW] V3.1 train spatial growth loss (init=$InitCkpt epochs=$Epochs)" -ForegroundColor Cyan
    $trainArgs = @(
        "scripts/train_clot_ml_v31_growth_gnn.py",
        "--anchor-dir", $AnchorDir,
        "--step0-json", $Step0Json,
        "--init-ckpt", $InitCkpt,
        "--val", $Val,
        "--epochs", "$Epochs",
        "--out-dir", $OutRoot
    )
    if ($NoInit) { $trainArgs += "--no-init" }
    Invoke-PythonRcCheck -Label "v31 train" -PyArgs $trainArgs
}

Write-Host ""
Write-Host "[NEW] V3.1 LOAO eval vs V3/V1" -ForegroundColor Cyan
Invoke-PythonRcCheck -Label "v31 eval" -PyArgs @(
    "scripts/eval_clot_ml_v31_growth_gnn.py",
    "--anchor-dir", $AnchorDir,
    "--step0-json", $Step0Json,
    "--v31-ckpt", $V31Ckpt,
    "--val", $Val,
    "--out", "$OutRoot/eval_loao.json"
)

if (-not $SkipViz) {
    Write-Host ""
    Write-Host "[NEW] V3.1 viz $Anchor vs V1" -ForegroundColor Cyan
    Invoke-PythonRcCheck -Label "v31 viz" -PyArgs @(
        "scripts/viz_clot_ml_v2_growth_gnn.py",
        "--anchor", $Anchor,
        "--anchor-dir", $AnchorDir,
        "--step0-json", $Step0Json,
        "--v31",
        "--v31-ckpt", $V31Ckpt,
        "--max-frames", "$MaxFrames"
    )
}

Write-Host ""
Write-Host "[OK] ckpt -> $V31Ckpt" -ForegroundColor Green
Write-Host "[OK] eval -> $OutRoot\eval_loao.json" -ForegroundColor Green
Write-Host "[OK] PNG  -> $VizDir\v31_gnn_${Anchor}.png" -ForegroundColor Green
