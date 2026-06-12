# V3: train band GNN growth rate + LOAO eval + viz (vs V1 step1).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_ml_v2_s3_growth_gnn.ps1"
#   powershell ... -Fast -SkipTrain   # eval/viz only

param(
    [string] $Anchor = "patient007",
    [string] $AnchorDir = "data/processed/graphs_biochem_anchors",
    [string] $Step0Json = "outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json",
    [string] $Step1Ckpt = "outputs/biochem/sweep_clot_ml_physics_6h/step1_a35/clot_ml_step1_best.pth",
    [string] $Val = "patient007",
    [int] $Epochs = 32,
    [int] $MaxFrames = 12,
    [switch] $Fast,
    [switch] $SkipTrain,
    [switch] $SkipViz,
    [switch] $NoTeacher
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_clot_prior_rule_winner_env.ps1")
. (Join-Path $PSScriptRoot "_python_rc.ps1")

$OutRoot = "outputs/biochem/clot_ml_ladder_v2/v3_growth_gnn"
$VizDir = "outputs/biochem/viz/clot_v2"
$V3Ckpt = "$OutRoot/clot_ml_v3_growth_gnn_best.pth"
New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot $OutRoot) | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot $VizDir) | Out-Null

if ($Fast) { $Epochs = 16 }

$env:CLOT_V2_NUCLEATION = "1"
$env:CLOT_V2_NUCLEATION_HOPS = "1"
$env:CLOT_V2_CATALYTIC_HOPS = "1"
$env:CLOT_V2_CATALYTIC_BETA = "1.0"
$env:CLOT_PHI_GROWTH_SEED = "gt"
$env:CLOT_ML_USE_MACRO_TAU = "1"
$env:CLOT_V3_RATE_SCALE = "5.0"
$env:PYTHONUNBUFFERED = "1"

if (-not $SkipTrain) {
    Write-Host "[NEW] V3 train band GNN growth rate (LOAO val=$Val epochs=$Epochs)" -ForegroundColor Cyan
    $trainArgs = @(
        "scripts/train_clot_ml_v2_growth_gnn.py",
        "--anchor-dir", $AnchorDir,
        "--step0-json", $Step0Json,
        "--val", $Val,
        "--epochs", "$Epochs",
        "--out-dir", $OutRoot
    )
    if ($NoTeacher) { $trainArgs += "--no-teacher" }
    Invoke-PythonRcCheck -Label "v3 train" -PyArgs $trainArgs
}

Write-Host ""
Write-Host "[NEW] V3 LOAO eval vs V1 step1" -ForegroundColor Cyan
Invoke-PythonRcCheck -Label "v3 eval" -PyArgs @(
    "scripts/eval_clot_ml_v2_growth_gnn.py",
    "--anchor-dir", $AnchorDir,
    "--step0-json", $Step0Json,
    "--v3-ckpt", $V3Ckpt,
    "--step1-ckpt", $Step1Ckpt,
    "--val", $Val,
    "--out", "$OutRoot/eval_loao.json"
)

if (-not $SkipViz) {
    Write-Host ""
    Write-Host "[NEW] V3 viz $Anchor vs V1" -ForegroundColor Cyan
    Invoke-PythonRcCheck -Label "v3 viz" -PyArgs @(
        "scripts/viz_clot_ml_v2_growth_gnn.py",
        "--anchor", $Anchor,
        "--anchor-dir", $AnchorDir,
        "--step0-json", $Step0Json,
        "--v3-ckpt", $V3Ckpt,
        "--step1-ckpt", $Step1Ckpt,
        "--max-frames", "$MaxFrames"
    )
}

Write-Host ""
Write-Host "[OK] ckpt -> $V3Ckpt" -ForegroundColor Green
Write-Host "[OK] eval -> $OutRoot\eval_loao.json" -ForegroundColor Green
Write-Host "[OK] PNG  -> $VizDir\v3_gnn_${Anchor}.png" -ForegroundColor Green
