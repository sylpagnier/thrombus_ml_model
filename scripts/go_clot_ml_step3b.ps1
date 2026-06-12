# Step 3b: continuous-time extrap finetune (init step1_a35) + 5x phi viz.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_ml_step3b.ps1"
#   powershell ... -SkipTrain -SimEndScale 5.0 -MaxFrames 20

param(
    [string] $Anchor = "patient007",
    [string] $AnchorDir = "data/processed/graphs_biochem_anchors",
    [string] $Step0Json = "outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json",
    [string] $InitStep1Ckpt = "outputs/biochem/sweep_clot_ml_physics_6h/step1_a35/clot_ml_step1_best.pth",
    [string] $Ckpt = "outputs/biochem/clot_ml_ladder/step3b_extrap/clot_ml_step3b_best.pth",
    [string] $Recipe = "data/reference/clot_ml_deploy_v1_extrap.json",
    [string] $KineCkpt = "outputs/kinematics/kinematics_best.pth",
    [double] $SimEndScale = 5.0,
    [int] $Epochs = 24,
    [int] $TrainSimEndScale = 2.0,
    [int] $MaxFrames = 20,
    [int] $TimeStride = 1,
    [string] $Val = "patient007",
    [switch] $SkipTrain
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_clot_prior_rule_winner_env.ps1")
. (Join-Path $PSScriptRoot "_python_rc.ps1")

$env:CLOT_ML_DEVICE = "cuda"
$env:CLOT_PHI_KINE_CKPT = $KineCkpt
$env:CLOT_PHI_MINIMAL_FEATURES = "1"
$env:CLOT_ML_USE_MACRO_TAU = "1"
$env:CLOT_ML_CONTINUOUS_EXTRAP = "1"
$env:CLOT_ML_SIM_END_SCALE = "$SimEndScale"
$env:PYTHONUNBUFFERED = "1"

Write-Host ""
Write-Host "[NEW] Step 3b continuous extrap (train H=$TrainSimEndScale, viz H=$SimEndScale)" -ForegroundColor Cyan

if (-not $SkipTrain) {
    Invoke-PythonRcCheck -Label "step3b train" -PyArgs @(
        "scripts/train_clot_ml_step3b_extrap.py",
        "--anchor-dir", $AnchorDir,
        "--step0-json", $Step0Json,
        "--init-step1-ckpt", $InitStep1Ckpt,
        "--val", $Val,
        "--epochs", "$Epochs",
        "--sim-end-scale", "$TrainSimEndScale"
    )
}

Invoke-PythonRcCheck -Label "step3b viz 5x phi" -PyArgs @(
    "scripts/viz_clot_ml_deploy_v1_timeline.py",
    "--anchor", $Anchor,
    "--anchor-dir", $AnchorDir,
    "--recipe", $Recipe,
    "--phi-only",
    "--no-flow",
    "--sim-end-scale", "$SimEndScale",
    "--time-stride", "$TimeStride",
    "--max-frames", "$MaxFrames",
    "--scatter-size", "4.0",
    "--growth-curve"
)

Write-Host ""
Write-Host "[OK] ckpt: $Ckpt" -ForegroundColor Green
Write-Host "[OK] PNG: outputs/biochem/viz/clot_deploy/deploy_v1_${Anchor}_phi_H*.png" -ForegroundColor Green
