# Step 7b ML ladder: frozen rule_mixture shell + residual MLP + viz.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_ml_step7b.ps1"
#   powershell ... -Epochs 40 -SkipTrain

param(
    [string] $Anchor = "patient007",
    [string] $AnchorDir = "data/processed/graphs_biochem_anchors",
    [string] $Step0Json = "outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json",
    [string] $MixtureCkpt = "outputs/biochem/clot_ml_ladder/pivot_rule_mixture/clot_ml_pivot_rule_mixture_best.pth",
    [string] $Ckpt = "outputs/biochem/clot_ml_ladder/step7b_hybrid/clot_ml_step7b_best.pth",
    [string] $KineCkpt = "outputs/kinematics/kinematics_best.pth",
    [string] $Val = "patient007",
    [int] $Epochs = 40,
    [int] $Keyframes = 8,
    [switch] $SkipTrain
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_clot_prior_rule_winner_env.ps1")
. (Join-Path $PSScriptRoot "_python_rc.ps1")

$env:CLOT_ML_DEVICE = "cuda"

Write-Host ""
Write-Host "[NEW] Step 7b hybrid (frozen rule_mixture + residual MLP, pred GINO-DEQ)" -ForegroundColor Cyan

if (-not $SkipTrain) {
    Invoke-PythonRcCheck -Label "step7b train" -PyArgs @(
        "scripts/train_clot_ml_step7b_hybrid.py",
        "--anchor-dir", $AnchorDir,
        "--step0-json", $Step0Json,
        "--mixture-ckpt", $MixtureCkpt,
        "--val", $Val,
        "--epochs", "$Epochs"
    )
}

$vizArgs = @(
    "scripts/viz_clot_temporal_rule_timeline.py",
    "--anchor", $Anchor,
    "--anchor-dir", $AnchorDir,
    "--keyframes", "$Keyframes",
    "--step7b-ckpt", $Ckpt,
    "--vel-source", "kinematics",
    "--kine-ckpt", $KineCkpt
)
Invoke-PythonRcCheck -Label "step7b viz" -PyArgs $vizArgs

Write-Host ""
Write-Host "[OK] outputs:" -ForegroundColor Green
Write-Host "  ckpt: $Ckpt"
Write-Host "  PNG:  outputs/biochem/viz/clot_deploy/temporal_rule_${Anchor}_timeline_step7b.png"
