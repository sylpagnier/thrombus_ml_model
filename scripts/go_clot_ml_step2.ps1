# Step 2 ML ladder: band GNN risk ranker + viz.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_ml_step2.ps1"
#   powershell ... -Epochs 50 -SkipTrain

param(
    [string] $Anchor = "patient007",
    [string] $AnchorDir = "data/processed/graphs_biochem_anchors",
    [string] $Step0Json = "outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json",
    [string] $Ckpt = "outputs/biochem/clot_ml_ladder/step2_band_gnn/clot_ml_step2_best.pth",
    [string] $KineCkpt = "outputs/kinematics/kinematics_best.pth",
    [string] $Val = "patient007",
    [int] $Epochs = 50,
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
Write-Host "[NEW] Step 2 band GNN ranker (frozen Step0 progressive shell, pred GINO-DEQ)" -ForegroundColor Cyan

if (-not $SkipTrain) {
    Invoke-PythonRcCheck -Label "step2 train" -PyArgs @(
        "scripts/train_clot_ml_step2_band_gnn.py",
        "--anchor-dir", $AnchorDir,
        "--step0-json", $Step0Json,
        "--val", $Val,
        "--epochs", "$Epochs"
    )
}

$vizArgs = @(
    "scripts/viz_clot_temporal_rule_timeline.py",
    "--anchor", $Anchor,
    "--anchor-dir", $AnchorDir,
    "--keyframes", "$Keyframes",
    "--step2-ckpt", $Ckpt,
    "--vel-source", "kinematics",
    "--kine-ckpt", $KineCkpt
)
Invoke-PythonRcCheck -Label "step2 viz" -PyArgs $vizArgs

Write-Host ""
Write-Host "[OK] outputs:" -ForegroundColor Green
Write-Host "  ckpt: $Ckpt"
Write-Host "  PNG:  outputs/biochem/viz/clot_deploy/temporal_rule_${Anchor}_timeline_step2.png"
