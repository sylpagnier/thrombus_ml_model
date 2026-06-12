# Clot ML side pivots: soft_commit | rule_mixture | data_driven
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_ml_pivots.ps1" -Pivot soft_commit
#   powershell ... -Pivot all -Epochs 30
#   powershell ... -Pivot data_driven -SkipTrain

param(
    [ValidateSet("soft_commit", "rule_mixture", "data_driven", "all")]
    [string] $Pivot = "soft_commit",
    [string] $Anchor = "patient007",
    [string] $AnchorDir = "data/processed/graphs_biochem_anchors",
    [string] $Step0Json = "outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json",
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

function Get-PivotCkpt([string] $Name) {
    switch ($Name) {
        "soft_commit" { return "outputs/biochem/clot_ml_ladder/pivot_soft_commit/clot_ml_pivot_soft_commit_best.pth" }
        "rule_mixture" { return "outputs/biochem/clot_ml_ladder/pivot_rule_mixture/clot_ml_pivot_rule_mixture_best.pth" }
        "data_driven" { return "outputs/biochem/clot_ml_ladder/pivot_data_driven/clot_ml_pivot_data_driven_best.pth" }
        default { throw "unknown pivot $Name" }
    }
}

$pivots = if ($Pivot -eq "all") { @("soft_commit", "rule_mixture", "data_driven") } else { @($Pivot) }

foreach ($p in $pivots) {
    Write-Host ""
    Write-Host "[NEW] pivot=$p (pred GINO-DEQ)" -ForegroundColor Cyan
    $ckpt = Get-PivotCkpt $p

    if (-not $SkipTrain) {
        Invoke-PythonRcCheck -Label "pivot $p train" -PyArgs @(
            "scripts/train_clot_ml_pivot.py",
            "--pivot", $p,
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
        "--pivot-ckpt", $ckpt,
        "--vel-source", "kinematics",
        "--kine-ckpt", $KineCkpt
    )
    Invoke-PythonRcCheck -Label "pivot $p viz" -PyArgs $vizArgs

    Write-Host "[OK] pivot=$p ckpt=$ckpt" -ForegroundColor Green
    Write-Host "  PNG: outputs/biochem/viz/clot_deploy/temporal_rule_${Anchor}_timeline_pivot_${p}.png"
}
