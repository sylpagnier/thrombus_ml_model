# Step 0 ML ladder: learn rule coefficients (pred GINO-DEQ) + promote + viz.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_ml_step0.ps1"
#   powershell ... -Fast -Anchor patient007
#   powershell ... -Loao
#   powershell ... -SkipTrain   # viz only from existing best_coef.json

param(
    [string] $Anchor = "patient007",
    [string] $AnchorDir = "data/processed/graphs_biochem_anchors",
    [string] $KineCkpt = "outputs/kinematics/kinematics_best.pth",
    [string] $OutJson = "outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json",
    [int] $Keyframes = 8,
    [switch] $Fast,
    [switch] $Loao,
    [switch] $SkipTrain
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_clot_prior_rule_winner_env.ps1")
. (Join-Path $PSScriptRoot "_python_rc.ps1")

$env:CLOT_ML_DEVICE = "cuda"

Write-Host ""
Write-Host "[NEW] Step 0 ML rule coef learn + viz ($Anchor)" -ForegroundColor Cyan

if (-not $SkipTrain) {
    $trainArgs = @(
        "scripts/train_clot_ml_step0_coef.py",
        "--anchor-dir", $AnchorDir,
        "--out-dir", (Split-Path -Parent $OutJson)
    )
    if ($Fast) { $trainArgs += "--fast" }
    if ($Loao) {
        $trainArgs += "--loao"
    } elseif ($Anchor) {
        $trainArgs += "--anchor", $Anchor
    }
    Invoke-PythonRcCheck -Label "step0 coef train" -PyArgs $trainArgs
}

Invoke-PythonRcCheck -Label "step0 promote" -PyArgs @(
    "scripts/promote_clot_ml_step0_coef.py",
    "--json", $OutJson
)

$step0Env = Join-Path $PSScriptRoot "_clot_ml_step0_env.ps1"
if (Test-Path $step0Env) { . $step0Env }

$vizArgs = @(
    "scripts/viz_clot_temporal_rule_timeline.py",
    "--anchor", $Anchor,
    "--anchor-dir", $AnchorDir,
    "--keyframes", $Keyframes,
    "--step0-json", $OutJson,
    "--vel-source", "kinematics",
    "--kine-ckpt", $KineCkpt
)
Invoke-PythonRcCheck -Label "step0 viz" -PyArgs $vizArgs

Write-Host ""
Write-Host "[OK] outputs:" -ForegroundColor Green
Write-Host "  coef: $OutJson"
Write-Host "  env:  scripts/_clot_ml_step0_env.ps1"
Write-Host "  PNG:  outputs/biochem/viz/clot_deploy/temporal_rule_${Anchor}_timeline_step0.png"
