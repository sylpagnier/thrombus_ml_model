# Deploy-mode clot timeline: GINO-DEQ kinematics (no GT flow) + GT-flow comparison.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_deploy_predkine_viz.ps1"
#   powershell ... -Anchor patient007

param(
    [string] $Anchor = "patient007",
    [string] $AnchorDir = "data/processed/graphs_biochem_anchors",
    [string] $KineCkpt = "outputs/kinematics/kinematics_best.pth",
    [int] $Keyframes = 8
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_clot_prior_rule_winner_env.ps1")
if (Test-Path (Join-Path $PSScriptRoot "_clot_architecture_winner_env.ps1")) {
    . (Join-Path $PSScriptRoot "_clot_architecture_winner_env.ps1")
}
. (Join-Path $PSScriptRoot "_python_rc.ps1")

$env:CLOT_TEMPORAL_VEL_SOURCE = "kinematics"
$env:CLOT_PHI_KINE_CKPT = $KineCkpt

Write-Host ""
Write-Host "[NEW] deploy clot timeline (GINO-DEQ kine) + GT comparison ($Anchor)" -ForegroundColor Cyan

Invoke-PythonRcCheck -Label "pred-kine deploy viz" -PyArgs @(
    "scripts/viz_clot_temporal_rule_timeline.py",
    "--anchor", $Anchor,
    "--anchor-dir", $AnchorDir,
    "--keyframes", "$Keyframes",
    "--vel-source", "kinematics",
    "--kine-ckpt", $KineCkpt,
    "--compare-gt"
)

Write-Host ""
Write-Host "[OK] PNG: outputs/biochem/viz/clot_deploy/temporal_rule_${Anchor}_timeline_predkine.png" -ForegroundColor Green
