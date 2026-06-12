# Timeline viz for temporal growing rule (patient007 default).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_temporal_rule_timeline_viz.ps1"
#   powershell ... -Anchor patient007 -Keyframes 10

param(
    [string] $Anchor = "patient007",
    [string] $AnchorDir = "data/processed/graphs_biochem_anchors",
    [int] $Keyframes = 8
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_clot_prior_rule_winner_env.ps1")
if (Test-Path (Join-Path $PSScriptRoot "_clot_architecture_winner_env.ps1")) {
    . (Join-Path $PSScriptRoot "_clot_architecture_winner_env.ps1")
} else {
    . (Join-Path $PSScriptRoot "_clot_localized_rule_winner_env.ps1")
}
. (Join-Path $PSScriptRoot "_python_rc.ps1")

$pyArgs = @(
    "scripts/viz_clot_temporal_rule_timeline.py",
    "--anchor", $Anchor,
    "--anchor-dir", $AnchorDir,
    "--keyframes", $Keyframes
)

Write-Host ""
Write-Host "[NEW] temporal growing rule timeline viz ($Anchor)" -ForegroundColor Cyan
Invoke-PythonRcCheck -Label "temporal rule timeline" -PyArgs $pyArgs
