# Comprehensive clot rule architecture sweep (clot_shape north-star).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_sweep_clot_rule_architectures.ps1"
#   powershell ... -Fast -Anchor patient007

param(
    [string] $AnchorDir = "data/processed/graphs_biochem_anchors",
    [string] $Anchor = "",
    [switch] $Fast,
    [switch] $IncludeOracle
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_clot_prior_rule_winner_env.ps1")
. (Join-Path $PSScriptRoot "_python_rc.ps1")

$pyArgs = @(
    "scripts/sweep_clot_rule_architectures.py",
    "--anchor-dir", $AnchorDir
)
if ($Anchor) { $pyArgs += @("--anchor", $Anchor) }
if ($Fast) { $pyArgs += "--fast" }
if ($IncludeOracle) { $pyArgs += "--include-oracle" }

Write-Host ""
Write-Host "[NEW] comprehensive clot rule architecture sweep (clot_shape metric)" -ForegroundColor Cyan
Invoke-PythonRcCheck -Label "clot rule architecture sweep" -PyArgs $pyArgs
