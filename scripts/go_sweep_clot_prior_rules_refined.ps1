# S0 refined rule sweep: off-wall stag, raw dx ranking, tie-break (dx + hop).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_sweep_clot_prior_rules_refined.ps1"
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_sweep_clot_prior_rules_refined.ps1" -Fast -Top 15

param(
    [string] $AnchorDir = "data/processed/graphs_biochem_anchors",
    [switch] $Fast,
    [int] $Top = 20
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")
. (Join-Path $PSScriptRoot "_clot_deploy_binary_base.ps1")

$env:BIOCHEM_PRIOR_COMSOL_ALIGNED = "1"
$env:BIOCHEM_PRIOR_NORM_MASK = "adjacent"

$pyArgs = @(
    "scripts/sweep_clot_prior_rules_refined.py",
    "--anchor-dir", $AnchorDir,
    "--top", "$Top"
)
if ($Fast) { $pyArgs += "--fast" }

Write-Host "[NEW] refined prior rule sweep (offwall stag | raw dx | tie dx+hop)" -ForegroundColor Cyan
Invoke-PythonRcCheck @pyArgs -Label "refined prior rule sweep"

Write-Host "[OK]  json -> outputs/biochem/diagnostics/clot_prior_rule_sweep_refined.json" -ForegroundColor Green
