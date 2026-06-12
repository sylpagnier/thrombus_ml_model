# Option A: prior rule baseline (no training).
# Rule: sweep winner prior_p0.80 inside ceiling; fixed_mu projection.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_deploy_s0_rule_baseline.ps1"
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_deploy_s0_rule_baseline.ps1" -Anchor patient007

param(
    [string] $AnchorDir = "data/processed/graphs_biochem_anchors",
    [string] $Anchor = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")
. (Join-Path $PSScriptRoot "_clot_deploy_binary_base.ps1")

. (Join-Path $PSScriptRoot "_clot_prior_rule_winner_env.ps1")
$env:BIOCHEM_PRIOR_COMSOL_ALIGNED = "1"
$env:BIOCHEM_PRIOR_NORM_MASK = "adjacent"

$pyArgs = @("scripts/eval_clot_prior_rule_baseline.py", "--anchor-dir", $AnchorDir)
if ($Anchor) { $pyArgs += @("--anchor", $Anchor) }

Write-Host "[NEW] prior rule baseline (prior_p0.80, ceiling wall+2, fixed_mu)" -ForegroundColor Cyan
Invoke-PythonRcCheck @pyArgs -Label "prior rule eval"

Write-Host "[OK]  json -> outputs/biochem/diagnostics/clot_prior_rule_baseline.json" -ForegroundColor Green
