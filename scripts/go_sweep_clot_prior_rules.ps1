# Sweep prior rule OR-legs (prior_p, t0_strip, flux_stream, wall filters) on all anchors.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_sweep_clot_prior_rules.ps1"
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_sweep_clot_prior_rules.ps1" -Fast -Top 20

param(
    [string] $AnchorDir = "data/processed/graphs_biochem_anchors",
    [switch] $Fast,
    [int] $Top = 15
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")
. (Join-Path $PSScriptRoot "_clot_deploy_binary_base.ps1")

$env:BIOCHEM_PRIOR_COMSOL_ALIGNED = "1"
$env:BIOCHEM_PRIOR_NORM_MASK = "adjacent"

$pyArgs = @(
    "scripts/sweep_clot_prior_rules.py",
    "--anchor-dir", $AnchorDir,
    "--top", "$Top"
)
if ($Fast) { $pyArgs += "--fast" }

Write-Host "[NEW] prior rule sweep (S0 static_final, ceiling wall+2)" -ForegroundColor Cyan
Invoke-PythonRcCheck @pyArgs -Label "prior rule sweep"

Write-Host "[OK]  json -> outputs/biochem/diagnostics/clot_prior_rule_sweep.json" -ForegroundColor Green
