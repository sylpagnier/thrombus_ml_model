# Sweep deploy rank pools (centerline cut, sdf cap, hops, dgamma, ...).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_diagnose_clot_rule_rank_masks.ps1"
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_diagnose_clot_rule_rank_masks.ps1" -Top 20

param(
    [string] $AnchorDir = "data/processed/graphs_biochem_anchors",
    [int] $Top = 15
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
. (Join-Path $PSScriptRoot "_clot_deploy_binary_base.ps1")

Invoke-PythonRcCheck @(
    "scripts/diagnose_clot_rule_rank_masks.py",
    "--anchor-dir", $AnchorDir,
    "--top", "$Top"
) -Label "clot rule rank mask diagnostic"

Write-Host "[OK]  json -> outputs/biochem/diagnostics/clot_rule_rank_mask_diagnostic.json" -ForegroundColor Green
