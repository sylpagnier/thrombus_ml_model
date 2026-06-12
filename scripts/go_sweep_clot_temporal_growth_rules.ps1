# Temporal clot dynamics probe + growing rule sweep (all biochem anchors).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_sweep_clot_temporal_growth_rules.ps1"
#   powershell ... -Anchor patient007 -ProbeOnly
#   powershell ... -RulesOnly

param(
    [string] $AnchorDir = "data/processed/graphs_biochem_anchors",
    [string] $Anchor = "",
    [switch] $ProbeOnly,
    [switch] $RulesOnly,
    [switch] $Localized,
    [switch] $SpeciesProbe
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_clot_prior_rule_winner_env.ps1")
. (Join-Path $PSScriptRoot "_python_rc.ps1")

$pyArgs = @(
    "scripts/sweep_clot_temporal_growth_rules.py",
    "--anchor-dir", $AnchorDir
)
if ($Anchor) { $pyArgs += @("--anchor", $Anchor) }
if ($ProbeOnly) { $pyArgs += "--probe-only" }
if ($RulesOnly) { $pyArgs += "--rules-only" }
if ($Localized) { $pyArgs += "--localized" }
if ($SpeciesProbe) { $pyArgs += "--species-probe" }

Write-Host ""
Write-Host "[NEW] temporal clot probe + growing rule sweep" -ForegroundColor Cyan
Invoke-PythonRcCheck -Label "clot temporal growth sweep" -PyArgs $pyArgs
