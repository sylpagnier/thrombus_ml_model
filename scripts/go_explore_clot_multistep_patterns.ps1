# 5-snapshot clot pattern probe + rule sweep (all biochem anchors).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_explore_clot_multistep_patterns.ps1"
#   powershell ... -Anchor patient007 -Detail

param(
    [string] $AnchorDir = "data/processed/graphs_biochem_anchors",
    [string] $Anchor = "",
    [switch] $Detail
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")

$pyArgs = @(
    "scripts/explore_clot_multistep_patterns.py",
    "--anchor-dir", $AnchorDir
)
if ($Anchor) { $pyArgs += @("--anchor", $Anchor) }
if ($Detail) { $pyArgs += "--detail" }

Write-Host ""
Write-Host "[NEW] multi-timestep clot pattern probe (5 snapshots)" -ForegroundColor Cyan
Invoke-PythonRcCheck -Label "clot multistep probe" -PyArgs $pyArgs
