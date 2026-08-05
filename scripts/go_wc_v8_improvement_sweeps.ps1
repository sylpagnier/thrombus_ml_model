# WC v8 compound post-promote improvement sweeps.
#
# Axes: hops eval sweep, 010 FP polish, frontier-h1 retrain, 007 recall, partial unfreeze.
# Driver: scripts/run_wc_v8_improvement_sweeps.py
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_wc_v8_improvement_sweeps.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_wc_v8_improvement_sweeps.ps1 -Only hops_sweep,fp_polish_010
#

param(
    [double] $DeadlineHours = 12.0,
    [string] $RunRoot = "outputs/biochem/offwall_model/wc_v8_improvement_sweeps",
    [string] $InitGrowth = "outputs/biochem/biochem_gnn/locked/compound_growth_best.pth",
    [string] $Only = "",
    [switch] $Fresh
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"
$env:SPECIES_CONTINUOUS_VEL_DECAY = "1"
$env:SPECIES_CONTINUOUS_VEL_DECAY_WALL_ONLY = "1"

Write-Host "[NEW] go_wc_v8_improvement_sweeps hours=$DeadlineHours out=$RunRoot" -ForegroundColor Cyan

$pyArgs = @(
    "scripts/run_wc_v8_improvement_sweeps.py",
    "--deadline-hours", "$DeadlineHours",
    "--run-root", $RunRoot,
    "--init-growth", $InitGrowth
)
if ($Only) { $pyArgs += @("--only", $Only) }
if ($Fresh) { $pyArgs += "--fresh" }

Invoke-PythonRcCheck -Label "wc_v8_improvement_sweeps" -PyArgs $pyArgs
Write-Host "[OK] improvement sweeps finished -> $RunRoot" -ForegroundColor Green
