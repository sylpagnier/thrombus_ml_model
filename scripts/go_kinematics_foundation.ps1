# Stage-A foundation: mixed L0/L1/L2 geometry curriculum (no --limit-data).
# Prerequisite: mixed cohort graphs + backfill geometry_level (see below).
param(
    [switch]$Fresh,
    [int]$LimitData = 0
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "=== Kinematics foundation (geometry curriculum auto, full data) ===" -ForegroundColor Cyan
Write-Host "If graphs lack geometry_level, run backfill first:" -ForegroundColor Yellow
Write-Host "  python -m src.data_gen.backfill_kinematics_geometry_level" -ForegroundColor Yellow

$trainArgs = @(
    "-m", "src.training.train_kinematics_predictor",
    "--geometry-phase", "auto",
    "--hard-mining-start-epoch", "16",
    "--l0l1-only-epochs", "6"
)
if ($Fresh) { $trainArgs += "--fresh" }
if ($LimitData -gt 0) {
    $trainArgs += @("--limit-data", "$LimitData")
    Write-Host "Smoke mode: --limit-data $LimitData" -ForegroundColor Yellow
}

& python @trainArgs
