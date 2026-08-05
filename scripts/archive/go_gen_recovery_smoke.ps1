# Generalization Recovery - 15-Minute Smoke Test (< 3 min actual execution)
#
# Verifies end-to-end execution of anti-memorization training + held-out evaluation.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_gen_recovery_smoke.ps1 -Fresh

param(
    [switch] $Fresh
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "[SMOKE TEST] Generalization Recovery Pipeline (< 3 min)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Fast uncoupled flow during training so per-epoch validation is instant (~1s)
$env:SPECIES_CLOSED_LOOP_COUPLING = "0"

$smokeArgs = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $PSScriptRoot "go_gen_recovery_sweep.ps1"),
    "-Epochs", "2",
    "-EarlyStop", "2",
    "-MaxWindows", "12",
    "-TrainAnchors", "patient005",
    "-ValAnchor", "patient020",
    "-HoldoutAnchors", "patient020",
    "-ArmFilter", "K",
    "-CheapVal",
    "-RunRoot", "outputs/biochem/eda/smoke_gen_recovery"
)
if ($Fresh) { $smokeArgs += "-Fresh" }

$start = Get-Date
& powershell @smokeArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERR] Smoke test failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
$elapsed = [int]((Get-Date) - $start).TotalMinutes

Write-Host ""
Write-Host "[OK] Smoke test PASSED in $elapsed min!" -ForegroundColor Green
Write-Host "[i] All training, CLI overrides, and held-out evals verified cleanly." -ForegroundColor DarkGray
