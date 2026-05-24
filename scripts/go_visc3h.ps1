# Unattended ~3h viscosity/velocity architecture sweep (one command, full log).
# Resumes automatically: skips archived legs; recovers L0 if train finished but copy failed.
# From anywhere:
#   powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\pgssy\thrombus_ml_model\scripts\go_visc3h.ps1"
# From repo root:
#   .\scripts\go_visc3h.ps1
#
param(
    [string[]] $SweepArgs = @()
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$logDir = Join-Path $RepoRoot "outputs\reports\training\biochem"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
$logPath = Join-Path $logDir ("visc3h_console_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

Write-Host "Visc-velocity 3h sweep | repo=$RepoRoot" -ForegroundColor Cyan
Write-Host "Console log: $logPath" -ForegroundColor Cyan
Write-Host "Checkpoints: outputs\biochem\sweep_visc_velocity_3h\" -ForegroundColor DarkGray
Write-Host ""

$sweepScript = Join-Path $RepoRoot "scripts\run_biochem_visc_velocity_arch_sweep_3h.ps1"
# Native python writes UserWarnings to stderr; with Stop + Tee-Object that aborts the sweep.
$ErrorActionPreference = "Continue"
& $sweepScript @SweepArgs 2>&1 | Tee-Object -FilePath $logPath
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) { exit $exitCode }
Write-Host ""
Write-Host "Done. Log: $logPath" -ForegroundColor Green
