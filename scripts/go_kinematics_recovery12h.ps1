# One-line overnight kinematics recovery sweep (~12h) on main graphs_kinematics/newtonian.
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_kinematics_recovery12h.ps1"
param(
    [string[]] $Legs = @(),
    [switch] $DryRun,
    [switch] $Force
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = Join-Path $RepoRoot "outputs\reports\training\kinematics"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}
$logPath = Join-Path $logDir "recovery12h_console_$ts.log"

Write-Host ""
Write-Host "go_kinematics_recovery12h - 8-leg recovery sweep" -ForegroundColor Cyan
Write-Host "  Prerequisite: python -m src.data_gen.backfill_kinematics_geometry_level" -ForegroundColor Yellow
Write-Host "  Data: data/processed/graphs_kinematics/newtonian (NOT ab_bend_*)" -ForegroundColor DarkGray
Write-Host "  Log: $logPath" -ForegroundColor DarkGray
Write-Host ""

$sweepArgs = @()
if ($Legs.Count -gt 0) { $sweepArgs += "-Legs"; $sweepArgs += $Legs }
if ($DryRun) { $sweepArgs += "-DryRun" }
if ($Force) { $sweepArgs += "-Force" }

$sweepScript = Join-Path $RepoRoot 'scripts\run_kinematics_recovery_sweep_12h.ps1'
& powershell -NoProfile -ExecutionPolicy Bypass -File $sweepScript @sweepArgs *>&1 |
    Tee-Object -FilePath $logPath
exit $LASTEXITCODE
