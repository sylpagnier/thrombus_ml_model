# Overnight ~10h health architecture sweep (tee console log).
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = Join-Path $RepoRoot "outputs\reports\training\biochem"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
$logPath = Join-Path $logDir "health10h_console_$ts.log"
Write-Host "Console log: $logPath" -ForegroundColor Cyan
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts\run_biochem_health_arch_sweep_10h.ps1") *>&1 |
    Tee-Object -FilePath $logPath
exit $LASTEXITCODE
