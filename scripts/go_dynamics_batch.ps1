$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "[i] Starting dynamics legs batch (1/3): WG_sched_sample" -ForegroundColor Cyan
pwsh scripts/go_wall_gen_probe.ps1 -Leg "WG_sched_sample" -Epochs 30 -MaxWindows 24

Write-Host "`n[i] Starting dynamics legs batch (2/3): WG_noise_boost" -ForegroundColor Cyan
pwsh scripts/go_wall_gen_probe.ps1 -Leg "WG_noise_boost" -Epochs 30 -MaxWindows 24

Write-Host "`n[i] Starting dynamics legs batch (3/3): WG_long_tbptt" -ForegroundColor Cyan
pwsh scripts/go_wall_gen_probe.ps1 -Leg "WG_long_tbptt" -Epochs 30 -MaxWindows 24

Write-Host "`n[OK] Dynamics batch completed." -ForegroundColor Green
