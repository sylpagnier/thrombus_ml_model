# Overnight ~10h health architecture sweep (9 legs, tee console log).
# Includes K0 Carreau-only kinematic probe first, then Gemini / simple-μ / μ₁μ₂ / reference legs.
# Each leg saves: outputs\biochem\sweep_health_arch_10h\<leg_id>\biochem_teacher_best_high_mu.pth
#
# One line from repo root (go AFK):
#   .\scripts\go_health10h.ps1
#
# Morning: sort manifest by viz_health_score (lower = healthier rollout).

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = Join-Path $RepoRoot "outputs\reports\training\biochem"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}
$logPath = Join-Path $logDir "health10h_console_$ts.log"

$postPretrain = Join-Path $RepoRoot "outputs\biochem\biochem_post_pretrain.pth"
Write-Host ""
Write-Host 'go_health10h - unattended sweep (~10h target)' -ForegroundColor Cyan
Write-Host '  K0_carreau_kinematic [8 ep, no clot] then R0, G0, G1, S0, S1, M0, M1, M2' -ForegroundColor DarkGray
if (Test-Path $postPretrain) {
    Write-Host "  Warm-start: REUSE $postPretrain on legs 2-9" -ForegroundColor DarkGray
} else {
    Write-Host '  Warm-start: none (K0 runs AE+ODE pretrain first, then saves post_pretrain)' -ForegroundColor Yellow
    Write-Host '  Do NOT pass -ForcePretrain (that skips pretrain). Missing post_pretrain is OK for a fresh run.' -ForegroundColor DarkGray
}
Write-Host "  Console log: $logPath" -ForegroundColor DarkGray
Write-Host "  Manifest:    outputs\biochem\sweep_health_arch_10h\manifest.jsonl" -ForegroundColor DarkGray
Write-Host ""

$sweepScript = Join-Path $RepoRoot "scripts\run_biochem_health_arch_sweep_10h.ps1"
& powershell -NoProfile -ExecutionPolicy Bypass -File $sweepScript *>&1 |
    Tee-Object -FilePath $logPath
exit $LASTEXITCODE
