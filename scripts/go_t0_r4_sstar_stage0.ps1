# S-star Stage 0 (no training): T0 preflight + s*_0 baseline eval/viz + optional audits.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_t0_r4_sstar_stage0.ps1"
#   powershell ... -Anchor patient007 -IncludeAudits

param(
    [string] $Anchor = "patient007",
    [string] $Times = "0,15,29,53",
    [switch] $IncludeAudits,
    [string] $TeacherCkpt = "outputs/biochem/biochem_teacher_last.pth"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")

$env:PYTHONUNBUFFERED = "1"
$env:CUDA_VISIBLE_DEVICES = if ($env:CUDA_VISIBLE_DEVICES) { $env:CUDA_VISIBLE_DEVICES } else { "0" }

$StageDir = Join-Path $RepoRoot "outputs\biochem\t0_r4_sstar\stage0"
New-Item -ItemType Directory -Force -Path $StageDir | Out-Null

Write-Host "[NEW] S-star Stage 0: T0 preflight ($Anchor)" -ForegroundColor Cyan
Invoke-PythonRcCheck -Label "s-star preflight" -PyArgs @(
    "scripts/diagnose_t0_r4_sweep_preflight.py",
    "--anchor", $Anchor,
    "--out", (Join-Path $StageDir "preflight.json")
)

Write-Host "[NEW] S-star s*_0 baseline eval+viz (step=s0)" -ForegroundColor Cyan
& (Join-Path $PSScriptRoot "go_t0_rung4_step.ps1") -Anchor $Anchor -Times $Times -Step s0 -TeacherCkpt $TeacherCkpt

if ($IncludeAudits) {
    Write-Host "[NEW] Audit ceiling s0_oracle_fi_mat (never deploy)" -ForegroundColor Cyan
    & (Join-Path $PSScriptRoot "go_t0_rung4_step.ps1") -Anchor $Anchor -Times $Times -Step s0_oracle_fi_mat -TeacherCkpt $TeacherCkpt
}

Write-Host "[OK] Stage 0 done. Next: s*_G1 rule sweep, then s*_G4 gate train." -ForegroundColor Green
Write-Host "[i] preflight=$(Join-Path $StageDir 'preflight.json')" -ForegroundColor Green
Write-Host "[i] eval=outputs/biochem/clot_trigger/t0_rung4_s0_${Anchor}.json" -ForegroundColor Green
Write-Host "[i] viz=outputs/biochem/viz/clot_trigger/t0_rung4_s0_${Anchor}.png" -ForegroundColor Green
