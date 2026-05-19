# Teacher-only robust viscosity run (max complexity, all important losses active).
#
# Uses train_biochem_corrector preset:
#   BIOCHEM_PRESET=teacher_max_complexity
#
# This keeps STOP_AFTER_TEACHER=1 and enables full multitask step-3 teacher loss
# (Kendall + PDE + supervised data + mu anchors) with viscosity-focused defaults.
#
# Usage:
#   .\scripts\run_biochem_teacher_max_complexity.ps1
#   .\scripts\run_biochem_teacher_max_complexity.ps1 -TeacherEpochs 30 -ForcePretrain
#   .\scripts\run_biochem_teacher_max_complexity.ps1 -ExtraArgs @("--new")
#
param(
    [int] $TeacherEpochs = 24,
    [switch] $ForcePretrain,
    [string[]] $ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# Reset previous BIOCHEM_* shell state to avoid stale isolate/diagnostic vars.
Get-ChildItem Env:BIOCHEM_* -ErrorAction SilentlyContinue | ForEach-Object {
    Remove-Item "Env:$($_.Name)" -ErrorAction SilentlyContinue
}

$WarmStart = Join-Path $RepoRoot "outputs\biochem\biochem_post_pretrain.pth"
$UseWarmStart = (-not $ForcePretrain) -and (Test-Path $WarmStart)

$env:BIOCHEM_PRESET = "teacher_max_complexity"
$env:BIOCHEM_STOP_AFTER_TEACHER = "1"
$env:BIOCHEM_TEACHER_EPOCHS = "$TeacherEpochs"
$env:BIOCHEM_TEACHER_VAL_EVERY = "2"
$env:BIOCHEM_VAL_TIME_STRIDE = "10"
$env:BIOCHEM_DEBUG = "0"
$env:BIOCHEM_TRAIN_MODE = "new"
$env:BIOCHEM_SKIP_PRETRAIN = if ($UseWarmStart) { "1" } else { "0" }
$env:BIOCHEM_REUSE_LAST_PRETRAIN = if ($UseWarmStart) { "1" } else { "0" }

Write-Host "Teacher max-complexity run (preset=teacher_max_complexity)" -ForegroundColor Cyan
Write-Host "Teacher epochs: $TeacherEpochs | Warm-start: $UseWarmStart | Corrector: OFF" -ForegroundColor Cyan
if ($UseWarmStart) {
    Write-Host "Warm-start checkpoint: $WarmStart" -ForegroundColor Yellow
}

python -m src.training.train_biochem_corrector --new @ExtraArgs
if ($LASTEXITCODE -ne 0) {
    throw "train_biochem_corrector failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Done. Check outputs/reports/training/biochem/<timestamp>/metrics.jsonl" -ForegroundColor Green
Write-Host "Track val: mu_log_mae (all/wall/high), mu_pearson, and train L_Back."
