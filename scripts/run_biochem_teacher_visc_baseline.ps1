# Teacher-only viscosity baseline (minimal robust objective).
#
# Core backward objective (step-2 data-only):
#   L_Data_Kine + L_Data_Bio
#   + W_MuLog * L_MuLog (all truth)
#   + W_MuLogWall * L_MuLog_wall
#   + W_MuLogHigh * L_MuLog_high
#   + W_MuSI * L_MuSI
#
# This is the intended "solid base" before adding extra losses incrementally.
#
# Usage:
#   .\scripts\run_biochem_teacher_visc_baseline.ps1
#   .\scripts\run_biochem_teacher_visc_baseline.ps1 -TeacherEpochs 24
#   .\scripts\run_biochem_teacher_visc_baseline.ps1 -MuLogWallWeight 3.0 -MuLogHighWeight 2.0
#
param(
    [int] $TeacherEpochs = 18,
    [float] $MuLogWeight = 2.0,
    [float] $MuSiWeight = 2.0,
    [float] $MuLogWallWeight = 2.5,
    [float] $MuLogHighWeight = 1.5,
    [switch] $ForcePretrain,
    [string[]] $ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# Clear stale env from previous experiments.
Get-ChildItem Env:BIOCHEM_* -ErrorAction SilentlyContinue | ForEach-Object {
    Remove-Item "Env:$($_.Name)" -ErrorAction SilentlyContinue
}

$WarmStart = Join-Path $RepoRoot "outputs\biochem\biochem_post_pretrain.pth"
$UseWarmStart = (-not $ForcePretrain) -and (Test-Path $WarmStart)

$env:BIOCHEM_PRESET = "teacher_visc_baseline"
$env:BIOCHEM_STOP_AFTER_TEACHER = "1"
$env:BIOCHEM_TEACHER_EPOCHS = "$TeacherEpochs"
$env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "$MuLogWeight"
$env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "$MuSiWeight"
$env:BIOCHEM_MU_LOG_WALL_WEIGHT = "$MuLogWallWeight"
$env:BIOCHEM_MU_LOG_HIGH_WEIGHT = "$MuLogHighWeight"
$env:BIOCHEM_TRAIN_MODE = "new"
$env:BIOCHEM_SKIP_PRETRAIN = if ($UseWarmStart) { "1" } else { "0" }
$env:BIOCHEM_REUSE_LAST_PRETRAIN = if ($UseWarmStart) { "1" } else { "0" }

Write-Host "Teacher viscosity baseline run (preset=teacher_visc_baseline)" -ForegroundColor Cyan
Write-Host "Teacher epochs: $TeacherEpochs | Warm-start: $UseWarmStart | Corrector: OFF" -ForegroundColor Cyan
Write-Host "Weights: MuLog=$MuLogWeight MuSI=$MuSiWeight MuLogWall=$MuLogWallWeight MuLogHigh=$MuLogHighWeight" -ForegroundColor Cyan
if ($UseWarmStart) {
    Write-Host "Warm-start checkpoint: $WarmStart" -ForegroundColor Yellow
}

python -m src.training.train_biochem_corrector --new @ExtraArgs
if ($LASTEXITCODE -ne 0) {
    throw "train_biochem_corrector failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Done. Check outputs/reports/training/biochem/<timestamp>/metrics.jsonl" -ForegroundColor Green
Write-Host "Track val: mu_log_mae(all/wall/high), mu_pearson, and train L_Back."
