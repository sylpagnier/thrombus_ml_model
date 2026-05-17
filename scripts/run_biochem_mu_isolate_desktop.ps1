# Desktop μ hunt: full TBPTT (no detach), low-TF, five teacher legs (P→D).
# From repo root, one command:
#   .\mu_desktop.ps1

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Legs = @("P", "A", "B", "C", "D")
$TeacherEpochs = 22
$ValStride = 5

$env:BIOCHEM_TRAIN_MODE = "new"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$WarmStart = Join-Path $RepoRoot "outputs\biochem\biochem_post_pretrain.pth"
$UseWarmStart = Test-Path $WarmStart

$TbpttWindow = "6"
$Rk4Sub = "24"
$DetachMacro = "0"
$Workers = "4"

$Base = @{
    BIOCHEM_STOCK_DEFAULTS = "1"
    BIOCHEM_SKIP_PRETRAIN = if ($UseWarmStart) { "1" } else { "0" }
    BIOCHEM_REUSE_LAST_PRETRAIN = if ($UseWarmStart) { "1" } else { "0" }
    BIOCHEM_STOP_AFTER_TEACHER = "1"
    BIOCHEM_NO_TEACHER_DEFAULTS = "1"
    BIOCHEM_LOSS_DATA_ONLY = "1"
    BIOCHEM_PRESET = ""
    BIOCHEM_COMPLEXITY_STEP = "2"
    BIOCHEM_TEACHER_MU_RATIO_MAX = "80.0"
    BIOCHEM_TEACHER_EPOCHS = "$TeacherEpochs"
    BIOCHEM_TEACHER_VAL_EVERY = "2"
    BIOCHEM_VAL_TIME_STRIDE = "$ValStride"
    BIOCHEM_TBPTT_MAX_WINDOW = $TbpttWindow
    BIOCHEM_TBPTT_WINDOW_CURRICULUM = "0"
    BIOCHEM_DETACH_MACRO_STATE = $DetachMacro
    BIOCHEM_ADJOINT_RK4_SUBSTEPS = $Rk4Sub
    BIOCHEM_DATALOADER_WORKERS = $Workers
    BIOCHEM_MU_SI_MULTI_STEP = "1"
    BIOCHEM_MU_SI_HUBER_DELTA = "0.25"
    BIOCHEM_MU_ANCHOR_LATE_TIME_WEIGHT = "1"
    BIOCHEM_MU_LATE_TIME_POWER = "2.0"
    BIOCHEM_FI_GATE_START_WEIGHT = "0.0"
    BIOCHEM_TEACHER_SKIP_VAL = "0"
    BIOCHEM_DEBUG = "0"
    BIOCHEM_PSEUDO_MIN_TEACHER_MU_SCORE = "-1.35"
}

$LegDefs = @{
    P = @{
        Label = "P_past_lowTF_musi_earlywin"
        BIOCHEM_LOSS_ISOLATE = "MU_SI"
        BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "8.0"
        BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "0.0"
        BIOCHEM_TEACHER_FORCE_MIN = "0.0"
        BIOCHEM_TEACHER_TF_WARMUP_EPOCHS = "4"
        BIOCHEM_TBPTT_ANCHOR_END_BIAS = "0"
        BIOCHEM_TBPTT_ANCHOR_RANDOM_START = "0"
        BIOCHEM_TEACHER_TBPTT_RANDOM_ANCHOR = "0"
    }
    A = @{
        Label = "A_joint_mu_endbias_detach0"
        BIOCHEM_LOSS_ISOLATE = "MU_SI"
        BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "8.0"
        BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "2.0"
        BIOCHEM_TEACHER_FORCE_MIN = "0.0"
        BIOCHEM_TEACHER_TF_WARMUP_EPOCHS = "4"
        BIOCHEM_TBPTT_ANCHOR_END_BIAS = "1"
        BIOCHEM_TBPTT_ANCHOR_RANDOM_START = "0"
        BIOCHEM_TEACHER_TBPTT_RANDOM_ANCHOR = "0"
    }
    B = @{
        Label = "B_mu_log_only_endbias"
        BIOCHEM_LOSS_ISOLATE = "MU_LOG"
        BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "0.0"
        BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "2.0"
        BIOCHEM_TEACHER_FORCE_MIN = "0.0"
        BIOCHEM_TEACHER_TF_WARMUP_EPOCHS = "4"
        BIOCHEM_TBPTT_ANCHOR_END_BIAS = "1"
        BIOCHEM_TBPTT_ANCHOR_RANDOM_START = "0"
        BIOCHEM_TEACHER_TBPTT_RANDOM_ANCHOR = "0"
    }
    C = @{
        Label = "C_mu_si_only_endbias"
        BIOCHEM_LOSS_ISOLATE = "MU_SI"
        BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "8.0"
        BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "0.0"
        BIOCHEM_TEACHER_FORCE_MIN = "0.0"
        BIOCHEM_TEACHER_TF_WARMUP_EPOCHS = "4"
        BIOCHEM_TBPTT_ANCHOR_END_BIAS = "1"
        BIOCHEM_TBPTT_ANCHOR_RANDOM_START = "0"
        BIOCHEM_TEACHER_TBPTT_RANDOM_ANCHOR = "0"
    }
    D = @{
        Label = "D_early_window_control"
        BIOCHEM_LOSS_ISOLATE = "MU_SI"
        BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "8.0"
        BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "2.0"
        BIOCHEM_TEACHER_FORCE_MIN = "0.0"
        BIOCHEM_TEACHER_TF_WARMUP_EPOCHS = "4"
        BIOCHEM_TBPTT_ANCHOR_END_BIAS = "0"
        BIOCHEM_TBPTT_ANCHOR_RANDOM_START = "0"
        BIOCHEM_TEACHER_TBPTT_RANDOM_ANCHOR = "0"
    }
}

$SummaryPath = Join-Path $RepoRoot "outputs\reports\training\biochem\mu_isolate_desktop_summary.txt"
$SummaryDir = Split-Path -Parent $SummaryPath
if (-not (Test-Path $SummaryDir)) { New-Item -ItemType Directory -Path $SummaryDir -Force | Out-Null }

Write-Host "Desktop mu-isolate (detach=0 TBPTT=6 legs P,A,B,C,D x $TeacherEpochs ep)" -ForegroundColor Cyan
if ($UseWarmStart) { Write-Host "Warm-start: $WarmStart" -ForegroundColor Yellow }
"started $(Get-Date -Format o)" | Set-Content $SummaryPath

$legIndex = 0
foreach ($leg in $Legs) {
    $key = $leg.ToUpper()
    $legIndex++
    $def = $LegDefs[$key]
    $label = $def.Label

    Write-Host ""
    Write-Host "========== Leg $key : $label ($legIndex / $($Legs.Count)) ==========" -ForegroundColor Cyan

    Get-ChildItem Env:BIOCHEM_* -ErrorAction SilentlyContinue | ForEach-Object {
        Remove-Item "Env:$($_.Name)" -ErrorAction SilentlyContinue
    }
    foreach ($k in $Base.Keys) { Set-Item -Path "Env:$k" -Value $Base[$k] }
    foreach ($k in $def.Keys) {
        if ($k -eq "Label") { continue }
        Set-Item -Path "Env:$k" -Value $def[$k]
    }
    if ($legIndex -gt 1 -or $UseWarmStart) {
        $env:BIOCHEM_SKIP_PRETRAIN = "1"
        $env:BIOCHEM_REUSE_LAST_PRETRAIN = "1"
    }

    $t0 = Get-Date
    python -m src.training.train_biochem_corrector --new
    if ($LASTEXITCODE -ne 0) {
        $line = "Leg $key FAILED exit=$LASTEXITCODE at $(Get-Date -Format o)"
        Add-Content $SummaryPath $line
        Write-Host $line -ForegroundColor Red
        exit $LASTEXITCODE
    }
    $dt = (Get-Date) - $t0
    $line = "Leg $key OK $label minutes=$([int]$dt.TotalMinutes)"
    Add-Content $SummaryPath $line
    Write-Host $line -ForegroundColor Green
}

Write-Host ""
Write-Host "Done. Summary: $SummaryPath"
