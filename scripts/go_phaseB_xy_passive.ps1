# Phase B passive: ramp1 (data-only backward) then ramp2 (data + ADR in backward).
# GT [u,v,p]; no Stage-A kin training required (BIOCHEM_GT_KINE_VEL=1).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_phaseB_xy_passive.ps1"

param(
    [int] $Ramp1Epochs = 3,
    [int] $Ramp2Epochs = 6
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Set-PhaseBPassiveEnv {
    param([string] $RunNote)
    $env:PYTHONHASHSEED = "420"
    $env:CUBLAS_WORKSPACE_CONFIG = ":16:8"
    $env:BIOCHEM_PRESET = "passive_transport"
    $env:BIOCHEM_STOCK_DEFAULTS = "0"
    $env:BIOCHEM_NO_TEACHER_DEFAULTS = "1"
    $env:BIOCHEM_RUN_NOTE = $RunNote
    $env:BIOCHEM_TRAIN_MODE = "new"
    $env:BIOCHEM_RESUME = "0"
    $env:BIOCHEM_SKIP_PRETRAIN = "1"
    $env:BIOCHEM_INIT_FROM_BEST = "1"
    $env:BIOCHEM_REUSE_LAST_PRETRAIN = "1"
    $env:BIOCHEM_STOP_AFTER_TEACHER = "1"
    $env:BIOCHEM_TEACHER_FORCE_MIN = "1.0"
    $env:BIOCHEM_TEACHER_TF_WARMUP_EPOCHS = "99"
    $env:BIOCHEM_TEACHER_ODE_FREEZE_EPOCHS = "0"
    $env:BIOCHEM_TRAIN_ODE = "1"
    $env:BIOCHEM_TRAIN_KIN_LORA = "0"
    $env:BIOCHEM_GT_KINE_VEL = "1"
    $env:BIOCHEM_GT_KINE_SKIP_DEQ = "1"
    $env:BIOCHEM_TEACHER_LR = "1e-3"
    $env:BIOCHEM_TEACHER_PHYSICS_CLIP_NORM = "10"
    $env:BIOCHEM_TBPTT_MAX_WINDOW = "5"
    $env:BIOCHEM_TBPTT_WINDOW_CURRICULUM = "0"
    $env:BIOCHEM_TBPTT_ANCHOR_RANDOM_START = "0"
    $env:BIOCHEM_DATALOADER_WORKERS = "0"
    $env:BIOCHEM_PIN_MEMORY = "0"
    $env:BIOCHEM_LOSS_ISOLATE = "PASSIVE"
    $env:BIOCHEM_LOSS_DATA_ONLY = "0"
    $env:BIOCHEM_PASSIVE_DATA_KINE_WEIGHT = "0"
    $env:BIOCHEM_PASSIVE_DATA_BIO_WEIGHT = "1.0"
    $env:BIOCHEM_DATA_BIO_MASK_MODE = "clot_band"
    $env:BIOCHEM_DATA_BIO_FI_WEIGHT = "3.0"
    $env:BIOCHEM_DATA_BIO_MAT_WEIGHT = "2.0"
    $env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "0"
    $env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "0"
    $env:BIOCHEM_MU_LOG_WALL_WEIGHT = "0"
    $env:BIOCHEM_MU_LOG_HIGH_WEIGHT = "0"
    $env:BIOCHEM_VAL_TIME_STRIDE = "10"
    $env:BIOCHEM_TEACHER_VAL_EVERY = "3"
    $env:BIOCHEM_DETACH_MACRO_STATE = "0"
}

$ramp1Ckpt = Join-Path $RepoRoot "outputs\biochem\biochem_teacher_phaseB_ramp1_last.pth"
$best = Join-Path $RepoRoot "outputs\biochem\biochem_teacher_best_high_mu.pth"

if ($Ramp1Epochs -gt 0) {
    Write-Host "[NEW] Phase B ramp1 (data-only backward, ADR log-only)" -ForegroundColor Cyan
    Set-PhaseBPassiveEnv -RunNote "phaseB_XY_ramp1_data"
    $env:BIOCHEM_PASSIVE_ADR_BACKPROP = "0"
    $env:BIOCHEM_TEACHER_EPOCHS = "$Ramp1Epochs"
    $env:BIOCHEM_EPOCHS = "$Ramp1Epochs"
    $env:BIOCHEM_CLI_TEACHER_EPOCHS = "$Ramp1Epochs"

    python -m src.training.train_biochem_corrector --new --skip-pretrain --init-from-best --epochs $Ramp1Epochs --save-best --run-name phaseB_XY_ramp1_data
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $last = Join-Path $RepoRoot "outputs\biochem\biochem_teacher_last.pth"
    if (-not (Test-Path $last)) {
        Write-Host "[ERR] Missing $last after ramp1" -ForegroundColor Red
        exit 1
    }
    Copy-Item $last $ramp1Ckpt -Force
    Copy-Item $ramp1Ckpt $best -Force
} else {
    if (-not (Test-Path $ramp1Ckpt)) {
        Write-Host "[ERR] Ramp1Epochs=0 but missing $ramp1Ckpt (copy phaseB ramp1 ckpt first)" -ForegroundColor Red
        exit 1
    }
    Copy-Item $ramp1Ckpt $best -Force
    Write-Host "[i] Ramp1 skipped; using existing $ramp1Ckpt -> biochem_teacher_best_high_mu.pth" -ForegroundColor Cyan
}
Write-Host "[i] ramp2 init: $ramp1Ckpt -> biochem_teacher_best_high_mu.pth" -ForegroundColor Cyan

if ($Ramp2Epochs -gt 0) {
    Write-Host "[NEW] Phase B ramp2 (data + ADR in backward)" -ForegroundColor Cyan
    Set-PhaseBPassiveEnv -RunNote "phaseB_XY_ramp2_data_adr"
    $env:BIOCHEM_PASSIVE_ADR_BACKPROP = "1"
    # Raw ADR ~1e6 vs L_Data_Bio ~1e4; scale so both terms co-train (tune 1e-4 .. 1e-2).
    $env:BIOCHEM_PASSIVE_ADR_WEIGHT = "1e-3"
    $env:BIOCHEM_TEACHER_EPOCHS = "$Ramp2Epochs"
    $env:BIOCHEM_EPOCHS = "$Ramp2Epochs"
    $env:BIOCHEM_CLI_TEACHER_EPOCHS = "$Ramp2Epochs"

    python -m src.training.train_biochem_corrector --new --skip-pretrain --init-from-best --epochs $Ramp2Epochs --save-best --run-name phaseB_XY_ramp2_data_adr
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
    Write-Host "[skip] Ramp2Epochs=0; skipping ADR ramp (use go_m3_align_probe.ps1 for alignment test)" -ForegroundColor Cyan
}

Write-Host "[OK] Phase B passive ramp complete" -ForegroundColor Green
