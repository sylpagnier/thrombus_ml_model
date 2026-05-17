# μ-isolate sweep (~8h on 4 GiB GPU): teacher-only, backward = μ loss only.
# Each leg starts a NEW run (--new), reuses post-pretrain warm-start after leg 1.
#
# Usage (repo root):
#   .\scripts\run_biochem_mu_isolate_sweep.ps1
#   .\scripts\run_biochem_mu_isolate_sweep.ps1 -Legs @("A","B")   # subset
#
# After completion, compare val mu_log_mae (all / wall / high-mu_gt) and r in:
#   outputs/reports/training/biochem/<timestamp>/training_diary_main.jsonl
#
# Leg matrix (default all four ≈ 8h):
#   A  joint μ (SI+log), late TBPTT (end-bias)     — primary reference
#   B  log-MAE only (MU_LOG isolate)               — matches val metric
#   C  SI Huber only (MU_SI, no log term)        — scale vs log
#   D  early-window control (no end-bias)          — temporal hypothesis

param(
    [string[]] $Legs = @("A", "B", "C", "D"),
    [int] $TeacherEpochs = 14,
    [switch] $HighVram,
    [switch] $ForcePretrain,
    [string[]] $ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# expandable_segments is Linux-only; harmless warning on Windows.
if ($HighVram) {
    $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
}
$env:BIOCHEM_TRAIN_MODE = "new"

$WarmStart = Join-Path $RepoRoot "outputs\biochem\biochem_post_pretrain.pth"
$UseWarmStart = (-not $ForcePretrain) -and (Test-Path $WarmStart)

# 4 GiB defaults: short TBPTT, no t0→start_idx warmup when end-bias (see train_biochem_corrector.py).
$LaptopMode = -not $HighVram
$TbpttWindow = if ($LaptopMode) { "4" } else { "6" }
$Rk4Sub = if ($LaptopMode) { "12" } else { "16" }
$DetachMacro = if ($LaptopMode) { "1" } else { "0" }

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
    BIOCHEM_TEACHER_FORCE_MIN = "0.0"
    BIOCHEM_TEACHER_TF_WARMUP_EPOCHS = "4"
    BIOCHEM_TEACHER_EPOCHS = "$TeacherEpochs"
    BIOCHEM_TEACHER_VAL_EVERY = "2"
    BIOCHEM_VAL_TIME_STRIDE = "10"
    BIOCHEM_TBPTT_MAX_WINDOW = $TbpttWindow
    BIOCHEM_TBPTT_WINDOW_CURRICULUM = "0"
    BIOCHEM_DETACH_MACRO_STATE = $DetachMacro
    BIOCHEM_ADJOINT_RK4_SUBSTEPS = $Rk4Sub
    BIOCHEM_DATALOADER_WORKERS = "0"
    BIOCHEM_MU_SI_MULTI_STEP = "1"
    BIOCHEM_MU_SI_HUBER_DELTA = "0.25"
    BIOCHEM_MU_ANCHOR_LATE_TIME_WEIGHT = "1"
    BIOCHEM_MU_LATE_TIME_POWER = "2.0"
    BIOCHEM_FI_GATE_START_WEIGHT = "0.0"
    BIOCHEM_TEACHER_SKIP_VAL = "0"
    BIOCHEM_DEBUG = "0"
}

if ($UseWarmStart) {
    Write-Host "Using existing warm-start: $WarmStart (skip AE/ODE-RXN on leg 1)." -ForegroundColor Yellow
}
if ($LaptopMode) {
    Write-Host "Laptop VRAM mode: TBPTT=$TbpttWindow DETACH_MACRO=$DetachMacro RK4=$Rk4Sub (end-bias skips long warmup rollforward)." -ForegroundColor Yellow
}

$LegDefs = @{
    A = @{
        Label = "A_joint_mu_endbias"
        BIOCHEM_LOSS_ISOLATE = "MU_SI"
        BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "8.0"
        BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "2.0"
        BIOCHEM_TBPTT_ANCHOR_END_BIAS = "1"
        BIOCHEM_TBPTT_ANCHOR_RANDOM_START = "0"
        BIOCHEM_TEACHER_TBPTT_RANDOM_ANCHOR = "0"
    }
    B = @{
        Label = "B_mu_log_only"
        BIOCHEM_LOSS_ISOLATE = "MU_LOG"
        BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "0.0"
        BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "2.0"
        BIOCHEM_TBPTT_ANCHOR_END_BIAS = "1"
        BIOCHEM_TBPTT_ANCHOR_RANDOM_START = "0"
        BIOCHEM_TEACHER_TBPTT_RANDOM_ANCHOR = "0"
    }
    C = @{
        Label = "C_mu_si_only"
        BIOCHEM_LOSS_ISOLATE = "MU_SI"
        BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "8.0"
        BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "0.0"
        BIOCHEM_TBPTT_ANCHOR_END_BIAS = "1"
        BIOCHEM_TBPTT_ANCHOR_RANDOM_START = "0"
        BIOCHEM_TEACHER_TBPTT_RANDOM_ANCHOR = "0"
    }
    D = @{
        Label = "D_early_window_control"
        BIOCHEM_LOSS_ISOLATE = "MU_SI"
        BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "8.0"
        BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "2.0"
        BIOCHEM_TBPTT_ANCHOR_END_BIAS = "0"
        BIOCHEM_TBPTT_ANCHOR_RANDOM_START = "0"
        BIOCHEM_TEACHER_TBPTT_RANDOM_ANCHOR = "0"
    }
}

$OptionalE = @{
    Label = "E_random_window"
    BIOCHEM_LOSS_ISOLATE = "MU_SI"
    BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "8.0"
    BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "2.0"
    BIOCHEM_TBPTT_ANCHOR_END_BIAS = "0"
    BIOCHEM_TBPTT_ANCHOR_RANDOM_START = "1"
    BIOCHEM_TEACHER_TBPTT_RANDOM_ANCHOR = "1"
}
$LegDefs["E"] = $OptionalE

$SummaryPath = Join-Path $RepoRoot "outputs\reports\training\biochem\mu_isolate_sweep_summary.txt"
$SummaryDir = Split-Path -Parent $SummaryPath
if (-not (Test-Path $SummaryDir)) { New-Item -ItemType Directory -Path $SummaryDir -Force | Out-Null }
"μ-isolate sweep started $(Get-Date -Format o)" | Set-Content -Path $SummaryPath -Encoding utf8

$legIndex = 0
foreach ($leg in $Legs) {
    $key = $leg.ToUpper()
    if (-not $LegDefs.ContainsKey($key)) {
        Write-Warning "Unknown leg '$leg'; skip. Valid: A,B,C,D,E"
        continue
    }
    $legIndex++
    $def = $LegDefs[$key]
    $label = $def.Label

    Write-Host ""
    Write-Host "========== Leg $key : $label ($legIndex / $($Legs.Count)) ==========" -ForegroundColor Cyan

  # Clear stale env from prior legs / presets
    Get-ChildItem Env:BIOCHEM_* | ForEach-Object { Remove-Item "Env:$($_.Name)" -ErrorAction SilentlyContinue }
    foreach ($k in $Base.Keys) { Set-Item -Path "Env:$k" -Value $Base[$k] }
    foreach ($k in $def.Keys) {
        if ($k -eq "Label") { continue }
        Set-Item -Path "Env:$k" -Value $def[$k]
    }
    $env:BIOCHEM_RUN_NOTE = $label

    if ($legIndex -gt 1 -or $UseWarmStart) {
        $env:BIOCHEM_SKIP_PRETRAIN = "1"
        $env:BIOCHEM_REUSE_LAST_PRETRAIN = "1"
    }

    $t0 = Get-Date
    python -m src.training.train_biochem_corrector --new @ExtraArgs
    if ($LASTEXITCODE -ne 0) {
        $line = "Leg $key FAILED (exit $LASTEXITCODE) at $(Get-Date -Format o)"
        Add-Content -Path $SummaryPath -Value $line
        Write-Host $line -ForegroundColor Red
        break
    }
    $dt = (Get-Date) - $t0
    $line = "Leg $key OK $label duration=$([int]$dt.TotalMinutes)m note=$($env:BIOCHEM_RUN_NOTE)"
    Add-Content -Path $SummaryPath -Value $line
    Write-Host $line -ForegroundColor Green
}

Write-Host ""
Write-Host "Sweep done. Summary: $SummaryPath"
Write-Host "Compare latest folders under outputs/reports/training/biochem/ for validation mu_log_mae lines."
