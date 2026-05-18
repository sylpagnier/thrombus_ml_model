# Laptop A - overnight best-practice TEACHER stage (patient007 val).
# Distills marathon + A0 lessons: step-2 data-only, W_MuLog=2, mu-path, low TF, TBPTT=6.
# No LOSS_ISOLATE (full supervised teacher objective for one long run).
#
# Usage (repo root, venv active):
#   .\scripts\run_biochem_teacher_overnight_laptop_a.ps1
#   .\scripts\run_biochem_teacher_overnight_laptop_a.ps1 -TeacherEpochs 20 -DryRun
#
# Prereq: outputs\biochem\biochem_post_pretrain.pth
# Budget: ~3-5h (18 ep, val every 2, stride=10 on patient007 ~11 min/val)
# Writes: outputs/biochem/biochem_teacher_best.pth
# Summary: outputs/reports/training/biochem/laptop_a_overnight_teacher_summary.txt

param(
    [int] $TeacherEpochs = 18,
    [int] $OomSafe = 1,
    [switch] $DryRun,
    [switch] $ForcePretrain,
    [string[]] $ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_biochem_teacher_complexity_common.ps1")

$WarmStart = Join-Path $RepoRoot "outputs\biochem\biochem_post_pretrain.pth"
$UseWarmStart = (-not $ForcePretrain) -and (Test-Path $WarmStart)

$Detach = if ($OomSafe -ne 0) { "1" } else { "0" }
$Rk4 = if ($OomSafe -ne 0) { "10" } else { "12" }

$Base = @{
    BIOCHEM_STOCK_DEFAULTS = "1"
    BIOCHEM_SKIP_PRETRAIN = if ($UseWarmStart) { "1" } else { "0" }
    BIOCHEM_REUSE_LAST_PRETRAIN = if ($UseWarmStart) { "1" } else { "0" }
    BIOCHEM_STOP_AFTER_TEACHER = "1"
    BIOCHEM_NO_TEACHER_DEFAULTS = "1"
    BIOCHEM_LOSS_DATA_ONLY = "1"
    BIOCHEM_PRESET = ""
    BIOCHEM_COMPLEXITY_STEP = "2"
    BIOCHEM_TEACHER_MU_RATIO_MAX = "20.0"
    BIOCHEM_TEACHER_FORCE_MIN = "0.0"
    BIOCHEM_TEACHER_TF_WARMUP_EPOCHS = "4"
    BIOCHEM_TEACHER_EPOCHS = "$TeacherEpochs"
    BIOCHEM_TEACHER_VAL_EVERY = "2"
    BIOCHEM_VAL_TIME_STRIDE = "10"
    BIOCHEM_TBPTT_MAX_WINDOW = "6"
    BIOCHEM_TBPTT_WINDOW_CURRICULUM = "0"
    BIOCHEM_DETACH_MACRO_STATE = $Detach
    BIOCHEM_ADJOINT_RK4_SUBSTEPS = $Rk4
    BIOCHEM_DATALOADER_WORKERS = "0"
    BIOCHEM_TEACHER_SKIP_VAL = "0"
    BIOCHEM_DEBUG = "0"
    BIOCHEM_TRAIN_MU_ENCODER = "1"
    BIOCHEM_USE_MU_PATH_GROUP = "1"
    BIOCHEM_USE_DELTA_MU_HEAD = "1"
    BIOCHEM_DELTA_MU_LOG_CLIP = "2.0"
    BIOCHEM_MU_SI_MULTI_STEP = "1"
    BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "2.0"
    BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "0.0"
    BIOCHEM_DATA_ONLY_PHYS_TEMP = "0"
    BIOCHEM_FI_GATE_START_WEIGHT = "0.0"
}

$EarlyWindow = @{
    BIOCHEM_TBPTT_ANCHOR_END_BIAS = "0"
    BIOCHEM_TBPTT_ANCHOR_RANDOM_START = "0"
    BIOCHEM_TEACHER_TBPTT_RANDOM_ANCHOR = "0"
}

$LegDef = @{
    Label = "overnight_A_best_practice_teacher_step2"
}

$SummaryPath = Join-Path $RepoRoot "outputs\reports\training\biochem\laptop_a_overnight_teacher_summary.txt"
$SummaryDir = Split-Path -Parent $SummaryPath
if (-not (Test-Path $SummaryDir)) { New-Item -ItemType Directory -Path $SummaryDir -Force | Out-Null }

Write-Host "Overnight A: best-practice teacher (step-2, TBPTT=6, W_MuLog=2)" -ForegroundColor Cyan
Write-Host "  epochs=$TeacherEpochs  val_every=2  stride=10  DETACH=$Detach  OOM-safe=$OomSafe" -ForegroundColor DarkGray
if ($UseWarmStart) { Write-Host "Warm-start: $WarmStart" -ForegroundColor Yellow }
else { Write-Host "No warm-start - will run AE+ODE pretrain first (much longer)." -ForegroundColor Red }

"overnight A started $(Get-Date -Format o) ep=$TeacherEpochs" | Set-Content -Path $SummaryPath -Encoding utf8

Invoke-BiochemTeacherLeg -LegKey "A" -LegDef $LegDef -Base $Base -EarlyWindow $EarlyWindow `
    -UseWarmStart $UseWarmStart -LegIndex 1 -LegTotal 1 -SummaryPath $SummaryPath `
    -ExtraArgs $ExtraArgs -DryRun:$DryRun

Write-Host ""
Write-Host "Done. Check val mu_log_mae (all | wall | high-mu) and:" -ForegroundColor Green
Write-Host "  outputs/biochem/biochem_teacher_best.pth"
Write-Host "  $SummaryPath"
