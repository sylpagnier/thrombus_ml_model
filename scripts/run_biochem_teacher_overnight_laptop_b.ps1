# Laptop B - overnight step 2.5 TEACHER (step-2 + COMSOL temporal PhysTemp in backward).
# Same best-practice stack as laptop A, plus w_pt * L_PhysTemp on anchor trajectories.
# Next complexity after isolated-loss marathon (marathon: PhysTemp-alone only ~1.36 val mu).
#
# Usage (repo root, venv active):
#   .\scripts\run_biochem_teacher_overnight_laptop_b.ps1
#   .\scripts\run_biochem_teacher_overnight_laptop_b.ps1 -TeacherEpochs 20
#   .\scripts\run_biochem_teacher_overnight_laptop_b.ps1 -Variant step2_plus_musi   # alt: +W_MuSI=4, no PhysTemp
#
# Prereq: outputs\biochem\biochem_post_pretrain.pth
# Budget: ~3-5h (same as laptop A)
# Summary: outputs/reports/training/biochem/laptop_b_overnight_teacher_summary.txt

param(
    [ValidateSet("step25", "step2_plus_musi")]
    [string] $Variant = "step25",
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
    BIOCHEM_COMSOL_TEMPORAL_WEIGHT = "0.02"
    BIOCHEM_DATA_ONLY_PHYS_TEMP = "0"
    BIOCHEM_FI_GATE_START_WEIGHT = "0.0"
}

if ($Variant -eq "step25") {
    $Base["BIOCHEM_DATA_ONLY_PHYS_TEMP"] = "1"
    $label = "overnight_B_step25_teacher_phys_temp"
    $variantNote = "step-2 + W_MuLog=2 + w_pt*L_PhysTemp (DATA_ONLY_PHYS_TEMP=1)"
} else {
    $Base["BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT"] = "4.0"
    $label = "overnight_B_step2_plus_musi"
    $variantNote = "step-2 + W_MuLog=2 + W_MuSI=4 (no PhysTemp; reproduces marathon J2 coupling)"
}

$EarlyWindow = @{
    BIOCHEM_TBPTT_ANCHOR_END_BIAS = "0"
    BIOCHEM_TBPTT_ANCHOR_RANDOM_START = "0"
    BIOCHEM_TEACHER_TBPTT_RANDOM_ANCHOR = "0"
}

$LegDef = @{ Label = $label }

$SummaryPath = Join-Path $RepoRoot "outputs\reports\training\biochem\laptop_b_overnight_teacher_summary.txt"
$SummaryDir = Split-Path -Parent $SummaryPath
if (-not (Test-Path $SummaryDir)) { New-Item -ItemType Directory -Path $SummaryDir -Force | Out-Null }

Write-Host "Overnight B: $variantNote" -ForegroundColor Cyan
Write-Host "  epochs=$TeacherEpochs  TBPTT=6  DETACH=$Detach  variant=$Variant" -ForegroundColor DarkGray
if ($UseWarmStart) { Write-Host "Warm-start: $WarmStart" -ForegroundColor Yellow }

"overnight B started $(Get-Date -Format o) variant=$Variant ep=$TeacherEpochs" |
    Set-Content -Path $SummaryPath -Encoding utf8

Invoke-BiochemTeacherLeg -LegKey "B" -LegDef $LegDef -Base $Base -EarlyWindow $EarlyWindow `
    -UseWarmStart $UseWarmStart -LegIndex 1 -LegTotal 1 -SummaryPath $SummaryPath `
    -ExtraArgs $ExtraArgs -DryRun:$DryRun

Write-Host ""
Write-Host "Done. Compare to laptop A overnight (step-2 only). If val mu rises >0.05 vs A, revert PhysTemp/MuSI." -ForegroundColor Green
Write-Host "  $SummaryPath"
