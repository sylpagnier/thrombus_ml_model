# Laptop B - teacher complexity marathon (~3h): aux / physics isolates, TBPTT widen, temporal.
# Order: isolate PhysTemp & ADR -> MU_LOG with longer TBPTT -> combine mu + temporal.
#
# Run on LAPTOP B only (LAPTOP A runs run_biochem_teacher_complexity_laptop_a.ps1 in parallel).
#
# Usage (repo root, venv active):
#   .\scripts\run_biochem_teacher_complexity_laptop_b.ps1
#   .\scripts\run_biochem_teacher_complexity_laptop_b.ps1 -ListLegs
#   .\scripts\run_biochem_teacher_complexity_laptop_b.ps1 -OomSafe 0   # >=8GB: T2 uses DETACH=0
#
# Prereq: outputs\biochem\biochem_post_pretrain.pth
# Summary: outputs/reports/training/biochem/laptop_b_teacher_complexity_summary.txt
#
# Pass (manual): I5/I6 train loss finite & trends down; T1/T2 val mu vs ep0 (MU_LOG);
#   J3 step-2+PhysTemp: val mu not >>0.05 worse than T1 best.

param(
    [string[]] $Legs = @("I5", "I6", "T1", "T2", "J3"),
    [switch] $ListLegs,
    [switch] $DryRun,
    [switch] $ForcePretrain,
    [int] $OomSafe = 1,
    [string[]] $ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_biochem_teacher_complexity_common.ps1")

$LegCatalog = @{
    I5 = "PHYS_TEMP isolate (COMSOL d/dt Huber; 5 ep)"
    I6 = "ADR_F isolate (fast ADR residual; 5 ep)"
    T1 = "MU_LOG + TBPTT=5 (5GB-safe temporal; 7 ep)"
    T2 = "MU_LOG + TBPTT=6, DETACH=1 (7 ep); use -OomSafe 0 for DETACH=0 on 8GB+"
    T3 = "MU_LOG + TBPTT=8 (6 ep) - skip if over budget"
    J3 = "MU_LOG isolate + DATA_ONLY_PHYS_TEMP (D3 coupling; 7 ep)"
    J4 = "Full step-2 + PhysTemp (7 ep)"
}

if ($ListLegs) {
    Write-Host "Laptop B - teacher complexity (physics / temporal track):" -ForegroundColor Cyan
    foreach ($k in ($LegCatalog.Keys | Sort-Object)) { Write-Host "  $k  $($LegCatalog[$k])" }
    Write-Host ""
    Write-Host "Default: I5,I6,T1,T2,J3 (~3h). Add T3 or J4 only if finishing early."
    Write-Host "Pair with: .\scripts\run_biochem_teacher_complexity_laptop_a.ps1 on the other machine."
    exit 0
}

$WarmStart = Join-Path $RepoRoot "outputs\biochem\biochem_post_pretrain.pth"
$UseWarmStart = (-not $ForcePretrain) -and (Test-Path $WarmStart)

$DetachDefault = if ($OomSafe -ne 0) { "1" } else { "0" }
$Rk4Default = if ($OomSafe -ne 0) { "10" } else { "12" }

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
    BIOCHEM_TEACHER_EPOCHS = "6"
    BIOCHEM_TEACHER_VAL_EVERY = "3"
    BIOCHEM_VAL_TIME_STRIDE = "10"
    BIOCHEM_TBPTT_MAX_WINDOW = "4"
    BIOCHEM_TBPTT_WINDOW_CURRICULUM = "0"
    BIOCHEM_DETACH_MACRO_STATE = $DetachDefault
    BIOCHEM_ADJOINT_RK4_SUBSTEPS = $Rk4Default
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
}

$EarlyWindow = @{
    BIOCHEM_TBPTT_ANCHOR_END_BIAS = "0"
    BIOCHEM_TBPTT_ANCHOR_RANDOM_START = "0"
    BIOCHEM_TEACHER_TBPTT_RANDOM_ANCHOR = "0"
}

$T2Detach = if ($OomSafe -ne 0) { "1" } else { "0" }
$T2Rk4 = if ($OomSafe -ne 0) { "10" } else { "12" }

$LegDefs = @{
    I5 = @{
        Label = "B_I5_PHYS_TEMP"
        TeacherEpochs = 5
        BIOCHEM_LOSS_ISOLATE = "PHYS_TEMP"
        BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "0.0"
        BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "0.0"
    }
    I6 = @{
        Label = "B_I6_ADR_F"
        TeacherEpochs = 5
        BIOCHEM_LOSS_ISOLATE = "ADR_F"
        BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "0.0"
        BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "0.0"
    }
    T1 = @{
        Label = "B_T1_MU_LOG_TBPTT5"
        TeacherEpochs = 7
        BIOCHEM_LOSS_ISOLATE = "MU_LOG"
        BIOCHEM_TBPTT_MAX_WINDOW = "5"
        BIOCHEM_ADJOINT_RK4_SUBSTEPS = "10"
    }
    T2 = @{
        Label = "B_T2_MU_LOG_TBPTT6"
        TeacherEpochs = 7
        BIOCHEM_LOSS_ISOLATE = "MU_LOG"
        BIOCHEM_TBPTT_MAX_WINDOW = "6"
        BIOCHEM_DETACH_MACRO_STATE = $T2Detach
        BIOCHEM_ADJOINT_RK4_SUBSTEPS = $T2Rk4
    }
    T3 = @{
        Label = "B_T3_MU_LOG_TBPTT8"
        TeacherEpochs = 6
        BIOCHEM_LOSS_ISOLATE = "MU_LOG"
        BIOCHEM_TBPTT_MAX_WINDOW = "8"
        BIOCHEM_DETACH_MACRO_STATE = "1"
        BIOCHEM_ADJOINT_RK4_SUBSTEPS = "10"
    }
    J3 = @{
        Label = "B_J3_MU_LOG_plus_phys_temp"
        TeacherEpochs = 7
        BIOCHEM_LOSS_ISOLATE = "MU_LOG"
        BIOCHEM_DATA_ONLY_PHYS_TEMP = "1"
    }
    J4 = @{
        Label = "B_J4_joint_step2_plus_phys_temp"
        TeacherEpochs = 7
        ClearLossIsolate = $true
        BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "2.0"
        BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "4.0"
        BIOCHEM_DATA_ONLY_PHYS_TEMP = "1"
    }
}

if ($OomSafe -eq 0) {
    Write-Host "OomSafe=0: T2 uses DETACH_MACRO=0 (needs ~8GB VRAM)." -ForegroundColor Yellow
}

$SummaryPath = Join-Path $RepoRoot "outputs\reports\training\biochem\laptop_b_teacher_complexity_summary.txt"
$SummaryDir = Split-Path -Parent $SummaryPath
if (-not (Test-Path $SummaryDir)) { New-Item -ItemType Directory -Path $SummaryDir -Force | Out-Null }

Write-Host "Laptop B teacher complexity (~3h target) | legs=$($Legs -join ',') | OOM-safe=$OomSafe" -ForegroundColor Cyan
if ($UseWarmStart) { Write-Host "Warm-start: $WarmStart" -ForegroundColor Yellow }
else { Write-Host "No warm-start - first leg runs pretrain (may exceed 3h)." -ForegroundColor Red }

"laptop B started $(Get-Date -Format o) legs=$($Legs -join ',') oom_safe=$OomSafe" |
    Set-Content -Path $SummaryPath -Encoding utf8

$legIndex = 0
$legTotal = $Legs.Count
foreach ($leg in $Legs) {
    $key = $leg.ToUpper()
    if (-not $LegDefs.ContainsKey($key)) {
        Write-Warning "Unknown leg '$leg'; skip. Use -ListLegs."
        continue
    }
    $legIndex++
    Invoke-BiochemTeacherLeg -LegKey $key -LegDef $LegDefs[$key] -Base $Base -EarlyWindow $EarlyWindow `
        -UseWarmStart $UseWarmStart -LegIndex $legIndex -LegTotal $legTotal -SummaryPath $SummaryPath `
        -ExtraArgs $ExtraArgs -DryRun:$DryRun
}

Write-Host ""
Write-Host "Laptop B done. Summary: $SummaryPath" -ForegroundColor Green
Write-Host "If T2 OOMs on 5GB GPU, rerun with default -OomSafe 1 or drop T2 and keep T1+J3."
