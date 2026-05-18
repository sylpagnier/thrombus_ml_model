# Laptop A — teacher complexity marathon (~3h): supervised / μ losses, then combine.
# Order: isolate each anchor loss → joint step-2 (add terms one at a time).
#
# Run on LAPTOP A only (LAPTOP B runs run_biochem_teacher_complexity_laptop_b.ps1 in parallel).
#
# Usage (repo root, venv active):
#   .\scripts\run_biochem_teacher_complexity_laptop_a.ps1
#   .\scripts\run_biochem_teacher_complexity_laptop_a.ps1 -ListLegs
#   .\scripts\run_biochem_teacher_complexity_laptop_a.ps1 -Legs @("I1","J1","J2") -DryRun
#
# Prereq: outputs\biochem\biochem_post_pretrain.pth (or first leg runs AE+ODE pretrain).
# Budget: ~3h with warm-start, full anchors, VAL_TIME_STRIDE=10, val every 3 ep.
# Logs: outputs/reports/training/biochem/<timestamp>/metrics.jsonl
# Summary: outputs/reports/training/biochem/laptop_a_teacher_complexity_summary.txt
#
# Pass (manual): I1 val mu_log_mae drops vs ep0; I3 train L_Data_Bio down; I4 L_Data_Kine down;
#   J1/J2: val mu not worse than ~1.5 vs I1 (joint is harder than MU_LOG isolate).

param(
    [string[]] $Legs = @("I1", "I2", "I3", "I4", "J1", "J2"),
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
    I1 = "MU_LOG isolate + mu-path (reference; 8 ep)"
    I2 = "MU_SI isolate (5 ep) — expect weak val μ"
    I3 = "DATA_BIO isolate (5 ep) — species only"
    I4 = "DATA_KINE isolate (5 ep) — u,v,p,mu_nd"
    J1 = "Joint: L_Data_Kine + W_MuLog·L_MuLog (no isolate; 7 ep)"
    J2 = "Joint step-2: + W_MuSI·L_MuSI (7 ep)"
}

if ($ListLegs) {
    Write-Host "Laptop A — teacher complexity (supervision track):" -ForegroundColor Cyan
    foreach ($k in ($LegCatalog.Keys | Sort-Object)) { Write-Host "  $k  $($LegCatalog[$k])" }
    Write-Host ""
    Write-Host "Default order: $(($LegCatalog.Keys | Sort-Object) -join ', ')"
    Write-Host "Pair with: .\scripts\run_biochem_teacher_complexity_laptop_b.ps1 on the other machine."
    exit 0
}

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
    BIOCHEM_TEACHER_EPOCHS = "6"
    BIOCHEM_TEACHER_VAL_EVERY = "3"
    BIOCHEM_VAL_TIME_STRIDE = "10"
    BIOCHEM_TBPTT_MAX_WINDOW = "4"
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
    BIOCHEM_DATA_ONLY_PHYS_TEMP = "0"
}

$EarlyWindow = @{
    BIOCHEM_TBPTT_ANCHOR_END_BIAS = "0"
    BIOCHEM_TBPTT_ANCHOR_RANDOM_START = "0"
    BIOCHEM_TEACHER_TBPTT_RANDOM_ANCHOR = "0"
}

$LegDefs = @{
    I1 = @{
        Label = "A_I1_MU_LOG_mu_path"
        TeacherEpochs = 8
        BIOCHEM_LOSS_ISOLATE = "MU_LOG"
        BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "2.0"
        BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "0.0"
    }
    I2 = @{
        Label = "A_I2_MU_SI"
        TeacherEpochs = 5
        BIOCHEM_LOSS_ISOLATE = "MU_SI"
        BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "0.0"
        BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "8.0"
    }
    I3 = @{
        Label = "A_I3_DATA_BIO"
        TeacherEpochs = 5
        BIOCHEM_LOSS_ISOLATE = "DATA_BIO"
        BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "0.0"
        BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "0.0"
    }
    I4 = @{
        Label = "A_I4_DATA_KINE"
        TeacherEpochs = 5
        BIOCHEM_LOSS_ISOLATE = "DATA_KINE"
        BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "0.0"
        BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "0.0"
    }
    J1 = @{
        Label = "A_J1_joint_kine_plus_mu_log"
        TeacherEpochs = 7
        ClearLossIsolate = $true
        BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "2.0"
        BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "0.0"
    }
    J2 = @{
        Label = "A_J2_joint_step2_mu_log_plus_mu_si"
        TeacherEpochs = 7
        ClearLossIsolate = $true
        BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "2.0"
        BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "4.0"
    }
}

$SummaryPath = Join-Path $RepoRoot "outputs\reports\training\biochem\laptop_a_teacher_complexity_summary.txt"
$SummaryDir = Split-Path -Parent $SummaryPath
if (-not (Test-Path $SummaryDir)) { New-Item -ItemType Directory -Path $SummaryDir -Force | Out-Null }

Write-Host "Laptop A teacher complexity (~3h target) | legs=$($Legs -join ',') | OOM-safe=$OomSafe" -ForegroundColor Cyan
if ($UseWarmStart) { Write-Host "Warm-start: $WarmStart" -ForegroundColor Yellow }
else { Write-Host "No warm-start — leg I1 will run AE+ODE pretrain (longer than 3h)." -ForegroundColor Red }

"laptop A started $(Get-Date -Format o) legs=$($Legs -join ',') oom_safe=$OomSafe" |
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
Write-Host "Laptop A done. Summary: $SummaryPath" -ForegroundColor Green
Write-Host "Compare metrics.jsonl: val mu_log_mae (all | wall | high-mu), train L_Back / L_Data_* per leg."
