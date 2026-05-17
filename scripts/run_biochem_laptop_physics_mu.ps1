# Laptop B — physics-informed mu hunt (~8h on ~4 GiB GPU).
# Step-2 joint supervision, temporal physics, and VISC isolate vs low-TF baseline (leg P).
#
# Usage (repo root):
#   .\scripts\run_biochem_laptop_physics_mu.ps1
#   .\scripts\run_biochem_laptop_physics_mu.ps1 -Legs @("P","S2","S25")
#   .\scripts\run_biochem_laptop_physics_mu.ps1 -Legs @("P","S2","S25","G0")   # optional gelation-off leg
#
# Default four legs (~2h each): P, S2, S25, V
#
# Success criteria (manual, after run):
#   P:  same as simple laptop (reproduce low-TF early-window MU_SI movement if possible).
#   S2/S25/V: val mu_log_mae improves vs P or vs ep0; compare wall and high-mu subsets.
#
# Logs per leg:
#   outputs/reports/training/biochem/<timestamp>/metrics.jsonl
#   outputs/reports/training/biochem/<timestamp>/training_diary_main.jsonl
# Summary: outputs/reports/training/biochem/laptop_physics_mu_summary.txt
#
# Optional leg G0 (gelation prior off): not in default 8h matrix; use -Legs to include.
# Stride=1 follow-up: if P improves with stride=10, rerun P with VAL_TIME_STRIDE=1 (~3h).

param(
    [string[]] $Legs = @("P", "S2", "S25", "V"),
    [int] $TeacherEpochs = 14,
    [switch] $ForcePretrain,
    [string[]] $ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$env:BIOCHEM_TRAIN_MODE = "new"

$WarmStart = Join-Path $RepoRoot "outputs\biochem\biochem_post_pretrain.pth"
$UseWarmStart = (-not $ForcePretrain) -and (Test-Path $WarmStart)

# 4 GiB laptop profile.
$TbpttWindow = "4"
$Rk4Sub = "12"
$DetachMacro = "1"

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
    BIOCHEM_DATA_ONLY_PHYS_TEMP = "0"
    BIOCHEM_COMSOL_TEMPORAL_WEIGHT = "0"
}

$EarlyWindow = @{
    BIOCHEM_TBPTT_ANCHOR_END_BIAS = "0"
    BIOCHEM_TBPTT_ANCHOR_RANDOM_START = "0"
    BIOCHEM_TEACHER_TBPTT_RANDOM_ANCHOR = "0"
}

$LegP = @{
    Label = "P_baseline_lowTF_MU_SI"
    BIOCHEM_LOSS_ISOLATE = "MU_SI"
    BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "8.0"
    BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "0.0"
}

$LegS2 = @{
    Label = "S2_joint_step2_lowTF"
    ClearLossIsolate = $true
    BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "8.0"
    BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "2.0"
}

$LegDefs = @{
    P = $LegP
    S2 = $LegS2
    S25 = @{
        Label = "S25_step2_plus_phys_temp"
        ClearLossIsolate = $true
        BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "8.0"
        BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "2.0"
        BIOCHEM_DATA_ONLY_PHYS_TEMP = "1"
        BIOCHEM_COMSOL_TEMPORAL_WEIGHT = "0.02"
    }
    V = @{
        Label = "V_VISC_isolate_lowTF"
        BIOCHEM_LOSS_ISOLATE = "VISC"
        BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "0.0"
        BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "0.0"
    }
    G0 = @{
        Label = "G0_gelation_gate_off_step2"
        ClearLossIsolate = $true
        BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "8.0"
        BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "2.0"
        BIOCHEM_GELATION_PRIOR_GATE = "0"
    }
}

$SummaryPath = Join-Path $RepoRoot "outputs\reports\training\biochem\laptop_physics_mu_summary.txt"
$SummaryDir = Split-Path -Parent $SummaryPath
if (-not (Test-Path $SummaryDir)) { New-Item -ItemType Directory -Path $SummaryDir -Force | Out-Null }

Write-Host "Laptop physics mu (~4 GiB): TBPTT=$TbpttWindow DETACH=$DetachMacro RK4=$Rk4Sub legs=$($Legs -join ',')" -ForegroundColor Cyan
if ($UseWarmStart) { Write-Host "Warm-start: $WarmStart" -ForegroundColor Yellow }
"laptop physics mu started $(Get-Date -Format o) tbptt=$TbpttWindow teacher_ep=$TeacherEpochs legs=$($Legs -join ',')" |
    Set-Content -Path $SummaryPath -Encoding utf8

function Clear-CudaCache {
    python -c "import gc; gc.collect(); import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>$null
}

$legIndex = 0
foreach ($leg in $Legs) {
    $key = $leg.ToUpper()
    if (-not $LegDefs.ContainsKey($key)) {
        Write-Warning "Unknown leg '$leg'; skip. Valid: P,S2,S25,V,G0"
        continue
    }
    $legIndex++
    $def = $LegDefs[$key]
    $label = $def.Label

    if ($legIndex -gt 1) { Clear-CudaCache }

    Write-Host ""
    Write-Host "========== Leg $key : $label ($legIndex / $($Legs.Count)) ==========" -ForegroundColor Cyan

    Get-ChildItem Env:BIOCHEM_* -ErrorAction SilentlyContinue | ForEach-Object {
        Remove-Item "Env:$($_.Name)" -ErrorAction SilentlyContinue
    }
    foreach ($k in $Base.Keys) { Set-Item -Path "Env:$k" -Value $Base[$k] }
    foreach ($k in $EarlyWindow.Keys) { Set-Item -Path "Env:$k" -Value $EarlyWindow[$k] }

    if ($def.ClearLossIsolate) {
        Remove-Item Env:BIOCHEM_LOSS_ISOLATE -ErrorAction SilentlyContinue
    }

    foreach ($k in $def.Keys) {
        if ($k -in @("Label", "ClearLossIsolate")) { continue }
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
        $line = "Leg $key FAILED exit=$LASTEXITCODE at $(Get-Date -Format o) note=$label"
        Add-Content -Path $SummaryPath -Value $line
        Write-Host $line -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Clear-CudaCache
    $dt = (Get-Date) - $t0
    $line = "Leg $key OK $label duration=$([int]$dt.TotalMinutes)m note=$label"
    Add-Content -Path $SummaryPath -Value $line
    Write-Host $line -ForegroundColor Green
}

Write-Host ""
Write-Host "Done. Summary: $SummaryPath"
Write-Host "Compare P vs S2 vs S25 vs V: val mu_log_mae (all / wall / high-mu) and mu_pearson."
