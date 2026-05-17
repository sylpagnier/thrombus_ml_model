# Laptop A — simple mu hunt (~8h on ~4 GiB GPU).
# Capacity probes (1-anchor overfit, mu-only backward) + reproduction of the low-TF winner.
#
# Usage (repo root):
#   .\scripts\run_biochem_laptop_simple_mu.ps1
#   .\scripts\run_biochem_laptop_simple_mu.ps1 -Legs @("P","O1")
#
# Default four legs (~2h each): O1, O2, P, B
#
# Success criteria (manual, after run):
#   O1/O2: train L_Back / L_tot drops >=10x from ep0 on the single graph (see metrics.jsonl).
#   P/B:   val mu_log_mae drops >=0.01 vs ep0 (all), or wall/high-mu improves with stable mu_pearson.
#
# Logs per leg:
#   outputs/reports/training/biochem/<timestamp>/metrics.jsonl
#   outputs/reports/training/biochem/<timestamp>/training_diary_main.jsonl
# Summary: outputs/reports/training/biochem/laptop_simple_mu_summary.txt
#
# Stride=1 follow-up (if P moves with stride=10): run leg P only with
#   $env:BIOCHEM_VAL_TIME_STRIDE='1'; $env:BIOCHEM_TEACHER_VAL_EVERY='6' (~3h extra).

param(
    [string[]] $Legs = @("O1", "O2", "P", "B"),
    [int] $TeacherEpochs = 14,
    [int] $OverfitEpochs = 24,
    [switch] $ForcePretrain,
    [string[]] $ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$env:BIOCHEM_TRAIN_MODE = "new"

$WarmStart = Join-Path $RepoRoot "outputs\biochem\biochem_post_pretrain.pth"
$UseWarmStart = (-not $ForcePretrain) -and (Test-Path $WarmStart)

# 4 GiB laptop profile (matches run_biochem_mu_isolate_sweep.ps1).
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
}

$OverfitCommon = @{
    BIOCHEM_MAX_LOAD_VESSELS = "1"
    BIOCHEM_MAX_LOAD_SHUFFLE = "0"
    BIOCHEM_LOW_ANCHOR_MODE = "1"
    BIOCHEM_TEACHER_SKIP_VAL = "1"
    BIOCHEM_TBPTT_ANCHOR_END_BIAS = "0"
    BIOCHEM_TBPTT_ANCHOR_RANDOM_START = "0"
    BIOCHEM_TEACHER_TBPTT_RANDOM_ANCHOR = "0"
}

$EarlyWindow = @{
    BIOCHEM_TBPTT_ANCHOR_END_BIAS = "0"
    BIOCHEM_TBPTT_ANCHOR_RANDOM_START = "0"
    BIOCHEM_TEACHER_TBPTT_RANDOM_ANCHOR = "0"
}

$LegDefs = @{
    O1 = @{
        Label = "O1_overfit_1anchor_MU_LOG"
        UseOverfitEpochs = $true
        BIOCHEM_LOSS_ISOLATE = "MU_LOG"
        BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "0.0"
        BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "2.0"
    }
    O2 = @{
        Label = "O2_overfit_1anchor_MU_SI"
        UseOverfitEpochs = $true
        BIOCHEM_LOSS_ISOLATE = "MU_SI"
        BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "8.0"
        BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "0.0"
    }
    P = @{
        Label = "P_repro_lowTF_earlywin_MU_SI"
        BIOCHEM_LOSS_ISOLATE = "MU_SI"
        BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "8.0"
        BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "0.0"
    }
    B = @{
        Label = "B_MU_LOG_earlywin"
        BIOCHEM_LOSS_ISOLATE = "MU_LOG"
        BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "0.0"
        BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "2.0"
    }
}

$SummaryPath = Join-Path $RepoRoot "outputs\reports\training\biochem\laptop_simple_mu_summary.txt"
$SummaryDir = Split-Path -Parent $SummaryPath
if (-not (Test-Path $SummaryDir)) { New-Item -ItemType Directory -Path $SummaryDir -Force | Out-Null }

Write-Host "Laptop simple mu (~4 GiB): TBPTT=$TbpttWindow DETACH=$DetachMacro RK4=$Rk4Sub legs=$($Legs -join ',')" -ForegroundColor Cyan
if ($UseWarmStart) { Write-Host "Warm-start: $WarmStart" -ForegroundColor Yellow }
"laptop simple mu started $(Get-Date -Format o) tbptt=$TbpttWindow teacher_ep=$TeacherEpochs overfit_ep=$OverfitEpochs legs=$($Legs -join ',')" |
    Set-Content -Path $SummaryPath -Encoding utf8

function Clear-CudaCache {
    python -c "import gc; gc.collect(); import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>$null
}

$legIndex = 0
foreach ($leg in $Legs) {
    $key = $leg.ToUpper()
    if (-not $LegDefs.ContainsKey($key)) {
        Write-Warning "Unknown leg '$leg'; skip. Valid: O1,O2,P,B"
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

    if ($def.UseOverfitEpochs) {
        foreach ($k in $OverfitCommon.Keys) { Set-Item -Path "Env:$k" -Value $OverfitCommon[$k] }
        $env:BIOCHEM_TEACHER_EPOCHS = "$OverfitEpochs"
    }

    foreach ($k in $def.Keys) {
        if ($k -in @("Label", "UseOverfitEpochs")) { continue }
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
Write-Host "Compare newest outputs/reports/training/biochem/<timestamp>/ for val mu_log_mae (all / wall / high-mu) and train L_Back."
