# Fair wall-gate A/B sweep (~2.5–3.5 h per laptop, unattended).
# Split across two machines — run ONE command per laptop from repo root:
#
#   Laptop A (baseline arm):
#     .\scripts\run_biochem_wall_gate_fair_sweep_3h.ps1 -Arm A
#
#   Laptop B (sweep_free_wall_a arm + related presets):
#     .\scripts\run_biochem_wall_gate_fair_sweep_3h.ps1 -Arm B
#
# Dry run:
#     .\scripts\run_biochem_wall_gate_fair_sweep_3h.ps1 -Arm A -DryRun
#
# After both finish, compare:
#   outputs\reports\training\biochem\wall_gate_fair_sweep_3h_summary.txt
#   latest run folders under outputs\reports\training\biochem\
#
param(
    [ValidateSet("A", "B")]
    [string] $Arm = "A",
    [switch] $ForcePretrain,
    [switch] $DryRun,
    [string[]] $ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$hostName = $env:COMPUTERNAME
$SummaryPath = Join-Path $RepoRoot "outputs\reports\training\biochem\wall_gate_fair_sweep_3h_summary.txt"
$SummaryDir = Split-Path -Parent $SummaryPath
if (-not (Test-Path $SummaryDir)) { New-Item -ItemType Directory -Path $SummaryDir -Force | Out-Null }

$WarmStart = Join-Path $RepoRoot "outputs\biochem\biochem_post_pretrain.pth"
$batchStart = Get-Date
$batchIso = $batchStart.ToString("o")

# Epoch ladder (~8–34 ep teacher); tuned for ~3 h with warm-start on 4 GB GPUs.
$EpochLadder = @(8, 14, 20, 26, 30, 34)

function Set-FairBase {
    param([int]$Ep, [int]$ValEvery = 2)

    Get-ChildItem Env:BIOCHEM_* | ForEach-Object { Remove-Item "Env:$($_.Name)" -ErrorAction SilentlyContinue }

    $env:BIOCHEM_TRAIN_MODE = "new"
    $env:BIOCHEM_PRESET = ""
    $env:BIOCHEM_COMPLEXITY_STEP = "2"
    $env:BIOCHEM_NO_TEACHER_DEFAULTS = "1"
    $env:BIOCHEM_LOSS_DATA_ONLY = "1"
    $env:BIOCHEM_LOSS_ISOLATE = "MU_LOG"
    $env:BIOCHEM_STOP_AFTER_TEACHER = "1"
    $env:BIOCHEM_TEACHER_EPOCHS = "$Ep"
    $env:BIOCHEM_EPOCHS = "$Ep"
    $env:BIOCHEM_CLI_TEACHER_EPOCHS = "$Ep"
    $env:BIOCHEM_TEACHER_VAL_EVERY = "$ValEvery"
    $env:BIOCHEM_VAL_TIME_STRIDE = "10"
    $env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "1.0"
    $env:BIOCHEM_MU_LOG_WALL_WEIGHT = "2.0"
    $env:BIOCHEM_MU_LOG_HIGH_WEIGHT = "2.0"
    $env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "0.0"
    $env:BIOCHEM_USE_MU_PATH_GROUP = "1"
    $env:BIOCHEM_TRAIN_MU_ENCODER = "1"
    $env:BIOCHEM_USE_DELTA_MU_HEAD = "1"
    $env:BIOCHEM_USE_SPLIT_MU_HEAD = "1"
    $env:BIOCHEM_USE_WALL_DELTA_HEAD = "1"
    $env:BIOCHEM_USE_BIO_GATE_SUPPRESSOR = "1"
    $env:BIOCHEM_BIO_SUPPRESS_WALL_ALPHA = "0.0"
    $env:BIOCHEM_DELTA_MU_LOG_CLIP_BULK = "1.5"
    $env:BIOCHEM_DELTA_MU_LOG_CLIP_WALL = "5.0"
    $env:BIOCHEM_WALL_HEAD_PHYS_MIX = "1.0"
    $env:BIOCHEM_WALL_SPATIAL_DECAY = "0"
    $env:BIOCHEM_WALL_HEAD_ISOLATE_GEOM = "1"
    $env:BIOCHEM_WALL_BIO_BLEND_WEIGHT = "0.15"
    $env:BIOCHEM_LORA_RANK = "0"
    $env:BIOCHEM_STOCK_DEFAULTS = "0"
    $env:BIOCHEM_TEACHER_PARETO_CHECKPOINT = "0"
    $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
    $env:BIOCHEM_DATALOADER_WORKERS = "0"
    $env:BIOCHEM_PIN_MEMORY = "0"
    $env:BIOCHEM_DETACH_MACRO_STATE = "1"
    $env:BIOCHEM_TBPTT_MAX_WINDOW = "5"
    $env:BIOCHEM_TBPTT_WINDOW_CURRICULUM = "0"
    $env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "8"
    $env:BIOCHEM_ODE_GRADIENT_CHECKPOINT = "1"
    $env:BIOCHEM_KIN_GRADIENT_CHECKPOINT = "1"
}

function Set-WarmStartEnv {
    param([bool]$UseWarm, [bool]$IsFirstLeg)

    if ($UseWarm -and (-not $ForcePretrain)) {
        $env:BIOCHEM_SKIP_PRETRAIN = "1"
        $env:BIOCHEM_REUSE_LAST_PRETRAIN = "1"
    } elseif ($IsFirstLeg -and (-not $ForcePretrain)) {
        $env:BIOCHEM_SKIP_PRETRAIN = "0"
        $env:BIOCHEM_REUSE_LAST_PRETRAIN = "0"
    } else {
        $env:BIOCHEM_SKIP_PRETRAIN = "1"
        $env:BIOCHEM_REUSE_LAST_PRETRAIN = "1"
    }
}

$LegDefsA = [ordered]@{}
foreach ($ep in $EpochLadder) {
    $LegDefsA["B0_ep$ep"] = @{
        Label = "baseline_ep${ep}"
        Epochs = $ep
        Preset = ""
    }
}
$LegDefsA["B0_ep20_pareto"] = @{
    Label = "baseline_ep20_pareto_ckpt"
    Epochs = 20
    Preset = ""
    ParetoCheckpoint = "1"
}
$LegDefsA["WS_ep18"] = @{
    Label = "sweep_wall_sentinel_ep18"
    Epochs = 18
    Preset = "sweep_wall_sentinel"
}

$LegDefsB = [ordered]@{}
foreach ($ep in $EpochLadder) {
    $LegDefsB["FWa_ep$ep"] = @{
        Label = "sweep_free_wall_a_ep${ep}"
        Epochs = $ep
        Preset = "sweep_free_wall_a"
    }
}
$LegDefsB["FWb_ep20"] = @{
    Label = "sweep_free_wall_b_ep20"
    Epochs = 20
    Preset = "sweep_free_wall_b"
}
$LegDefsB["BIO_ep18"] = @{
    Label = "sweep_bio_suppressor_ep18"
    Epochs = 18
    Preset = "sweep_bio_suppressor"
}

$LegDefs = if ($Arm -eq "A") { $LegDefsA } else { $LegDefsB }
$UseWarmStart = (-not $ForcePretrain) -and (Test-Path $WarmStart)
$legKeys = @($LegDefs.Keys)
$legTotal = $legKeys.Count

Write-Host "Wall-gate fair 3h sweep | Arm=$Arm | legs=$legTotal | host=$hostName | warm_start=$UseWarmStart" -ForegroundColor Cyan
Add-Content -Path $SummaryPath -Value "BATCH_START arm=$Arm host=$hostName ts=$batchIso legs=$legTotal warm_start=$UseWarmStart"

$legIndex = 0
foreach ($key in $legKeys) {
    $legIndex++
    $def = $LegDefs[$key]
    $ep = [int]$def.Epochs
    $label = $def.Label
    $preset = [string]$def.Preset

    Write-Host ""
    Write-Host "========== [$Arm] $key : $label ($legIndex / $legTotal) ==========" -ForegroundColor Yellow

    Set-FairBase -Ep $ep
    Set-WarmStartEnv -UseWarm $UseWarmStart -IsFirstLeg ($legIndex -eq 1)

    if ($preset) {
        $env:BIOCHEM_PRESET = $preset
        $env:BIOCHEM_WALL_HEAD_ISOLATE_GEOM = "1"
        # Preset overrides wall/high weights; fair base still sets geom-isolate + MU_LOG.
    }

    if ($def.ParetoCheckpoint -eq "1") {
        $env:BIOCHEM_TEACHER_PARETO_CHECKPOINT = "1"
    }

    $runNote = "wall3h_${Arm}_${key}_${hostName}"
    $env:BIOCHEM_RUN_NOTE = $runNote

    $cmd = @(
        "-m", "src.training.train_biochem_corrector",
        "--new", "--run-name", $runNote,
        "--epochs", "$ep", "--save-best"
    )
    if ($ExtraArgs.Count -gt 0) { $cmd += $ExtraArgs }

    Add-Content -Path $SummaryPath -Value "START arm=$Arm leg=$key label=$label ep=$ep preset=$preset ts=$(Get-Date -Format o)"

    if ($DryRun) {
        Write-Host "DryRun: python $($cmd -join ' ')" -ForegroundColor DarkGray
        continue
    }

    $t0 = Get-Date
    python @cmd
    if ($LASTEXITCODE -ne 0) {
        $failLine = "FAIL arm=$Arm leg=$key label=$label exit=$LASTEXITCODE ts=$(Get-Date -Format o)"
        Add-Content -Path $SummaryPath -Value $failLine
        throw $failLine
    }

    $mins = [int]((Get-Date) - $t0).TotalMinutes
    $okLine = "OK arm=$Arm leg=$key label=$label duration=${mins}m note=$runNote ts=$(Get-Date -Format o)"
    Add-Content -Path $SummaryPath -Value $okLine
    Write-Host $okLine -ForegroundColor Green

    python -c "import gc; gc.collect(); import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>$null
}

$batchMins = [int]((Get-Date) - $batchStart).TotalMinutes
Add-Content -Path $SummaryPath -Value "BATCH_OK arm=$Arm host=$hostName duration=${batchMins}m ts=$(Get-Date -Format o)"
Write-Host ""
Write-Host "Arm $Arm complete | batch duration=${batchMins}m" -ForegroundColor Green
Write-Host "Summary: $SummaryPath"
