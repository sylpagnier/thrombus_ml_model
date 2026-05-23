# Gate-fix deep exploration (~3.5–4.5 h per laptop, fair MU_LOG teacher base).
# Extends the 18ep gate-fix sweep with epoch ladders on the best legs + Pareto / hybrids.
#
# Run ONE arm per machine from repo root (venv active):
#
#   Laptop A (SPAGNIER — Fix D relu + Fix A curriculum):
#     .\scripts\run_biochem_gate_fix_deep_4h.ps1 -Arm A
#
#   Laptop B (SILKSPECTRE — fix_ac + fix_ab + sentinel wall weights):
#     .\scripts\run_biochem_gate_fix_deep_4h.ps1 -Arm B
#
# Dry run:
#     .\scripts\run_biochem_gate_fix_deep_4h.ps1 -Arm A -DryRun
#
# Summary:
#   outputs\reports\training\biochem\gate_fix_deep_4h_summary.txt
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
$SummaryPath = Join-Path $RepoRoot "outputs\reports\training\biochem\gate_fix_deep_4h_summary.txt"
$SummaryDir = Split-Path -Parent $SummaryPath
if (-not (Test-Path $SummaryDir)) { New-Item -ItemType Directory -Path $SummaryDir -Force | Out-Null }

$WarmStart = Join-Path $RepoRoot "outputs\biochem\biochem_post_pretrain.pth"
$batchStart = Get-Date
$batchIso = $batchStart.ToString("o")

function Set-FairBase {
    param([int]$Ep = 34, [int]$ValEvery = 2, [int]$Tbptt = 5)

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
    $env:BIOCHEM_DEBUG = "0"
    $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
    $env:BIOCHEM_DATALOADER_WORKERS = "0"
    $env:BIOCHEM_PIN_MEMORY = "0"
    $env:BIOCHEM_DETACH_MACRO_STATE = "1"
    $env:BIOCHEM_TBPTT_MAX_WINDOW = "$Tbptt"
    $env:BIOCHEM_TBPTT_WINDOW_CURRICULUM = "0"
    $env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "8"
    $env:BIOCHEM_ODE_GRADIENT_CHECKPOINT = "1"
    $env:BIOCHEM_KIN_GRADIENT_CHECKPOINT = "1"
    $env:BIOCHEM_WALL_GATE_CURRICULUM_EPOCHS = "0"
    $env:BIOCHEM_MU_WALL_BYPASS_WEIGHT = "0"
    $env:BIOCHEM_WALL_GATE_BIAS = "0"
    $env:BIOCHEM_MU_WALL_GATE_POS_INIT = ""
    $env:BIOCHEM_MU_WALL_MIX_MODE = "gate"
    $env:BIOCHEM_MU_WALL_HEAD_ACTIVATION = "silu"
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

function Apply-FixLeg {
    param([string]$FixKey)

    switch ($FixKey) {
        "baseline" { }
        "fix_a" {
            $env:BIOCHEM_WALL_GATE_CURRICULUM_EPOCHS = "12"
        }
        "fix_b" {
            $env:BIOCHEM_MU_WALL_BYPASS_WEIGHT = "1.0"
        }
        "fix_b_strong" {
            $env:BIOCHEM_MU_WALL_BYPASS_WEIGHT = "1.5"
        }
        "fix_c" {
            $env:BIOCHEM_MU_WALL_GATE_POS_INIT = "3.0"
        }
        "fix_d_relu" {
            $env:BIOCHEM_MU_WALL_MIX_MODE = "relu_add"
            $env:BIOCHEM_MU_WALL_HEAD_ACTIVATION = "relu"
        }
        "fix_d_a" {
            $env:BIOCHEM_MU_WALL_MIX_MODE = "relu_add"
            $env:BIOCHEM_MU_WALL_HEAD_ACTIVATION = "relu"
            $env:BIOCHEM_WALL_GATE_CURRICULUM_EPOCHS = "12"
        }
        "fix_ab" {
            $env:BIOCHEM_WALL_GATE_CURRICULUM_EPOCHS = "12"
            $env:BIOCHEM_MU_WALL_BYPASS_WEIGHT = "1.0"
        }
        "fix_ab_strong" {
            $env:BIOCHEM_WALL_GATE_CURRICULUM_EPOCHS = "12"
            $env:BIOCHEM_MU_WALL_BYPASS_WEIGHT = "1.5"
        }
        "fix_ab_sentinel" {
            $env:BIOCHEM_WALL_GATE_CURRICULUM_EPOCHS = "12"
            $env:BIOCHEM_MU_WALL_BYPASS_WEIGHT = "1.0"
            $env:BIOCHEM_PRESET = "sweep_wall_sentinel"
        }
        "fix_ac" {
            $env:BIOCHEM_WALL_GATE_CURRICULUM_EPOCHS = "12"
            $env:BIOCHEM_MU_WALL_GATE_POS_INIT = "3.0"
        }
        "sentinel" {
            $env:BIOCHEM_PRESET = "sweep_wall_sentinel"
        }
        default { throw "Unknown fix key: $FixKey" }
    }
}

# ~4 h budget: epoch ladders (6–7 points) + 4–5 anchor legs per arm @ ~8–14 min/leg (fair MU_LOG).
$LadderA = @(8, 14, 20, 26, 30, 34, 40)
$LadderB = @(8, 14, 20, 26, 30, 34, 40)

$LegDefsA = [ordered]@{}
foreach ($ep in $LadderA) {
    $LegDefsA["D_relu_ep$ep"] = @{
        Fix = "fix_d_relu"
        Label = "deep_D_relu_ep${ep}"
        Epochs = $ep
    }
}
foreach ($ep in @(14, 20, 26, 30, 34, 40)) {
    $LegDefsA["A_cur_ep$ep"] = @{
        Fix = "fix_a"
        Label = "deep_A_curriculum_ep${ep}"
        Epochs = $ep
    }
}
$LegDefsA["D_relu_ep34_pareto"] = @{
    Fix = "fix_d_relu"
    Label = "deep_D_relu_ep34_pareto"
    Epochs = 34
    ParetoCheckpoint = "1"
}
$LegDefsA["D_relu_A_ep34"] = @{
    Fix = "fix_d_a"
    Label = "deep_D_relu_plus_A_ep34"
    Epochs = 34
}
$LegDefsA["WS_ep34"] = @{
    Fix = "sentinel"
    Label = "deep_sentinel_preset_ep34"
    Epochs = 34
}
$LegDefsA["D_relu_ep34_tbptt6"] = @{
    Fix = "fix_d_relu"
    Label = "deep_D_relu_ep34_tbptt6"
    Epochs = 34
    Tbptt = 6
}

$LegDefsB = [ordered]@{}
foreach ($ep in $LadderB) {
    $LegDefsB["AC_ep$ep"] = @{
        Fix = "fix_ac"
        Label = "deep_AC_ep${ep}"
        Epochs = $ep
    }
}
foreach ($ep in @(20, 26, 30, 34, 40)) {
    $LegDefsB["AB_ep$ep"] = @{
        Fix = "fix_ab"
        Label = "deep_AB_ep${ep}"
        Epochs = $ep
    }
}
$LegDefsB["AC_ep34_pareto"] = @{
    Fix = "fix_ac"
    Label = "deep_AC_ep34_pareto"
    Epochs = 34
    ParetoCheckpoint = "1"
}
$LegDefsB["AB_sentinel_ep34"] = @{
    Fix = "fix_ab_sentinel"
    Label = "deep_AB_sentinel_wall_weights_ep34"
    Epochs = 34
}
$LegDefsB["WS_ep34"] = @{
    Fix = "sentinel"
    Label = "deep_sentinel_preset_ep34"
    Epochs = 34
}
$LegDefsB["AB_ep34_bypass15"] = @{
    Fix = "fix_ab_strong"
    Label = "deep_AB_bypass15_curriculum_ep34"
    Epochs = 34
}

$LegDefs = if ($Arm -eq "A") { $LegDefsA } else { $LegDefsB }
$UseWarmStart = (-not $ForcePretrain) -and (Test-Path $WarmStart)
$legKeys = @($LegDefs.Keys)
$legTotal = $legKeys.Count

Write-Host "Gate-fix deep 4h | Arm=$Arm | legs=$legTotal | host=$hostName | warm=$UseWarmStart" -ForegroundColor Cyan
Add-Content -Path $SummaryPath -Value "BATCH_START arm=$Arm host=$hostName ts=$batchIso legs=$legTotal warm=$UseWarmStart"

$legIndex = 0
foreach ($key in $legKeys) {
    $legIndex++
    $def = $LegDefs[$key]
    $ep = [int]$def.Epochs
    $label = $def.Label
    $fix = $def.Fix
    $tbptt = if ($def.Tbptt) { [int]$def.Tbptt } else { 5 }

    Write-Host ""
    Write-Host "========== [$Arm] $key : $label ($legIndex / $legTotal) fix=$fix ep=$ep ==========" -ForegroundColor Yellow

    Set-FairBase -Ep $ep -Tbptt $tbptt
    Apply-FixLeg -FixKey $fix
    if ($def.ParetoCheckpoint -eq "1") {
        $env:BIOCHEM_TEACHER_PARETO_CHECKPOINT = "1"
    }
    Set-WarmStartEnv -UseWarm $UseWarmStart -IsFirstLeg ($legIndex -eq 1)

    $runNote = "gate4h_${Arm}_${key}_${hostName}"
    $env:BIOCHEM_RUN_NOTE = $runNote

    $cmd = @(
        "-m", "src.training.train_biochem_corrector",
        "--new", "--run-name", $runNote,
        "--epochs", "$ep", "--save-best"
    )
    if ($ExtraArgs.Count -gt 0) { $cmd += $ExtraArgs }

    Add-Content -Path $SummaryPath -Value "START arm=$Arm leg=$key fix=$fix label=$label ep=$ep ts=$(Get-Date -Format o)"

    if ($DryRun) {
        Write-Host "DryRun: python $($cmd -join ' ')" -ForegroundColor DarkGray
        Write-Host "  TBPTT=$tbptt PARETO=$($env:BIOCHEM_TEACHER_PARETO_CHECKPOINT) PRESET=$($env:BIOCHEM_PRESET) MIX=$($env:BIOCHEM_MU_WALL_MIX_MODE)"
        continue
    }

    $t0 = Get-Date
    python @cmd
    if ($LASTEXITCODE -ne 0) {
        $failLine = "FAIL arm=$Arm leg=$key fix=$fix exit=$LASTEXITCODE ts=$(Get-Date -Format o)"
        Add-Content -Path $SummaryPath -Value $failLine
        throw $failLine
    }

    $mins = [int]((Get-Date) - $t0).TotalMinutes
    $okLine = "OK arm=$Arm leg=$key fix=$fix duration=${mins}m note=$runNote ts=$(Get-Date -Format o)"
    Add-Content -Path $SummaryPath -Value $okLine
    Write-Host $okLine -ForegroundColor Green

    python -c "import gc; gc.collect(); import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>$null
}

$batchMins = [int]((Get-Date) - $batchStart).TotalMinutes
Add-Content -Path $SummaryPath -Value "BATCH_OK arm=$Arm host=$hostName duration=${batchMins}m ts=$(Get-Date -Format o)"
Write-Host ""
Write-Host "Arm $Arm complete | batch duration=${batchMins}m" -ForegroundColor Green
Write-Host "Summary: $SummaryPath"
