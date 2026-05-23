# Gate-collapse fix A/B/C/D sweep (~18 ep per leg, fair MU_LOG teacher base).
# Run ONE arm per machine from repo root:
#
#   Laptop A (single-fix legs):
#     .\scripts\run_biochem_gate_fix_sweep.ps1 -Arm A
#
#   Laptop B (combos + sentinel reference):
#     .\scripts\run_biochem_gate_fix_sweep.ps1 -Arm B
#
# Dry run:
#     .\scripts\run_biochem_gate_fix_sweep.ps1 -Arm A -DryRun
#
# Summary:
#   outputs\reports\training\biochem\gate_fix_sweep_summary.txt
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
$SummaryPath = Join-Path $RepoRoot "outputs\reports\training\biochem\gate_fix_sweep_summary.txt"
$SummaryDir = Split-Path -Parent $SummaryPath
if (-not (Test-Path $SummaryDir)) { New-Item -ItemType Directory -Path $SummaryDir -Force | Out-Null }

$WarmStart = Join-Path $RepoRoot "outputs\biochem\biochem_post_pretrain.pth"
$TeacherEp = 18
$batchStart = Get-Date
$batchIso = $batchStart.ToString("o")

function Set-FairBase {
    param([int]$Ep = 18, [int]$ValEvery = 2)

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
    # Clear fix knobs (legs set explicitly).
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
            $env:BIOCHEM_MU_WALL_BYPASS_WEIGHT = "1.5"
        }
        "fix_c" {
            $env:BIOCHEM_MU_WALL_GATE_POS_INIT = "3.0"
        }
        "fix_d_relu" {
            $env:BIOCHEM_MU_WALL_MIX_MODE = "relu_add"
            $env:BIOCHEM_MU_WALL_HEAD_ACTIVATION = "relu"
        }
        "fix_d_siren" {
            $env:BIOCHEM_MU_WALL_MIX_MODE = "relu_add"
            $env:BIOCHEM_MU_WALL_HEAD_ACTIVATION = "siren"
        }
        "fix_ab" {
            $env:BIOCHEM_WALL_GATE_CURRICULUM_EPOCHS = "12"
            $env:BIOCHEM_MU_WALL_BYPASS_WEIGHT = "1.5"
        }
        "fix_ac" {
            $env:BIOCHEM_WALL_GATE_CURRICULUM_EPOCHS = "12"
            $env:BIOCHEM_MU_WALL_GATE_POS_INIT = "3.0"
        }
        "fix_abc" {
            $env:BIOCHEM_WALL_GATE_CURRICULUM_EPOCHS = "12"
            $env:BIOCHEM_MU_WALL_BYPASS_WEIGHT = "1.5"
            $env:BIOCHEM_MU_WALL_GATE_POS_INIT = "3.0"
        }
        "sentinel" {
            $env:BIOCHEM_PRESET = "sweep_wall_sentinel"
        }
        default { throw "Unknown fix key: $FixKey" }
    }
}

$LegDefsA = [ordered]@{
    "A_baseline" = @{ Fix = "baseline"; Label = "gate_fix_baseline_ep18" }
    "A_fix_a"    = @{ Fix = "fix_a"; Label = "gate_fix_a_curriculum_ep18" }
    "A_fix_b"    = @{ Fix = "fix_b"; Label = "gate_fix_b_bypass_ep18" }
    "A_fix_c"    = @{ Fix = "fix_c"; Label = "gate_fix_c_pos_gate_init_ep18" }
    "A_fix_d_relu"  = @{ Fix = "fix_d_relu"; Label = "gate_fix_d_relu_add_ep18" }
    "A_fix_d_siren" = @{ Fix = "fix_d_siren"; Label = "gate_fix_d_siren_add_ep18" }
}

$LegDefsB = [ordered]@{
    "B_sentinel" = @{ Fix = "sentinel"; Label = "gate_fix_sentinel_ref_ep18" }
    "B_fix_ab"   = @{ Fix = "fix_ab"; Label = "gate_fix_ab_curriculum_bypass_ep18" }
    "B_fix_ac"   = @{ Fix = "fix_ac"; Label = "gate_fix_ac_curriculum_posinit_ep18" }
    "B_fix_abc"  = @{ Fix = "fix_abc"; Label = "gate_fix_abc_combo_ep18" }
}

$LegDefs = if ($Arm -eq "A") { $LegDefsA } else { $LegDefsB }
$UseWarmStart = (-not $ForcePretrain) -and (Test-Path $WarmStart)
$legKeys = @($LegDefs.Keys)
$legTotal = $legKeys.Count

Write-Host "Gate-fix sweep | Arm=$Arm | legs=$legTotal | ep=$TeacherEp | host=$hostName | warm=$UseWarmStart" -ForegroundColor Cyan
Add-Content -Path $SummaryPath -Value "BATCH_START arm=$Arm host=$hostName ts=$batchIso legs=$legTotal ep=$TeacherEp warm=$UseWarmStart"

$legIndex = 0
foreach ($key in $legKeys) {
    $legIndex++
    $def = $LegDefs[$key]
    $label = $def.Label
    $fix = $def.Fix

    Write-Host ""
    Write-Host "========== [$Arm] $key : $label ($legIndex / $legTotal) fix=$fix ==========" -ForegroundColor Yellow

    Set-FairBase -Ep $TeacherEp
    Apply-FixLeg -FixKey $fix
    Set-WarmStartEnv -UseWarm $UseWarmStart -IsFirstLeg ($legIndex -eq 1)

    $runNote = "gatefix_${Arm}_${key}_${hostName}"
    $env:BIOCHEM_RUN_NOTE = $runNote

    $cmd = @(
        "-m", "src.training.train_biochem_corrector",
        "--new", "--run-name", $runNote,
        "--epochs", "$TeacherEp", "--save-best"
    )
    if ($ExtraArgs.Count -gt 0) { $cmd += $ExtraArgs }

    Add-Content -Path $SummaryPath -Value "START arm=$Arm leg=$key fix=$fix label=$label ts=$(Get-Date -Format o)"

    if ($DryRun) {
        Write-Host "DryRun: python $($cmd -join ' ')" -ForegroundColor DarkGray
        Write-Host "  WALL_CURR=$env:BIOCHEM_WALL_GATE_CURRICULUM_EPOCHS BYPASS=$env:BIOCHEM_MU_WALL_BYPASS_WEIGHT POS_INIT=$env:BIOCHEM_MU_WALL_GATE_POS_INIT MIX=$env:BIOCHEM_MU_WALL_MIX_MODE ACT=$env:BIOCHEM_MU_WALL_HEAD_ACTIVATION PRESET=$env:BIOCHEM_PRESET"
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
