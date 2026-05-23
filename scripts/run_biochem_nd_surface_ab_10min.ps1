# Short teacher probe with default COMSOL-aligned ND surface physics (geom-isolate best arch).
# Historical A/B (2026-05-23) vs legacy surface ODEs is obsolete — ND path is now always on.
#
#   .\scripts\run_biochem_nd_surface_ab_10min.ps1
#
param(
    [ValidateSet("NdSurface")]
    [string] $Variant = "NdSurface",
    [int] $TeacherEpochs = 8,
    [int] $ValEvery = 2,
    [switch] $ForcePretrain,
    [switch] $DryRun,
    [string[]] $ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$hostName = $env:COMPUTERNAME
$SummaryPath = Join-Path $RepoRoot "outputs\reports\training\biochem\nd_surface_ab_10min_summary.txt"
$SummaryDir = Split-Path -Parent $SummaryPath
if (-not (Test-Path $SummaryDir)) { New-Item -ItemType Directory -Path $SummaryDir -Force | Out-Null }

$WarmStart = Join-Path $RepoRoot "outputs\biochem\biochem_post_pretrain.pth"
$UseWarmStart = (-not $ForcePretrain) -and (Test-Path $WarmStart)

$legs = @($Variant)
$batchStart = Get-Date
$batchStartIso = $batchStart.ToString("o")

Write-Host "ND surface physics A/B | legs=$($legs -join ',') | epochs=$TeacherEpochs | warm_start=$UseWarmStart" -ForegroundColor Cyan
Add-Content -Path $SummaryPath -Value "BATCH_START host=$hostName ts=$batchStartIso legs=$($legs -join ',') epochs=$TeacherEpochs warm_start=$UseWarmStart"

function Set-CommonFairEnv {
    param([int] $Epochs, [int] $ValEveryEp)

    $env:BIOCHEM_TRAIN_MODE = "new"
    $env:BIOCHEM_PRESET = ""
    $env:BIOCHEM_COMPLEXITY_STEP = "2"
    $env:BIOCHEM_NO_TEACHER_DEFAULTS = "1"
    $env:BIOCHEM_LOSS_DATA_ONLY = "1"
    $env:BIOCHEM_LOSS_ISOLATE = "MU_LOG"
    $env:BIOCHEM_STOP_AFTER_TEACHER = "1"
    $env:BIOCHEM_TEACHER_EPOCHS = "$Epochs"
    $env:BIOCHEM_EPOCHS = "$Epochs"
    $env:BIOCHEM_TEACHER_VAL_EVERY = "$ValEveryEp"
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
    # Couple surface ODE physics into the same MU_LOG teacher objective (both legs).
    $env:BIOCHEM_WALL_BIO_BLEND_WEIGHT = "0.15"
    $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
    $env:BIOCHEM_DATALOADER_WORKERS = "0"
    $env:BIOCHEM_PIN_MEMORY = "0"
    $env:BIOCHEM_DETACH_MACRO_STATE = "1"
    $env:BIOCHEM_TBPTT_MAX_WINDOW = "5"
    $env:BIOCHEM_TBPTT_WINDOW_CURRICULUM = "0"
    $env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "8"
    $env:BIOCHEM_ODE_GRADIENT_CHECKPOINT = "1"
    $env:BIOCHEM_KIN_GRADIENT_CHECKPOINT = "1"
    $env:BIOCHEM_LORA_RANK = "0"
    $env:BIOCHEM_STOCK_DEFAULTS = "0"

    if ($UseWarmStart) {
        $env:BIOCHEM_SKIP_PRETRAIN = "1"
        $env:BIOCHEM_REUSE_LAST_PRETRAIN = "1"
    } else {
        $env:BIOCHEM_SKIP_PRETRAIN = "0"
        $env:BIOCHEM_REUSE_LAST_PRETRAIN = "0"
    }
}

foreach ($leg in $legs) {
    Set-CommonFairEnv -Epochs $TeacherEpochs -ValEveryEp $ValEvery

    $runNote = "nd_surface_default_ep${TeacherEpochs}_${hostName}"
    $env:BIOCHEM_RUN_NOTE = $runNote
    $startTs = Get-Date
    $startIso = $startTs.ToString("o")

    Write-Host ""
    Write-Host "▶ $leg | start=$startIso" -ForegroundColor Yellow
    Add-Content -Path $SummaryPath -Value "START variant=$leg host=$hostName ts=$startIso epochs=$TeacherEpochs"

    $cmd = @("-m", "src.training.train_biochem_corrector", "--new", "--run-name", $runNote, "--epochs", "$TeacherEpochs", "--save-best")
    if ($ExtraArgs.Count -gt 0) { $cmd += $ExtraArgs }

    if ($DryRun) {
        Write-Host "DryRun: python $($cmd -join ' ')" -ForegroundColor DarkGray
        continue
    }

    python @cmd
    if ($LASTEXITCODE -ne 0) {
        $failIso = (Get-Date).ToString("o")
        Add-Content -Path $SummaryPath -Value "FAIL variant=$leg host=$hostName ts=$failIso exit=$LASTEXITCODE run_note=$runNote"
        throw "Run failed: $leg (exit=$LASTEXITCODE)"
    }

    $endTs = Get-Date
    $endIso = $endTs.ToString("o")
    $mins = [int](($endTs - $startTs).TotalMinutes)
    $secs = [int](($endTs - $startTs).TotalSeconds)
    Add-Content -Path $SummaryPath -Value "OK variant=$leg host=$hostName start=$startIso end=$endIso duration=${mins}m (${secs}s) run_note=$runNote"
    Write-Host "Completed $leg | duration=${mins}m (${secs}s)" -ForegroundColor Green
}

$batchEnd = Get-Date
$batchMins = [int](($batchEnd - $batchStart).TotalMinutes)
$batchSecs = [int](($batchEnd - $batchStart).TotalSeconds)
Add-Content -Path $SummaryPath -Value "BATCH_OK host=$hostName start=$batchStartIso end=$($batchEnd.ToString('o')) duration=${batchMins}m (${batchSecs}s) legs=$($legs -join ',')"

Write-Host ""
Write-Host "ND surface A/B complete | batch duration=${batchMins}m (${batchSecs}s)" -ForegroundColor Green
Write-Host "Summary log: $SummaryPath"
