# Fair head-to-head A/B test for "best" vs "second-best" architecture variants.
# Runs one variant per machine with identical training setup and memory guardrails.
#
# Suggested usage on two computers at the same time:
#   Computer 1: .\scripts\run_biochem_best_vs_second_30min.ps1 -Variant BestAllArch
#   Computer 2: .\scripts\run_biochem_best_vs_second_30min.ps1 -Variant BalancedArch
#
# Optional dry run:
#   .\scripts\run_biochem_best_vs_second_30min.ps1 -Variant BestAllArch -DryRun
#
param(
    [ValidateSet("BestAllArch", "BalancedArch")]
    [string] $Variant = "BestAllArch",
    [int] $TeacherEpochs = 6,
    [int] $ValEvery = 2,
    [switch] $ForcePretrain,
    [switch] $DryRun,
    [string[]] $ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$hostName = $env:COMPUTERNAME
$startTs = Get-Date
$startIso = $startTs.ToString("o")
$runTag = "h2h_${Variant}_ep${TeacherEpochs}_${hostName}"

$SummaryPath = Join-Path $RepoRoot "outputs\reports\training\biochem\best_vs_second_30min_summary.txt"
$SummaryDir = Split-Path -Parent $SummaryPath
if (-not (Test-Path $SummaryDir)) { New-Item -ItemType Directory -Path $SummaryDir -Force | Out-Null }

$WarmStart = Join-Path $RepoRoot "outputs\biochem\biochem_post_pretrain.pth"
$UseWarmStart = (-not $ForcePretrain) -and (Test-Path $WarmStart)

Write-Host "Best-vs-second architecture test started: $Variant" -ForegroundColor Cyan
Write-Host "Host=$hostName start=$startIso epochs=$TeacherEpochs val_every=$ValEvery warm_start=$UseWarmStart" -ForegroundColor DarkGray

$lineStart = "START variant=$Variant host=$hostName ts=$startIso epochs=$TeacherEpochs val_every=$ValEvery warm_start=$UseWarmStart"
Add-Content -Path $SummaryPath -Value $lineStart

# -------- Common, fair setup (identical across both variants) --------
$env:BIOCHEM_TRAIN_MODE = "new"
$env:BIOCHEM_PRESET = ""
$env:BIOCHEM_COMPLEXITY_STEP = "2"
$env:BIOCHEM_NO_TEACHER_DEFAULTS = "1"
$env:BIOCHEM_LOSS_DATA_ONLY = "1"
$env:BIOCHEM_LOSS_ISOLATE = "MU_LOG"
$env:BIOCHEM_STOP_AFTER_TEACHER = "1"
$env:BIOCHEM_TEACHER_EPOCHS = "$TeacherEpochs"
$env:BIOCHEM_EPOCHS = "$TeacherEpochs"
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

# Memory guardrails (kept equal so architecture is the only variable).
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

# -------- Variant-specific architecture toggle (single-axis A/B) --------
if ($Variant -eq "BestAllArch") {
    # Best-all family: wall spatial decay pathway.
    $env:BIOCHEM_WALL_SPATIAL_DECAY = "1"
    $env:BIOCHEM_WALL_HEAD_ISOLATE_GEOM = "0"
    $env:BIOCHEM_RUN_NOTE = "${runTag}_wall_spatial_decay"
} else {
    # Best-balanced family: geometry-isolated wall head.
    $env:BIOCHEM_WALL_SPATIAL_DECAY = "0"
    $env:BIOCHEM_WALL_HEAD_ISOLATE_GEOM = "1"
    $env:BIOCHEM_RUN_NOTE = "${runTag}_wall_geom_isolate"
}

$cmd = @("-m", "src.training.train_biochem_corrector", "--new", "--run-name", $env:BIOCHEM_RUN_NOTE, "--epochs", "$TeacherEpochs", "--save-best")
if ($ExtraArgs.Count -gt 0) { $cmd += $ExtraArgs }

if ($DryRun) {
    Write-Host "DryRun: python $($cmd -join ' ')" -ForegroundColor DarkGray
    return
}

python @cmd
if ($LASTEXITCODE -ne 0) {
    $failIso = (Get-Date).ToString("o")
    $lineFail = "FAIL variant=$Variant host=$hostName ts=$failIso exit=$LASTEXITCODE run_note=$($env:BIOCHEM_RUN_NOTE)"
    Add-Content -Path $SummaryPath -Value $lineFail
    throw "Run failed: $Variant (exit=$LASTEXITCODE)"
}

$endTs = Get-Date
$endIso = $endTs.ToString("o")
$mins = [int](($endTs - $startTs).TotalMinutes)
$lineOk = "OK variant=$Variant host=$hostName start=$startIso end=$endIso duration=${mins}m run_note=$($env:BIOCHEM_RUN_NOTE)"
Add-Content -Path $SummaryPath -Value $lineOk

Write-Host ""
Write-Host "Completed $Variant | start=$startIso end=$endIso duration=${mins}m" -ForegroundColor Green
Write-Host "Summary log: $SummaryPath"
