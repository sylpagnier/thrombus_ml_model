# Reproduce best teacher recipe (~20 min on P2200/RTX500 class GPU with warm-start).
# Writes biochem_teacher_last.pth + biochem_teacher_best_high_mu.pth (global high-μ best).
#
# From repo root (venv active):
#   .\scripts\run_biochem_sentinel_teacher_20min.ps1
#
# Then visualize (default = global best high-μ teacher, else latest teacher):
#   python src/evaluation/visualize_pipeline.py --teacher-only
#
param(
    [int] $Epochs = 34,
    [int] $ValEvery = 2,
    [switch] $ForcePretrain,
    [string[]] $ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$hostName = $env:COMPUTERNAME
$WarmStart = Join-Path $RepoRoot "outputs\biochem\biochem_post_pretrain.pth"

Get-ChildItem Env:BIOCHEM_* | ForEach-Object { Remove-Item "Env:$($_.Name)" -ErrorAction SilentlyContinue }

$env:BIOCHEM_TRAIN_MODE = "new"
$env:BIOCHEM_PRESET = "sweep_wall_sentinel"
$env:BIOCHEM_WALL_HEAD_ISOLATE_GEOM = "1"
$env:BIOCHEM_STOP_AFTER_TEACHER = "1"
$env:BIOCHEM_TEACHER_EPOCHS = "$Epochs"
$env:BIOCHEM_EPOCHS = "$Epochs"
$env:BIOCHEM_CLI_TEACHER_EPOCHS = "$Epochs"
$env:BIOCHEM_TEACHER_VAL_EVERY = "$ValEvery"
$env:BIOCHEM_VAL_TIME_STRIDE = "10"
$env:BIOCHEM_DETACH_MACRO_STATE = "1"
$env:BIOCHEM_TEACHER_KEEP_GLOBAL_BEST = "1"
$env:BIOCHEM_DEBUG = "0"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:BIOCHEM_DATALOADER_WORKERS = "0"
$env:BIOCHEM_PIN_MEMORY = "0"

if ($ForcePretrain) {
    $env:BIOCHEM_SKIP_PRETRAIN = "0"
    $env:BIOCHEM_REUSE_LAST_PRETRAIN = "0"
} elseif (Test-Path $WarmStart) {
    $env:BIOCHEM_SKIP_PRETRAIN = "1"
    $env:BIOCHEM_REUSE_LAST_PRETRAIN = "1"
    Write-Host "Warm-start: $WarmStart" -ForegroundColor Cyan
} else {
    $env:BIOCHEM_SKIP_PRETRAIN = "0"
    $env:BIOCHEM_REUSE_LAST_PRETRAIN = "0"
    Write-Host "No post-pretrain found; running AE+ODE pretrain first." -ForegroundColor Yellow
}

$runNote = "WS_sentinel_ep${Epochs}_${hostName}"
$env:BIOCHEM_RUN_NOTE = $runNote

Write-Host "Sentinel teacher ~20min | ep=$Epochs val_every=$ValEvery | run_note=$runNote" -ForegroundColor Cyan

$cmd = @(
    "-m", "src.training.train_biochem_corrector",
    "--new",
    "--run-name", $runNote,
    "--epochs", "$Epochs",
    "--save-best"
)
if ($ExtraArgs.Count -gt 0) { $cmd += $ExtraArgs }

python @cmd
exit $LASTEXITCODE
