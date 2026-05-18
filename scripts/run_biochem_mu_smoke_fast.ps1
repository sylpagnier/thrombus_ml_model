# Fast μ smoke test (minutes) for train_biochem_corrector.
# Purpose: verify we can move μ-directed training signal quickly.
# Not a generalization benchmark.
#
# Usage (repo root):
#   .\scripts\run_biochem_mu_smoke_fast.ps1
#   .\scripts\run_biochem_mu_smoke_fast.ps1 -LossIsolate MU_SI -TeacherEpochs 4
#   .\scripts\run_biochem_mu_smoke_fast.ps1 -UseDeltaMuHead

param(
    [ValidateSet("MU_LOG", "MU_SI")]
    [string] $LossIsolate = "MU_LOG",
    [int] $TeacherEpochs = 3,
    [switch] $UseDeltaMuHead,
    [string[]] $ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "μ smoke fast: signal sanity only (not generalization)." -ForegroundColor Yellow
Write-Host "Target: finite run + directional movement in L_Back / μ isolate loss in 2-4 epochs." -ForegroundColor Yellow

Get-ChildItem Env:BIOCHEM_* -ErrorAction SilentlyContinue | ForEach-Object {
    Remove-Item "Env:$($_.Name)" -ErrorAction SilentlyContinue
}

$warmStart = Join-Path $RepoRoot "outputs\biochem\biochem_post_pretrain.pth"
$useWarm = Test-Path $warmStart

$env:BIOCHEM_RUN_NOTE = "mu_smoke_fast_$LossIsolate"
$env:BIOCHEM_STOCK_DEFAULTS = "1"
$env:BIOCHEM_PRESET = ""
$env:BIOCHEM_COMPLEXITY_STEP = "2"
$env:BIOCHEM_LOSS_DATA_ONLY = "1"
$env:BIOCHEM_LOSS_ISOLATE = $LossIsolate
$env:BIOCHEM_STOP_AFTER_TEACHER = "1"
$env:BIOCHEM_NO_TEACHER_DEFAULTS = "1"
$env:BIOCHEM_TEACHER_EPOCHS = "$TeacherEpochs"
$env:BIOCHEM_TEACHER_SKIP_VAL = "1"
$env:BIOCHEM_TEACHER_VAL_EVERY = "99"
$env:BIOCHEM_VAL_TIME_STRIDE = "10"
$env:BIOCHEM_MAX_LOAD_VESSELS = "1"
$env:BIOCHEM_MAX_LOAD_SHUFFLE = "0"
$env:BIOCHEM_LOW_ANCHOR_MODE = "1"
$env:BIOCHEM_DATALOADER_WORKERS = "0"
$env:BIOCHEM_TBPTT_MAX_WINDOW = "4"
$env:BIOCHEM_TBPTT_WINDOW_CURRICULUM = "0"
$env:BIOCHEM_TBPTT_ANCHOR_END_BIAS = "0"
$env:BIOCHEM_TBPTT_ANCHOR_RANDOM_START = "0"
$env:BIOCHEM_TEACHER_TBPTT_RANDOM_ANCHOR = "0"
$env:BIOCHEM_DETACH_MACRO_STATE = "1"
$env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "12"
$env:BIOCHEM_TEACHER_FORCE_MIN = "0.0"
$env:BIOCHEM_TEACHER_TF_WARMUP_EPOCHS = "2"
$env:BIOCHEM_TEACHER_MU_RATIO_MAX = "80.0"
$env:BIOCHEM_MU_SI_MULTI_STEP = "1"
$env:BIOCHEM_MU_SI_HUBER_DELTA = "0.25"
$env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = if ($LossIsolate -eq "MU_SI") { "8.0" } else { "0.0" }
$env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT = if ($LossIsolate -eq "MU_LOG") { "2.0" } else { "0.0" }
$env:BIOCHEM_TRAIN_MU_ENCODER = "1"
$env:BIOCHEM_USE_MU_PATH_GROUP = "1"
$env:BIOCHEM_MU_PATH_LR_MULT = "1.0"
$env:BIOCHEM_DEBUG = "0"

if ($UseDeltaMuHead) {
    $env:BIOCHEM_USE_DELTA_MU_HEAD = "1"
    $env:BIOCHEM_DELTA_MU_LOG_CLIP = "1.5"
} else {
    $env:BIOCHEM_USE_DELTA_MU_HEAD = "0"
}

if ($useWarm) {
    $env:BIOCHEM_SKIP_PRETRAIN = "1"
    $env:BIOCHEM_REUSE_LAST_PRETRAIN = "1"
    Write-Host "Warm-start enabled: $warmStart" -ForegroundColor Cyan
} else {
    $env:BIOCHEM_SKIP_PRETRAIN = "0"
    $env:BIOCHEM_REUSE_LAST_PRETRAIN = "0"
    Write-Host "Warm-start not found; pretrain will run (slower)." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Running: python -m src.training.train_biochem_corrector --new" -ForegroundColor Cyan
Write-Host "Loss isolate=$LossIsolate | epochs=$TeacherEpochs | delta_mu_head=$($env:BIOCHEM_USE_DELTA_MU_HEAD)"
Write-Host ""

python -m src.training.train_biochem_corrector --new @ExtraArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Done. Inspect newest outputs/reports/training/biochem/<timestamp>/metrics.jsonl" -ForegroundColor Green
Write-Host "Smoke pass: stable finite run + directional L_Back movement under μ isolate."
