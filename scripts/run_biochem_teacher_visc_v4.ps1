# Teacher-only viscosity V4 runner with two targeted profiles.
#
# Profiles:
#   global_plus  - improve overall/wall/high balance with stronger latent capacity.
#   high_mu_only - isolate high-mu tail objective to stress clot-region fidelity.
#
# Usage:
#   .\scripts\run_biochem_teacher_visc_v4.ps1 -Profile global_plus
#   .\scripts\run_biochem_teacher_visc_v4.ps1 -Profile high_mu_only
#   .\scripts\run_biochem_teacher_visc_v4.ps1 -ListProfiles
#
param(
    [ValidateSet("global_plus", "high_mu_only")]
    [string] $Profile = "global_plus",
    [switch] $ListProfiles,
    [switch] $ForcePretrain,
    [string[]] $ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if ($ListProfiles) {
    Write-Host "V4 teacher viscosity profiles:" -ForegroundColor Cyan
    Write-Host "  global_plus  - joint step-2 objective, latent=320 prior=2 delta-head=on"
    Write-Host "  high_mu_only - isolate MU_LOG_HIGH, latent=320 prior=4 delta-head=on"
    exit 0
}

# Clear stale env from prior experiments.
Get-ChildItem Env:BIOCHEM_* -ErrorAction SilentlyContinue | ForEach-Object {
    Remove-Item "Env:$($_.Name)" -ErrorAction SilentlyContinue
}

$WarmStart = Join-Path $RepoRoot "outputs\biochem\biochem_post_pretrain.pth"
$UseWarmStart = (-not $ForcePretrain) -and (Test-Path $WarmStart)

# Shared stable base from recent SAFEVAL/V3 runs.
$env:BIOCHEM_STOCK_DEFAULTS = "1"
$env:BIOCHEM_SKIP_PRETRAIN = if ($UseWarmStart) { "1" } else { "0" }
$env:BIOCHEM_REUSE_LAST_PRETRAIN = if ($UseWarmStart) { "1" } else { "0" }
$env:BIOCHEM_STOP_AFTER_TEACHER = "1"
$env:BIOCHEM_NO_TEACHER_DEFAULTS = "1"
$env:BIOCHEM_LOSS_DATA_ONLY = "1"
$env:BIOCHEM_TEACHER_EPOCHS = "24"
$env:BIOCHEM_TEACHER_VAL_EVERY = "4"
$env:BIOCHEM_VAL_TIME_STRIDE = "20"
$env:BIOCHEM_TBPTT_MAX_WINDOW = "6"
$env:BIOCHEM_TBPTT_WINDOW_CURRICULUM = "0"
$env:BIOCHEM_DETACH_MACRO_STATE = "1"
$env:BIOCHEM_TEACHER_FORCE_MIN = "0.0"
$env:BIOCHEM_TEACHER_TF_WARMUP_EPOCHS = "4"
$env:BIOCHEM_TEACHER_MU_RATIO_MAX = "80.0"
$env:BIOCHEM_MU_SI_MULTI_STEP = "1"
$env:BIOCHEM_USE_MU_PATH_GROUP = "1"
$env:BIOCHEM_TRAIN_MU_ENCODER = "1"
$env:BIOCHEM_USE_DELTA_MU_HEAD = "1"
$env:BIOCHEM_DATALOADER_WORKERS = "0"
$env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "8"

switch ($Profile) {
    "global_plus" {
        # Architecture change guided by recent sweeps: wider latent + delta head on.
        $env:BIOCHEM_LATENT_DIM = "320"
        $env:BIOCHEM_BIO_ENCODER_PRIOR_DIM = "2"

        # Balanced weights with slight tail emphasis (without wall over-push).
        $env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "2.0"
        $env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "2.0"
        $env:BIOCHEM_MU_LOG_WALL_WEIGHT = "1.6"
        $env:BIOCHEM_MU_LOG_HIGH_WEIGHT = "1.4"
        $env:BIOCHEM_MU_LOG_WALL_RAMP_EPOCHS = "10"
        $env:BIOCHEM_MU_LOG_HIGH_RAMP_EPOCHS = "6"
        $env:BIOCHEM_TEACHER_TARGET_MU_LOG_MAE = "0.50"
        $env:BIOCHEM_RUN_NOTE = "VISC_V4_GLOBAL_PLUS"
    }
    "high_mu_only" {
        # High-mu-focused architecture: wider latent + richer prior channels.
        $env:BIOCHEM_LATENT_DIM = "320"
        $env:BIOCHEM_BIO_ENCODER_PRIOR_DIM = "4"

        # Isolate the high-mu subset objective to learn clot-tail behavior.
        $env:BIOCHEM_LOSS_ISOLATE = "MU_LOG_HIGH"
        $env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "0.0"
        $env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "0.0"
        $env:BIOCHEM_MU_LOG_WALL_WEIGHT = "0.0"
        $env:BIOCHEM_MU_LOG_HIGH_WEIGHT = "3.0"
        $env:BIOCHEM_MU_LOG_WALL_RAMP_EPOCHS = "0"
        $env:BIOCHEM_MU_LOG_HIGH_RAMP_EPOCHS = "0"
        $env:BIOCHEM_TEACHER_FORCE_MIN = "0.1"
        $env:BIOCHEM_RUN_NOTE = "VISC_V4_HIGH_MU_ONLY"
    }
}

Write-Host "Teacher viscosity V4 run | profile=$Profile" -ForegroundColor Cyan
Write-Host "Warm-start: $UseWarmStart | Corrector: OFF | epochs=$env:BIOCHEM_TEACHER_EPOCHS" -ForegroundColor Cyan
Write-Host "Arch: latent=$env:BIOCHEM_LATENT_DIM prior_dim=$env:BIOCHEM_BIO_ENCODER_PRIOR_DIM delta_head=$env:BIOCHEM_USE_DELTA_MU_HEAD" -ForegroundColor Cyan
Write-Host "Weights: MuLog=$env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT MuSI=$env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT Wall=$env:BIOCHEM_MU_LOG_WALL_WEIGHT High=$env:BIOCHEM_MU_LOG_HIGH_WEIGHT" -ForegroundColor Cyan
if ($env:BIOCHEM_LOSS_ISOLATE) {
    Write-Host "Isolate objective: $env:BIOCHEM_LOSS_ISOLATE" -ForegroundColor Yellow
}
if ($UseWarmStart) {
    Write-Host "Warm-start checkpoint: $WarmStart" -ForegroundColor Yellow
}

python -m src.training.train_biochem_corrector --new @ExtraArgs
if ($LASTEXITCODE -ne 0) {
    throw "train_biochem_corrector failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Done. Check outputs/reports/training/biochem/<timestamp>/metrics.jsonl" -ForegroundColor Green
Write-Host "Track val subsets: mu_log_mae(all/wall/high), mu_pearson, and train L_Back."
