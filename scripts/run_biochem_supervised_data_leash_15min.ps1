# ~15 min probe: supervised data-only leash on top of best sentinel teacher weights.
# Tests whether L_Data_Kine + L_Data_Bio gradients (no MU_LOG isolate, no physics PDEs)
# improve mu / wall metrics vs the frozen MU_LOG-only recipe.
#
# Optional bulk-fluid surgical lock (prevents bulk head from absorbing global error):
#   BIOCHEM_DELTA_MU_LOG_CLIP_BULK=0.05, bio gate suppressor with floor 0.
# Re-applied after sweep_wall_sentinel preset (preset disables bio suppressor).
#
# From repo root (venv active):
#   .\scripts\run_biochem_supervised_data_leash_15min.ps1
#   .\scripts\run_biochem_supervised_data_leash_15min.ps1 -NoInitFromBest -ForcePretrain   # cold start (AE+ODE then teacher)
#   .\scripts\run_biochem_supervised_data_leash_15min.ps1 -NoInitFromBest -ForcePretrain -StrictMuFreeze  # μ-path only teacher
#   .\scripts\run_biochem_supervised_data_leash_15min.ps1 -NoInitFromBest -ForcePretrain -StrictMuFreeze -HardGateThreshold  # wall-only soft gate @ 0.15
#   .\scripts\run_biochem_supervised_data_leash_15min.ps1 -NoInitFromBest -ForcePretrain -StrictMuFreeze -HardGateThreshold -LearnedGateTemp  # + learnable gate temperature
#   .\scripts\run_biochem_supervised_data_leash_15min.ps1 -Epochs 20 -DryRun
#
# Success signals in console / run.jsonl:
#   - Startup: "BIOCHEM_SUPERVISED_DATA_LEASH=1" and NO "BIOCHEM_LOSS_ISOLATE='MU_LOG'"
#   - Train lines: L_Back reflects L_Data_Kine + L_Data_Bio (not isolate-only)
#   - val mu_log_mae / wall trend vs baseline ~0.31 / ~1.48 (20260524 sentinel ep34)
#
param(
    [int] $Epochs = 26,
    [int] $ValEvery = 2,
    [switch] $ForcePretrain,
    [switch] $NoInitFromBest,
    [switch] $StrictMuFreeze,
    [switch] $HardGateThreshold,
    [switch] $LearnedGateTemp,
    [switch] $DryRun,
    [string[]] $ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$hostName = $env:COMPUTERNAME
$TeacherBest = Join-Path $RepoRoot "outputs\biochem\biochem_teacher_best_high_mu.pth"
$TeacherBestLegacy = Join-Path $RepoRoot "outputs\biochem\biochem_teacher_best.pth"
$PostPretrain = Join-Path $RepoRoot "outputs\biochem\biochem_post_pretrain.pth"

Get-ChildItem Env:BIOCHEM_* -ErrorAction SilentlyContinue | ForEach-Object {
    Remove-Item "Env:$($_.Name)" -ErrorAction SilentlyContinue
}

# Sentinel architecture + wall weights (preset applies first; leash overrides loss graph).
$env:BIOCHEM_TRAIN_MODE = "new"
$env:BIOCHEM_PRESET = "sweep_wall_sentinel"
$env:BIOCHEM_WALL_HEAD_ISOLATE_GEOM = "1"
$env:BIOCHEM_STOP_AFTER_TEACHER = "1"
$env:BIOCHEM_TEACHER_EPOCHS = "$Epochs"
$env:BIOCHEM_EPOCHS = "$Epochs"
$env:BIOCHEM_CLI_TEACHER_EPOCHS = "$Epochs"
$env:BIOCHEM_TEACHER_VAL_EVERY = "$ValEvery"
$env:BIOCHEM_VAL_TIME_STRIDE = "10"
$env:BIOCHEM_TEACHER_KEEP_GLOBAL_BEST = "1"
$env:BIOCHEM_DEBUG = "0"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:BIOCHEM_DATALOADER_WORKERS = "0"
$env:BIOCHEM_PIN_MEMORY = "0"

# Middle-ground fix (applied in Python after preset bundle).
$env:BIOCHEM_SUPERVISED_DATA_LEASH = "1"
$env:BIOCHEM_LOSS_DATA_ONLY = "1"
Remove-Item Env:BIOCHEM_LOSS_ISOLATE -ErrorAction SilentlyContinue
$env:BIOCHEM_DETACH_MACRO_STATE = "0"
$env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "2.0"

# Surgical bulk-fluid lock (re-applied in Python after sentinel preset).
$env:BIOCHEM_BULK_FLUID_SURGICAL_FIX = "1"
$env:BIOCHEM_DELTA_MU_LOG_CLIP_BULK = "0.05"
$env:BIOCHEM_USE_BIO_GATE_SUPPRESSOR = "1"
$env:BIOCHEM_BIO_SUPPRESSOR_GATE_FLOOR = "0.0"

if ($StrictMuFreeze) {
    $env:BIOCHEM_TRAIN_ODE = "0"
    $env:BIOCHEM_TRAIN_BIO_ENCODER = "0"
    $env:BIOCHEM_TRAIN_KIN_LORA = "0"
    $env:BIOCHEM_TRAIN_BIO_DECODER = "0"
    $env:BIOCHEM_ODE_WARMUP_EPOCHS = "999"
}

if ($HardGateThreshold) {
    $env:BIOCHEM_MU_TRIGGER_GATE_HARD_THRESH = "0.15"
    $env:BIOCHEM_MU_TRIGGER_GATE_HARD_STEEPNESS = "10.0"
    $env:BIOCHEM_MU_SOFT_GATE_SCOPE = "wall_only"
    $env:BIOCHEM_TRIGGER_GATE_MIN = "0"
    $env:BIOCHEM_WALL_GATE_MIN = "0"
}
if ($LearnedGateTemp) {
    if (-not $HardGateThreshold) {
        Write-Host "-LearnedGateTemp requires -HardGateThreshold; enabling it." -ForegroundColor Yellow
        $HardGateThreshold = $true
        $env:BIOCHEM_MU_TRIGGER_GATE_HARD_THRESH = "0.15"
        $env:BIOCHEM_MU_SOFT_GATE_SCOPE = "wall_only"
        $env:BIOCHEM_TRIGGER_GATE_MIN = "0"
        $env:BIOCHEM_WALL_GATE_MIN = "0"
    }
    $env:BIOCHEM_MU_GATE_LEARNED_TEMP = "1"
}

if ($NoInitFromBest) {
    if ($ForcePretrain) {
        $env:BIOCHEM_SKIP_PRETRAIN = "0"
        $env:BIOCHEM_REUSE_LAST_PRETRAIN = "0"
    } elseif (Test-Path $PostPretrain) {
        $env:BIOCHEM_SKIP_PRETRAIN = "1"
        $env:BIOCHEM_REUSE_LAST_PRETRAIN = "1"
        Write-Host "Warm-start post-pretrain: $PostPretrain" -ForegroundColor Cyan
    } else {
        $env:BIOCHEM_SKIP_PRETRAIN = "0"
        $env:BIOCHEM_REUSE_LAST_PRETRAIN = "0"
        Write-Host "No post-pretrain; running AE+ODE first." -ForegroundColor Yellow
    }
} else {
    $ckpt = if (Test-Path $TeacherBest) { $TeacherBest } elseif (Test-Path $TeacherBestLegacy) { $TeacherBestLegacy } else { $null }
    if ($null -eq $ckpt) {
        Write-Host "No teacher best checkpoint; use -NoInitFromBest or train sentinel first." -ForegroundColor Red
        exit 1
    }
    $env:BIOCHEM_SKIP_PRETRAIN = "1"
    $env:BIOCHEM_REUSE_LAST_PRETRAIN = "0"
    Write-Host "Init from best teacher: $ckpt" -ForegroundColor Cyan
}

$tagParts = @()
if ($NoInitFromBest) { $tagParts += "cold" } else { $tagParts += "ws" }
if ($StrictMuFreeze) { $tagParts += "mufreeze" }
if ($HardGateThreshold) { $tagParts += "hardgate" }
if ($LearnedGateTemp) { $tagParts += "learntemp" }
$runTag = ($tagParts -join "_")
$runNote = "WS_sentinel_data_leash_bulklock_${runTag}_ep${Epochs}_${hostName}"
$env:BIOCHEM_RUN_NOTE = $runNote

Write-Host ""
Write-Host "Supervised data leash ~15min | ep=$Epochs val_every=$ValEvery | run_note=$runNote" -ForegroundColor Cyan
Write-Host "  DATA_ONLY=1  ISOLATE=(unset)  DETACH_MACRO=0  W_MuSI=2.0" -ForegroundColor DarkGray
Write-Host "  BULK_CLIP=0.05  BIO_GATE_SUPPRESSOR=1  GATE_FLOOR=0.0" -ForegroundColor DarkGray
if ($StrictMuFreeze) {
    Write-Host "  STRICT_MU_FREEZE: TRAIN_ODE=0 TRAIN_BIO_ENC=0 TRAIN_KIN_LORA=0 TRAIN_BIO_DEC=0" -ForegroundColor DarkGray
}
if ($HardGateThreshold) {
    $steepMsg = if ($LearnedGateTemp) { "LEARNED_TEMP=1" } else { "STEEPNESS=10" }
    Write-Host "  HARD_GATE_THRESH=0.15  SCOPE=wall_only  $steepMsg  GATE_MINS=0 (wall soft cutoff only)" -ForegroundColor DarkGray
}
Write-Host ""

$cmd = @(
    "-m", "src.training.train_biochem_corrector",
    "--run-name", $runNote,
    "--epochs", "$Epochs",
    "--save-best",
    "--new"
)
if ($NoInitFromBest) {
  # Do not pass --skip-pretrain: env/CLI must allow AE+ODE unless reusing post_pretrain.pth.
  if ($env:BIOCHEM_SKIP_PRETRAIN -eq "1") {
    $cmd += "--skip-pretrain"
  }
} else {
  $cmd += "--skip-pretrain"
  $cmd += "--init-from-best"
}
if ($ExtraArgs.Count -gt 0) { $cmd += $ExtraArgs }

if ($DryRun) {
    Write-Host "DryRun: python $($cmd -join ' ')" -ForegroundColor DarkGray
    exit 0
}

python @cmd
exit $LASTEXITCODE
