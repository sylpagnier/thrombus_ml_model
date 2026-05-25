# K4 Step 1: train wall delta head only (SDF geometry); clot/split tail frozen; no explicit gelation.
# Fresh start (delete outputs/biochem/*.pth first). Prereq: outputs/kinematics/kinematics_best.pth
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_k4_wall_head_only.ps1"
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_k4_wall_head_only.ps1" -Epochs 18

param(
    [int] $Epochs = 12,
    [int] $ValEvery = 3,
    [switch] $OomSafe = $true
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$KinCkpt = Join-Path $RepoRoot "outputs\kinematics\kinematics_best.pth"
if (-not (Test-Path $KinCkpt)) {
    Write-Host "Missing Stage-A ckpt: $KinCkpt" -ForegroundColor Red
    exit 1
}

$env:KINEMATICS_USE_HARD_BCS = "1"
$env:KINEMATICS_USE_WIDTH_PRIORS = "1"

$env:BIOCHEM_NO_TEACHER_DEFAULTS = "1"
$env:BIOCHEM_STOCK_DEFAULTS = "0"
$env:BIOCHEM_COMPLEXITY_STEP = "2"
$env:BIOCHEM_LOSS_DATA_ONLY = "1"
$env:BIOCHEM_LOSS_ISOLATE = "MU_LOG_WALL"
$env:BIOCHEM_STOP_AFTER_TEACHER = "1"
$env:BIOCHEM_TEACHER_KEEP_GLOBAL_BEST = "1"

# Sentinel-style split architecture; wall branch only trains (requires DELTA+SPLIT in forward).
$env:BIOCHEM_USE_DELTA_MU_HEAD = "1"
$env:BIOCHEM_USE_SPLIT_MU_HEAD = "1"
$env:BIOCHEM_USE_WALL_DELTA_HEAD = "1"
Remove-Item Env:BIOCHEM_MU_TRAIN_CLOT_ONLY -ErrorAction SilentlyContinue
$env:BIOCHEM_MU_TRAIN_WALL_ONLY = "1"
$env:BIOCHEM_MU_DISABLE_EXPLICIT_GELATION = "1"
$env:BIOCHEM_GELATION_PRIOR_GATE = "0"
$env:BIOCHEM_WALL_HEAD_ISOLATE_GEOM = "1"
$env:BIOCHEM_WALL_GATE_MIN = "0.06"
$env:BIOCHEM_MU_LOG_WALL_WEIGHT = "4.0"
$env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "0.25"
$env:BIOCHEM_MU_LOG_HIGH_WEIGHT = "0.0"
$env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "0.0"

$env:BIOCHEM_TRAIN_MU_ENCODER = "0"
$env:BIOCHEM_USE_MU_PATH_GROUP = "1"
$env:BIOCHEM_TRAIN_ODE = "0"
$env:BIOCHEM_TRAIN_BIO_ENCODER = "0"
$env:BIOCHEM_TRAIN_BIO_DECODER = "0"
$env:BIOCHEM_TRAIN_KIN_LORA = "0"
$env:BIOCHEM_TEACHER_FORCE_MIN = "1.0"
$env:BIOCHEM_TEACHER_MU_RATIO_MAX = "80.0"
$env:BIOCHEM_DETACH_MACRO_STATE = "1"

Remove-Item Env:BIOCHEM_SUPERVISED_DATA_LEASH -ErrorAction SilentlyContinue
Remove-Item Env:BIOCHEM_PRESET -ErrorAction SilentlyContinue

$env:BIOCHEM_TEACHER_EPOCHS = "$Epochs"
$env:BIOCHEM_EPOCHS = "$Epochs"
$env:BIOCHEM_CLI_TEACHER_EPOCHS = "$Epochs"
$env:BIOCHEM_TEACHER_VAL_EVERY = "$ValEvery"
$env:BIOCHEM_VAL_TIME_STRIDE = "10"
$env:BIOCHEM_RUN_NOTE = "K4_wall_head_only"

$env:BIOCHEM_SKIP_PRETRAIN = "0"
$env:BIOCHEM_REUSE_LAST_PRETRAIN = "0"
$env:BIOCHEM_AE_EPOCHS = "14"
$env:BIOCHEM_ODE_RXN_EPOCHS = "12"

if ($OomSafe) {
    $env:BIOCHEM_TBPTT_MAX_WINDOW = "5"
    $env:BIOCHEM_TBPTT_WINDOW_CURRICULUM = "0"
    $env:BIOCHEM_DATALOADER_WORKERS = "0"
    $env:BIOCHEM_PIN_MEMORY = "0"
    $env:BIOCHEM_KIN_GRADIENT_CHECKPOINT = "1"
    $env:BIOCHEM_ODE_GRADIENT_CHECKPOINT = "1"
    $env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "8"
    Write-Host "4GB OomSafe: TBPTT=5 workers=0" -ForegroundColor DarkGray
}

Write-Host "K4: wall delta head only | MU_LOG_WALL | AE+ODE pretrain then teacher" -ForegroundColor Cyan

python -m src.training.train_biochem_corrector --new --run-name K4_wall_head_only --epochs $Epochs --save-best
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
