# K2: K1 stack + explicit FI/Mat gelation + full multitask backward (not DATA_KINE isolate).
# Warm-start from K1 teacher weights; skip pretrain. 4 GiB: default -OomSafe.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_k2_physics_triggers_on.ps1"
#
# Prereq: K1 complete (`biochem_teacher_best_high_mu.pth` from K1_delta_mu_data_kine).

param(
    [switch] $Resume,
    [switch] $OomSafe = $true,
    [switch] $ColdStart
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$env:KINEMATICS_USE_HARD_BCS = "1"
$env:KINEMATICS_USE_WIDTH_PRIORS = "1"

# --- K2 physics (vs K1) ---
$env:BIOCHEM_MU_DISABLE_EXPLICIT_GELATION = "0"
$env:BIOCHEM_GELATION_PRIOR_GATE = "0"
$env:BIOCHEM_USE_DELTA_MU_HEAD = "1"
$env:BIOCHEM_USE_SPLIT_MU_HEAD = "0"
$env:BIOCHEM_USE_WALL_DELTA_HEAD = "0"
$env:BIOCHEM_TRAIN_MU_ENCODER = "1"
$env:BIOCHEM_USE_MU_PATH_GROUP = "1"
$env:BIOCHEM_TRAIN_ODE = "0"
$env:BIOCHEM_TRAIN_BIO_ENCODER = "0"
$env:BIOCHEM_TRAIN_BIO_DECODER = "0"
$env:BIOCHEM_TRAIN_KIN_LORA = "0"
$env:BIOCHEM_TEACHER_FORCE_MIN = "1.0"
$env:BIOCHEM_TEACHER_MU_RATIO_MAX = "80.0"

# --- Multitask backward (Kendall sum: data + PDE / wall / viscosity_reg, etc.) ---
$env:BIOCHEM_COMPLEXITY_STEP = "3"
$env:BIOCHEM_LOSS_DATA_ONLY = "0"
Remove-Item Env:BIOCHEM_LOSS_ISOLATE -ErrorAction SilentlyContinue

$env:BIOCHEM_STOP_AFTER_TEACHER = "1"
$env:BIOCHEM_DETACH_MACRO_STATE = "1"
$env:BIOCHEM_VAL_TIME_STRIDE = "10"
$env:BIOCHEM_TEACHER_VAL_EVERY = "3"
$env:BIOCHEM_TEACHER_EPOCHS = "12"
$env:BIOCHEM_EPOCHS = "12"
$env:BIOCHEM_CLI_TEACHER_EPOCHS = "12"
$env:BIOCHEM_NO_TEACHER_DEFAULTS = "1"
$env:BIOCHEM_STOCK_DEFAULTS = "0"
$env:BIOCHEM_RUN_NOTE = "K2_physics_triggers_on"

if ($OomSafe) {
    $env:BIOCHEM_TBPTT_MAX_WINDOW = "5"
    $env:BIOCHEM_TBPTT_WINDOW_CURRICULUM = "0"
    $env:BIOCHEM_DATALOADER_WORKERS = "0"
    $env:BIOCHEM_PIN_MEMORY = "0"
    $env:BIOCHEM_KIN_GRADIENT_CHECKPOINT = "1"
    $env:BIOCHEM_ODE_GRADIENT_CHECKPOINT = "1"
    $env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "8"
    Write-Host "4GB OomSafe: TBPTT=5 workers=0 kin_ckpt=1 RK4=8" -ForegroundColor DarkGray
} else {
    $env:BIOCHEM_TBPTT_MAX_WINDOW = "8"
}

if ($Resume) {
    $env:BIOCHEM_RESUME = "1"
    Remove-Item Env:BIOCHEM_INIT_FROM_BEST -ErrorAction SilentlyContinue
    $pyArgs = @("-m", "src.training.train_biochem_corrector", "--resume", "--epochs", "12", "--save-best")
    Write-Host "Resume: biochem_latest_checkpoint.pth (if present)" -ForegroundColor DarkGray
} else {
    $env:BIOCHEM_RESUME = "0"
    $env:BIOCHEM_SKIP_PRETRAIN = "1"
    $env:BIOCHEM_REUSE_LAST_PRETRAIN = "1"
    if ($ColdStart) {
        Remove-Item Env:BIOCHEM_INIT_FROM_BEST -ErrorAction SilentlyContinue
        Write-Host "ColdStart: random biochem init (not loading K1 ckpt)" -ForegroundColor Yellow
    } else {
        $env:BIOCHEM_INIT_FROM_BEST = "1"
        Write-Host "Warm-start: biochem_teacher_best_high_mu.pth (K1 weights)" -ForegroundColor DarkGray
    }
    $pyArgs = @(
        "-m", "src.training.train_biochem_corrector",
        "--new", "--run-name", "K2_physics_triggers_on",
        "--epochs", "12", "--save-best"
    )
}

Write-Host "K2: explicit gelation ON | multitask backward (COMPLEXITY_STEP=3, LOSS_DATA_ONLY=0)" -ForegroundColor Cyan

python @pyArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
