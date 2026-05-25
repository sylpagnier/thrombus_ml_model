# K10a Step A: steady GINO-DEQ mu at t=0 (BIOCHEM_MU_IC_STEADY_KIN=1) + K1 training stack.
# Prereq: outputs/kinematics/kinematics_best.pth. Delete outputs/biochem/*.pth for fresh run.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_k10a_ic_steady_kin.ps1" -Fresh

param(
    [switch] $Fresh,
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

$env:BIOCHEM_MU_IC_STEADY_KIN = "1"
$env:BIOCHEM_MU_DISABLE_EXPLICIT_GELATION = "1"
$env:BIOCHEM_USE_DELTA_MU_HEAD = "1"
$env:BIOCHEM_USE_SPLIT_MU_HEAD = "0"
Remove-Item Env:BIOCHEM_USE_WALL_DELTA_HEAD -ErrorAction SilentlyContinue
$env:BIOCHEM_TRAIN_MU_ENCODER = "1"
$env:BIOCHEM_USE_MU_PATH_GROUP = "1"
$env:BIOCHEM_TRAIN_ODE = "0"
$env:BIOCHEM_TRAIN_BIO_ENCODER = "0"
$env:BIOCHEM_TRAIN_BIO_DECODER = "0"
$env:BIOCHEM_TRAIN_KIN_LORA = "0"

$env:BIOCHEM_LOSS_ISOLATE = "DATA_KINE"
$env:BIOCHEM_LOSS_DATA_ONLY = "1"
$env:BIOCHEM_COMPLEXITY_STEP = "2"
$env:BIOCHEM_STOP_AFTER_TEACHER = "1"
$env:BIOCHEM_TEACHER_FORCE_MIN = "1.0"
$env:BIOCHEM_TEACHER_MU_RATIO_MAX = "80.0"
$env:BIOCHEM_DETACH_MACRO_STATE = "1"
$env:BIOCHEM_VAL_TIME_STRIDE = "10"
$env:BIOCHEM_TEACHER_VAL_EVERY = "3"
$env:BIOCHEM_TEACHER_EPOCHS = "12"
$env:BIOCHEM_EPOCHS = "12"
$env:BIOCHEM_CLI_TEACHER_EPOCHS = "12"
$env:BIOCHEM_NO_TEACHER_DEFAULTS = "1"
$env:BIOCHEM_STOCK_DEFAULTS = "0"
$env:BIOCHEM_RUN_NOTE = "K10a_ic_steady_kin_t0"

Remove-Item Env:BIOCHEM_SUPERVISED_DATA_LEASH,Env:BIOCHEM_PRESET,Env:BIOCHEM_MU_TRAIN_WALL_ONLY,Env:BIOCHEM_MU_TRAIN_CLOT_ONLY -ErrorAction SilentlyContinue

if ($OomSafe) {
    $env:BIOCHEM_TBPTT_MAX_WINDOW = "5"
    $env:BIOCHEM_TBPTT_WINDOW_CURRICULUM = "0"
    $env:BIOCHEM_DATALOADER_WORKERS = "0"
    $env:BIOCHEM_PIN_MEMORY = "0"
    $env:BIOCHEM_KIN_GRADIENT_CHECKPOINT = "1"
    $env:BIOCHEM_ODE_GRADIENT_CHECKPOINT = "1"
    $env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "8"
}

$env:BIOCHEM_RESUME = "0"
$env:BIOCHEM_SKIP_PRETRAIN = "0"
$env:BIOCHEM_REUSE_LAST_PRETRAIN = "0"
$env:BIOCHEM_AE_EPOCHS = "14"
$env:BIOCHEM_ODE_RXN_EPOCHS = "12"
Remove-Item Env:BIOCHEM_INIT_FROM_BEST -ErrorAction SilentlyContinue

$pyArgs = @(
    "-m", "src.training.train_biochem_corrector",
    "--new", "--epochs", "12", "--save-best",
    "--run-name", "K10a_ic_steady_kin_t0"
)
if ($Fresh) { Write-Host "Fresh K10a: AE+ODE+teacher (MU_IC_STEADY_KIN=1)." -ForegroundColor Cyan }

python @pyArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
