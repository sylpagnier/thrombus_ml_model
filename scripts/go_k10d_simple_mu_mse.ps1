# K10d proof: mu_eff = mu_ss (steady kin DEQ) + softplus(learned dMu SI); backward = MU_MSE only.
# Prereq: outputs/kinematics/kinematics_best.pth. Delete outputs/biochem/*.pth before fresh run.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_k10d_simple_mu_mse.ps1"

param([switch] $Fresh, [switch] $OomSafe = $true)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
if (-not (Test-Path (Join-Path $RepoRoot "outputs\kinematics\kinematics_best.pth"))) {
    Write-Host "Missing kinematics_best.pth" -ForegroundColor Red
    exit 1
}

$env:KINEMATICS_USE_HARD_BCS = "1"
$env:KINEMATICS_USE_WIDTH_PRIORS = "1"
$env:BIOCHEM_MU_K10D_SIMPLE = "1"
$env:BIOCHEM_MU_DISABLE_EXPLICIT_GELATION = "1"
$env:BIOCHEM_USE_DELTA_MU_HEAD = "1"
$env:BIOCHEM_USE_SPLIT_MU_HEAD = "0"
$env:BIOCHEM_K10D_MU_DELTA_SI_MAX = "0.08"
Remove-Item Env:BIOCHEM_USE_WALL_DELTA_HEAD,Env:BIOCHEM_MU_IC_STEADY_KIN,Env:BIOCHEM_MU_ADDITIVE_DELTA -ErrorAction SilentlyContinue
$env:BIOCHEM_TRAIN_MU_ENCODER = "1"
$env:BIOCHEM_USE_MU_PATH_GROUP = "1"
$env:BIOCHEM_TRAIN_ODE = "0"
$env:BIOCHEM_TRAIN_BIO_ENCODER = "0"
$env:BIOCHEM_TRAIN_BIO_DECODER = "0"
$env:BIOCHEM_TRAIN_KIN_LORA = "0"
$env:BIOCHEM_LOSS_ISOLATE = "MU_MSE"
Remove-Item Env:BIOCHEM_LOSS_DATA_ONLY,Env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT,Env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT -ErrorAction SilentlyContinue
Remove-Item Env:BIOCHEM_MU_LOG_HIGH_WEIGHT,Env:BIOCHEM_MU_LOG_WALL_WEIGHT -ErrorAction SilentlyContinue
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
$env:BIOCHEM_RUN_NOTE = "K10d_simple_mu_mse"
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
if ($Fresh) { Remove-Item (Join-Path $RepoRoot "outputs\biochem\*.pth") -ErrorAction SilentlyContinue }
python -m src.training.train_biochem_corrector --new --epochs 12 --save-best --run-name K10d_simple_mu_mse
exit $LASTEXITCODE
