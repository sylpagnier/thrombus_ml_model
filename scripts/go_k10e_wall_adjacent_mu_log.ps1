# K10e: mu_eff = mu_ss + adj_mask * softplus(dMu_nd); LOSS_ISOLATE=K10E (log + high + adjacent + bulk + light DATA_KINE).
# Prereq: outputs/kinematics/kinematics_best.pth. Delete outputs/biochem/*.pth before a fresh run.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_k10e_wall_adjacent_mu_log.ps1" -Fresh

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
$env:BIOCHEM_MU_K10E_SIMPLE = "1"
Remove-Item Env:BIOCHEM_MU_K10D_SIMPLE -ErrorAction SilentlyContinue
$env:BIOCHEM_MU_IC_STEADY_KIN = "1"
$env:BIOCHEM_MU_DISABLE_EXPLICIT_GELATION = "1"
$env:BIOCHEM_USE_DELTA_MU_HEAD = "1"
$env:BIOCHEM_USE_SPLIT_MU_HEAD = "0"
Remove-Item Env:BIOCHEM_USE_WALL_DELTA_HEAD,Env:BIOCHEM_MU_ADDITIVE_DELTA -ErrorAction SilentlyContinue
# Wall-adjacent band (SDF peak ~1 layer off wall; zero on mask_wall nodes)
$env:BIOCHEM_K10E_D_PEAK_ND = "0.004"
$env:BIOCHEM_K10E_SIGMA_ND = "0.0035"
$env:BIOCHEM_K10E_SDF_MAX_ND = "0.02"
$env:BIOCHEM_K10E_MU_DELTA_ND_MAX = "18"
$env:BIOCHEM_K10E_CORONA_GROWTH = "1"
$env:BIOCHEM_K10E_CORONA_MIX = "0.4"
$env:BIOCHEM_TRAIN_MU_ENCODER = "1"
$env:BIOCHEM_USE_MU_PATH_GROUP = "1"
$env:BIOCHEM_TRAIN_ODE = "0"
$env:BIOCHEM_TRAIN_BIO_ENCODER = "0"
$env:BIOCHEM_TRAIN_BIO_DECODER = "0"
$env:BIOCHEM_TRAIN_KIN_LORA = "0"
$env:BIOCHEM_LOSS_ISOLATE = "K10E"
$env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "1.0"
$env:BIOCHEM_MU_LOG_HIGH_WEIGHT = "4.0"
$env:BIOCHEM_MU_LOG_ADJACENT_WEIGHT = "3.0"
$env:BIOCHEM_K10E_BULK_DELTA_WEIGHT = "2.0"
$env:BIOCHEM_K10E_DATA_KINE_WEIGHT = "0.25"
Remove-Item Env:BIOCHEM_MU_LOG_WALL_WEIGHT,Env:BIOCHEM_LOSS_DATA_ONLY,Env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT -ErrorAction SilentlyContinue
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
$env:BIOCHEM_RUN_NOTE = "K10e_wall_adjacent_mu_log"
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
python -m src.training.train_biochem_corrector --new --epochs 12 --save-best --run-name K10e_wall_adjacent_mu_log
exit $LASTEXITCODE
