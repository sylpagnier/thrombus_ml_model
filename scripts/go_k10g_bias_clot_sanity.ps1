# K10g train sanity: wide K10e band + delta-head bias ~17 ND + strong DATA_KINE (flow), 6 teacher ep.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_k10g_bias_clot_sanity.ps1" -Fresh

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
Remove-Item Env:BIOCHEM_MU_K10D_SIMPLE,Env:BIOCHEM_K10G_ORACLE_CLOTS -ErrorAction SilentlyContinue
$env:BIOCHEM_MU_IC_STEADY_KIN = "1"
$env:BIOCHEM_MU_DISABLE_EXPLICIT_GELATION = "1"
$env:BIOCHEM_USE_DELTA_MU_HEAD = "1"
$env:BIOCHEM_USE_SPLIT_MU_HEAD = "0"
$env:BIOCHEM_K10E_D_PEAK_ND = "0.008"
$env:BIOCHEM_K10E_SIGMA_ND = "0.008"
$env:BIOCHEM_K10E_SDF_MAX_ND = "0.04"
$env:BIOCHEM_K10E_MU_DELTA_ND_MAX = "30"
$env:BIOCHEM_K10E_DELTA_BIAS_ND = "17"
$env:BIOCHEM_K10E_CORONA_GROWTH = "1"
$env:BIOCHEM_K10E_CORONA_MIX = "0.4"
$env:BIOCHEM_TRAIN_MU_ENCODER = "1"
$env:BIOCHEM_USE_MU_PATH_GROUP = "1"
$env:BIOCHEM_TRAIN_ODE = "0"
$env:BIOCHEM_TRAIN_BIO_ENCODER = "0"
$env:BIOCHEM_TRAIN_BIO_DECODER = "0"
$env:BIOCHEM_TRAIN_KIN_LORA = "0"
$env:BIOCHEM_LOSS_ISOLATE = "K10E"
$env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "0.5"
$env:BIOCHEM_MU_LOG_HIGH_WEIGHT = "2.0"
$env:BIOCHEM_MU_LOG_ADJACENT_WEIGHT = "2.0"
$env:BIOCHEM_K10E_BULK_DELTA_WEIGHT = "0.0"
$env:BIOCHEM_K10E_DATA_KINE_WEIGHT = "1.0"
Remove-Item Env:BIOCHEM_MU_LOG_WALL_WEIGHT,Env:BIOCHEM_LOSS_DATA_ONLY -ErrorAction SilentlyContinue
$env:BIOCHEM_COMPLEXITY_STEP = "2"
$env:BIOCHEM_STOP_AFTER_TEACHER = "1"
$env:BIOCHEM_TEACHER_FORCE_MIN = "1.0"
$env:BIOCHEM_TEACHER_MU_RATIO_MAX = "80.0"
$env:BIOCHEM_DETACH_MACRO_STATE = "1"
$env:BIOCHEM_VAL_TIME_STRIDE = "10"
$env:BIOCHEM_TEACHER_VAL_EVERY = "2"
$env:BIOCHEM_TEACHER_EPOCHS = "6"
$env:BIOCHEM_EPOCHS = "6"
$env:BIOCHEM_CLI_TEACHER_EPOCHS = "6"
$env:BIOCHEM_NO_TEACHER_DEFAULTS = "1"
$env:BIOCHEM_STOCK_DEFAULTS = "0"
$env:BIOCHEM_RUN_NOTE = "K10g_bias_clot_sanity"
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
python -m src.training.train_biochem_corrector --new --epochs 6 --save-best --run-name K10g_bias_clot_sanity
exit $LASTEXITCODE
