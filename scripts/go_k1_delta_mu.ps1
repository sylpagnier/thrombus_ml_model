# K1: K0 parity stack + train Delta-log-mu head under DATA_KINE (TF=1, no explicit gelation).
# 4 GiB RTX 500: use default -OomSafe (TBPTT=5, workers=0, kin checkpointing).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_k1_delta_mu.ps1"
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_k1_delta_mu.ps1" -Resume
#
# Physics under test (unchanged by -OomSafe): Carreau baseline + delta_log_mu head,
# no explicit FI/Mat gelation in mu_eff, DATA_KINE backward, TF=1, mu_encoder trainable.

param(
    [switch] $Resume,
    [switch] $OomSafe = $true
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# --- Stage-A parity (kinematics manifest / checkpoint) ---
$env:KINEMATICS_USE_HARD_BCS = "1"
$env:KINEMATICS_USE_WIDTH_PRIORS = "1"

# --- K1 physics / loss (do not change when fixing OOM) ---
$env:BIOCHEM_MU_DISABLE_EXPLICIT_GELATION = "1"
$env:BIOCHEM_USE_DELTA_MU_HEAD = "1"
$env:BIOCHEM_USE_SPLIT_MU_HEAD = "0"
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
$env:BIOCHEM_RUN_NOTE = "K1_delta_mu_data_kine"

# --- Memory only (same forward physics; shorter TBPTT adjoint / less host RAM) ---
if ($OomSafe) {
    $env:BIOCHEM_TBPTT_MAX_WINDOW = "5"
    $env:BIOCHEM_TBPTT_WINDOW_CURRICULUM = "0"
    $env:BIOCHEM_DATALOADER_WORKERS = "0"
    $env:BIOCHEM_PIN_MEMORY = "0"
    $env:BIOCHEM_KIN_GRADIENT_CHECKPOINT = "1"
    $env:BIOCHEM_ODE_GRADIENT_CHECKPOINT = "1"
    $env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "8"
    Write-Host "4GB OomSafe: TBPTT=5 workers=0 kin_ckpt=1 RK4=8 (physics flags unchanged)" -ForegroundColor DarkGray
} else {
    $env:BIOCHEM_TBPTT_MAX_WINDOW = "12"
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
    $env:BIOCHEM_INIT_FROM_BEST = "1"
    $pyArgs = @("-m", "src.training.train_biochem_corrector", "--new", "--run-name", "K1_delta_mu_data_kine", "--epochs", "12", "--save-best")
}

python @pyArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
