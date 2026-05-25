# K6: unified split+wall heads trained together (no K4/K5 staging).
# Matches the ~0.47 historical recipe: sweep_wall_sentinel + SUPERVISED_DATA_LEASH (step-2 joint backward).
# NOTE: leash forces BIOCHEM_LOSS_DATA_ONLY=1 — NOT step-3 Kendall multitask. Use -Multitask to try step-3 (risky on 4GB).
#
# Fresh (recommended after K4/K5): delete outputs/biochem/*.pth, keep kinematics_best.pth
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_k6_unified_kitchen_sink.ps1" -Fresh
#
# Warm-start teacher (if you have a good sentinel/leash ckpt):
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_k6_unified_kitchen_sink.ps1"
#
# True step-3 multitask (no leash — like K5, higher wall/clot risk):
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_k6_unified_kitchen_sink.ps1" -Fresh -Multitask

param(
    [int] $Epochs = 26,
    [int] $ValEvery = 2,
    [switch] $Fresh,
    [switch] $Multitask,
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

# Clear staged K4/K5 scope flags if a prior leg left them in the shell.
Remove-Item Env:BIOCHEM_MU_TRAIN_WALL_ONLY -ErrorAction SilentlyContinue
Remove-Item Env:BIOCHEM_MU_TRAIN_CLOT_ONLY -ErrorAction SilentlyContinue

$env:KINEMATICS_USE_HARD_BCS = "1"
$env:KINEMATICS_USE_WIDTH_PRIORS = "1"

$env:BIOCHEM_PRESET = "sweep_wall_sentinel"
$env:BIOCHEM_STOP_AFTER_TEACHER = "1"
$env:BIOCHEM_NO_TEACHER_DEFAULTS = "1"
$env:BIOCHEM_STOCK_DEFAULTS = "0"
$env:BIOCHEM_TEACHER_KEEP_GLOBAL_BEST = "1"

# Unified architecture: both heads trainable (default — do not set MU_TRAIN_WALL_ONLY / CLOT_ONLY).
$env:BIOCHEM_USE_DELTA_MU_HEAD = "1"
$env:BIOCHEM_USE_SPLIT_MU_HEAD = "1"
$env:BIOCHEM_USE_WALL_DELTA_HEAD = "1"
$env:BIOCHEM_TRAIN_MU_ENCODER = "1"
$env:BIOCHEM_USE_MU_PATH_GROUP = "1"
$env:BIOCHEM_WALL_HEAD_ISOLATE_GEOM = "1"

# Forward: explicit gelation + sentinel spatial mask (bulk triggers suppressed).
$env:BIOCHEM_MU_DISABLE_EXPLICIT_GELATION = "0"
$env:BIOCHEM_GELATION_PRIOR_GATE = "1"
$env:BIOCHEM_TRIGGER_GATE_MIN = "0.06"
$env:BIOCHEM_WALL_GATE_MIN = "0.06"

$env:BIOCHEM_BULK_FLUID_SURGICAL_FIX = "1"
$env:BIOCHEM_DELTA_MU_LOG_CLIP_BULK = "0.05"
$env:BIOCHEM_USE_BIO_GATE_SUPPRESSOR = "1"
$env:BIOCHEM_BIO_SUPPRESSOR_GATE_FLOOR = "0.0"

$env:BIOCHEM_TRAIN_ODE = "0"
$env:BIOCHEM_TRAIN_BIO_ENCODER = "0"
$env:BIOCHEM_TRAIN_BIO_DECODER = "0"
$env:BIOCHEM_TRAIN_KIN_LORA = "0"
$env:BIOCHEM_TEACHER_FORCE_MIN = "1.0"
$env:BIOCHEM_TEACHER_MU_RATIO_MAX = "80.0"
$env:BIOCHEM_TEACHER_MAX_RAW_GRAD_L2 = "50000"

$env:BIOCHEM_TEACHER_EPOCHS = "$Epochs"
$env:BIOCHEM_EPOCHS = "$Epochs"
$env:BIOCHEM_CLI_TEACHER_EPOCHS = "$Epochs"
$env:BIOCHEM_TEACHER_VAL_EVERY = "$ValEvery"
$env:BIOCHEM_VAL_TIME_STRIDE = "10"
$env:BIOCHEM_RUN_NOTE = "K6_unified_kitchen_sink"

$pyArgs = @(
    "-m", "src.training.train_biochem_corrector",
    "--new",
    "--epochs", "$Epochs",
    "--save-best",
    "--run-name", "K6_unified_kitchen_sink"
)

if ($Multitask) {
    $env:BIOCHEM_RUN_NOTE = "K6_unified_step3_multitask"
    $env:BIOCHEM_COMPLEXITY_STEP = "3"
    $env:BIOCHEM_LOSS_DATA_ONLY = "0"
    Remove-Item Env:BIOCHEM_SUPERVISED_DATA_LEASH -ErrorAction SilentlyContinue
    Remove-Item Env:BIOCHEM_LOSS_ISOLATE -ErrorAction SilentlyContinue
    $env:BIOCHEM_DETACH_MACRO_STATE = "1"
    $pyArgs[-2] = "K6_unified_step3_multitask"
    Write-Host "K6 -Multitask: step-3 Kendall backward (NO data leash)." -ForegroundColor Yellow
} else {
    # Leash hook (Python) clears LOSS_ISOLATE and sets LOSS_DATA_ONLY=1, DETACH_MACRO=0.
    $env:BIOCHEM_COMPLEXITY_STEP = "2"
    $env:BIOCHEM_SUPERVISED_DATA_LEASH = "1"
    $env:BIOCHEM_LOSS_DATA_ONLY = "1"
    Remove-Item Env:BIOCHEM_LOSS_ISOLATE -ErrorAction SilentlyContinue
    Write-Host "K6 default: sentinel + SUPERVISED_DATA_LEASH (step-2 joint; targets ~0.47 tier)." -ForegroundColor Cyan
}

if ($OomSafe) {
    $env:BIOCHEM_TBPTT_MAX_WINDOW = "5"
    $env:BIOCHEM_TBPTT_WINDOW_CURRICULUM = "0"
    $env:BIOCHEM_DATALOADER_WORKERS = "0"
    $env:BIOCHEM_PIN_MEMORY = "0"
    $env:BIOCHEM_KIN_GRADIENT_CHECKPOINT = "1"
    $env:BIOCHEM_ODE_GRADIENT_CHECKPOINT = "1"
    $env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "8"
    Write-Host "4GB OomSafe: TBPTT=5 workers=0 (leash uses DETACH_MACRO=0 → more VRAM)" -ForegroundColor DarkGray
}

$TeacherBest = Join-Path $RepoRoot "outputs\biochem\biochem_teacher_best_high_mu.pth"
if ($Fresh) {
    Remove-Item Env:BIOCHEM_INIT_FROM_BEST -ErrorAction SilentlyContinue
    $env:BIOCHEM_SKIP_PRETRAIN = "0"
    $env:BIOCHEM_REUSE_LAST_PRETRAIN = "0"
    $env:BIOCHEM_AE_EPOCHS = "14"
    $env:BIOCHEM_ODE_RXN_EPOCHS = "12"
    Write-Host "Fresh: AE+ODE pretrain then unified teacher (delete old biochem *.pth first)." -ForegroundColor Cyan
} elseif (Test-Path $TeacherBest) {
    $env:BIOCHEM_SKIP_PRETRAIN = "1"
    $env:BIOCHEM_REUSE_LAST_PRETRAIN = "0"
    $pyArgs += "--skip-pretrain", "--init-from-best"
    Write-Host "Warm-start teacher: $TeacherBest" -ForegroundColor DarkGray
} else {
    $env:BIOCHEM_SKIP_PRETRAIN = "0"
    $env:BIOCHEM_REUSE_LAST_PRETRAIN = "0"
    $env:BIOCHEM_AE_EPOCHS = "14"
    $env:BIOCHEM_ODE_RXN_EPOCHS = "12"
    Write-Host "No teacher ckpt; running AE+ODE then teacher." -ForegroundColor Yellow
}

python @pyArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
