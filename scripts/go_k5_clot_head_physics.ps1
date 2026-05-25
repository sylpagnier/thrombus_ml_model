# K5 Step 2: freeze wall head; train clot split tail/gate + explicit gelation + step-3 multitask.
# Run AFTER K4. Warm-starts biochem_teacher_best_high_mu.pth from K4 wall training.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_k5_clot_head_physics.ps1"
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_k5_clot_head_physics.ps1" -Epochs 15

param(
    [int] $Epochs = 15,
    [int] $ValEvery = 2,
    [switch] $OomSafe = $true
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$TeacherBest = Join-Path $RepoRoot "outputs\biochem\biochem_teacher_best_high_mu.pth"
if (-not (Test-Path $TeacherBest)) {
    Write-Host "Missing K4 teacher ckpt: $TeacherBest (run go_k4_wall_head_only.ps1 first)." -ForegroundColor Red
    exit 1
}

$env:KINEMATICS_USE_HARD_BCS = "1"
$env:KINEMATICS_USE_WIDTH_PRIORS = "1"

$env:BIOCHEM_NO_TEACHER_DEFAULTS = "1"
$env:BIOCHEM_STOCK_DEFAULTS = "0"
$env:BIOCHEM_COMPLEXITY_STEP = "3"
$env:BIOCHEM_LOSS_DATA_ONLY = "0"
Remove-Item Env:BIOCHEM_LOSS_ISOLATE -ErrorAction SilentlyContinue
$env:BIOCHEM_STOP_AFTER_TEACHER = "1"
$env:BIOCHEM_TEACHER_KEEP_GLOBAL_BEST = "1"

$env:BIOCHEM_USE_DELTA_MU_HEAD = "1"
$env:BIOCHEM_USE_SPLIT_MU_HEAD = "1"
$env:BIOCHEM_USE_WALL_DELTA_HEAD = "1"
Remove-Item Env:BIOCHEM_MU_TRAIN_WALL_ONLY -ErrorAction SilentlyContinue
$env:BIOCHEM_MU_TRAIN_CLOT_ONLY = "1"
$env:BIOCHEM_MU_DISABLE_EXPLICIT_GELATION = "0"
$env:BIOCHEM_GELATION_PRIOR_GATE = "1"
$env:BIOCHEM_WALL_HEAD_ISOLATE_GEOM = "1"
$env:BIOCHEM_TRIGGER_GATE_MIN = "0.06"
$env:BIOCHEM_WALL_GATE_MIN = "0.06"
$env:BIOCHEM_TEACHER_MAX_RAW_GRAD_L2 = "50000"

$env:BIOCHEM_TRAIN_MU_ENCODER = "1"
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
$env:BIOCHEM_RUN_NOTE = "K5_clot_head_physics"

$env:BIOCHEM_SKIP_PRETRAIN = "1"
$env:BIOCHEM_REUSE_LAST_PRETRAIN = "0"

if ($OomSafe) {
    $env:BIOCHEM_TBPTT_MAX_WINDOW = "5"
    $env:BIOCHEM_TBPTT_WINDOW_CURRICULUM = "0"
    $env:BIOCHEM_DATALOADER_WORKERS = "0"
    $env:BIOCHEM_PIN_MEMORY = "0"
    $env:BIOCHEM_KIN_GRADIENT_CHECKPOINT = "1"
    $env:BIOCHEM_ODE_GRADIENT_CHECKPOINT = "1"
    $env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "8"
    Write-Host "4GB OomSafe: TBPTT=5 workers=0 (step-3 multitask; watch VRAM)" -ForegroundColor DarkGray
}

Write-Host "K5: clot split head + gelation + step-3 | init-from $TeacherBest" -ForegroundColor Cyan

python -m src.training.train_biochem_corrector --new --skip-pretrain --init-from-best --run-name K5_clot_head_physics --epochs $Epochs --save-best
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
