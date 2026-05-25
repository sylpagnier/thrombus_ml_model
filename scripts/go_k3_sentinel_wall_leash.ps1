# K3: sentinel split-μ + data leash.
# Prereq (always): outputs/kinematics/kinematics_best.pth (Stage A).
# Warm-start (default): biochem_teacher_best_high_mu.pth. No biochem ckpts: use -Fresh.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_k3_sentinel_wall_leash.ps1" -Fresh
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_k3_sentinel_wall_leash.ps1" -Fresh -Epochs 26
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_k3_sentinel_wall_leash.ps1"

param(
    [int] $Epochs = 20,
    [int] $ValEvery = 2,
    [switch] $Resume,
    [switch] $OomSafe = $true,
    [switch] $Fresh,
    [switch] $ColdStart
)

$ErrorActionPreference = "Stop"
if ($ColdStart) { $Fresh = $true }
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$TeacherBest = Join-Path $RepoRoot "outputs\biochem\biochem_teacher_best_high_mu.pth"
$TeacherLegacy = Join-Path $RepoRoot "outputs\biochem\biochem_teacher_best.pth"

$env:KINEMATICS_USE_HARD_BCS = "1"
$env:KINEMATICS_USE_WIDTH_PRIORS = "1"

$env:BIOCHEM_PRESET = "sweep_wall_sentinel"
$env:BIOCHEM_COMPLEXITY_STEP = "2"
$env:BIOCHEM_STOP_AFTER_TEACHER = "1"
$env:BIOCHEM_NO_TEACHER_DEFAULTS = "1"
$env:BIOCHEM_STOCK_DEFAULTS = "0"
$env:BIOCHEM_TEACHER_KEEP_GLOBAL_BEST = "1"

$env:BIOCHEM_MU_DISABLE_EXPLICIT_GELATION = "1"
$env:BIOCHEM_GELATION_PRIOR_GATE = "1"
$env:BIOCHEM_USE_DELTA_MU_HEAD = "1"
$env:BIOCHEM_USE_SPLIT_MU_HEAD = "1"
$env:BIOCHEM_USE_WALL_DELTA_HEAD = "1"
$env:BIOCHEM_TRAIN_MU_ENCODER = "1"
$env:BIOCHEM_USE_MU_PATH_GROUP = "1"
$env:BIOCHEM_TEACHER_FORCE_MIN = "1.0"
$env:BIOCHEM_TEACHER_MU_RATIO_MAX = "80.0"

# Leash clears LOSS_ISOLATE; backward = L_Data_Kine + L_Data_Bio + μ anchors.
$env:BIOCHEM_SUPERVISED_DATA_LEASH = "1"
$env:BIOCHEM_LOSS_DATA_ONLY = "1"
Remove-Item Env:BIOCHEM_LOSS_ISOLATE -ErrorAction SilentlyContinue

$env:BIOCHEM_BULK_FLUID_SURGICAL_FIX = "1"
$env:BIOCHEM_DELTA_MU_LOG_CLIP_BULK = "0.05"
$env:BIOCHEM_USE_BIO_GATE_SUPPRESSOR = "1"
$env:BIOCHEM_BIO_SUPPRESSOR_GATE_FLOOR = "0.0"
$env:BIOCHEM_WALL_HEAD_ISOLATE_GEOM = "1"

# K3b fixes: allow μ+wall split-head grads (check is bio+mu L2, cap was 5000).
$env:BIOCHEM_TEACHER_MAX_RAW_GRAD_L2 = "50000"
$env:BIOCHEM_TRIGGER_GATE_MIN = "0.06"
$env:BIOCHEM_WALL_GATE_MIN = "0.06"

$env:BIOCHEM_TRAIN_ODE = "0"
$env:BIOCHEM_TRAIN_BIO_ENCODER = "0"
$env:BIOCHEM_TRAIN_BIO_DECODER = "0"
$env:BIOCHEM_TRAIN_KIN_LORA = "0"

$env:BIOCHEM_TEACHER_EPOCHS = "$Epochs"
$env:BIOCHEM_EPOCHS = "$Epochs"
$env:BIOCHEM_CLI_TEACHER_EPOCHS = "$Epochs"
$env:BIOCHEM_TEACHER_VAL_EVERY = "$ValEvery"
$env:BIOCHEM_VAL_TIME_STRIDE = "10"
$KinCkpt = Join-Path $RepoRoot "outputs\kinematics\kinematics_best.pth"
if (-not (Test-Path $KinCkpt)) {
    Write-Host "Missing Stage-A ckpt: $KinCkpt (train kinematics first)." -ForegroundColor Red
    exit 1
}

if ($OomSafe) {
    $env:BIOCHEM_TBPTT_MAX_WINDOW = "5"
    $env:BIOCHEM_TBPTT_WINDOW_CURRICULUM = "0"
    $env:BIOCHEM_DATALOADER_WORKERS = "0"
    $env:BIOCHEM_PIN_MEMORY = "0"
    $env:BIOCHEM_KIN_GRADIENT_CHECKPOINT = "1"
    $env:BIOCHEM_ODE_GRADIENT_CHECKPOINT = "1"
    $env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "8"
    Write-Host "4GB OomSafe: TBPTT=5 workers=0 (leash sets DETACH_MACRO=0)" -ForegroundColor DarkGray
}

if ($Resume) {
    $env:BIOCHEM_RESUME = "1"
    $pyArgs = @(
        "-m", "src.training.train_biochem_corrector",
        "--resume", "--epochs", "$Epochs", "--save-best"
    )
    Write-Host "Resume: biochem_latest_checkpoint.pth" -ForegroundColor DarkGray
} else {
    $env:BIOCHEM_RESUME = "0"
    $runName = "K3b_sentinel_wall_leash"
    $pyArgs = @(
        "-m", "src.training.train_biochem_corrector",
        "--new",
        "--epochs", "$Epochs", "--save-best"
    )

    $doFresh = $Fresh
    if (-not $doFresh) {
        if ((Test-Path $TeacherBest) -or (Test-Path $TeacherLegacy)) {
            $env:BIOCHEM_SKIP_PRETRAIN = "1"
            $env:BIOCHEM_REUSE_LAST_PRETRAIN = "0"
            $pyArgs += "--skip-pretrain", "--init-from-best"
            $ckpt = if (Test-Path $TeacherBest) { $TeacherBest } else { $TeacherLegacy }
            Write-Host "Warm-start teacher: $ckpt" -ForegroundColor DarkGray
        } else {
            $doFresh = $true
            Write-Host "No biochem teacher ckpt; auto-enabling -Fresh (AE+ODE then teacher)." -ForegroundColor Yellow
        }
    }

    if ($doFresh) {
        $runName = "K3_fresh_sentinel_wall_leash"
        Remove-Item Env:BIOCHEM_SKIP_PRETRAIN -ErrorAction SilentlyContinue
        Remove-Item Env:BIOCHEM_REUSE_LAST_PRETRAIN -ErrorAction SilentlyContinue
        $env:BIOCHEM_SKIP_PRETRAIN = "0"
        $env:BIOCHEM_REUSE_LAST_PRETRAIN = "0"
        # Shorter pretrain for 4GB; still writes biochem_post_pretrain.pth when done.
        $env:BIOCHEM_AE_EPOCHS = "14"
        $env:BIOCHEM_ODE_RXN_EPOCHS = "12"
        $pyArgs += "--run-name", $runName
        Write-Host "Fresh: Phase 3a AE ($($env:BIOCHEM_AE_EPOCHS)ep) + ODE-RXN ($($env:BIOCHEM_ODE_RXN_EPOCHS)ep), then teacher." -ForegroundColor Cyan
    } else {
        $pyArgs += "--run-name", $runName, "--skip-pretrain"
    }
}

$env:BIOCHEM_RUN_NOTE = if ($Resume) { "K3b_sentinel_wall_leash_resume" } else { $runName }
Write-Host "K3: sentinel + data leash | grad_cap=50000 | gate_floor=0.06 | ep=$Epochs" -ForegroundColor Cyan

python @pyArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
