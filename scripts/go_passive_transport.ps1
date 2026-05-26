# Step 2a passive biochem: 1-way transport (Data_Bio TBPTT; ADR metrics only by default).
# See src/docs/BIOCHEM_TRAINING_PROGRESS.md — enable BIOCHEM_PASSIVE_ADR_BACKPROP=1 only after L_Data_Bio falls.
# Prereq: outputs/kinematics/kinematics_best.pth (Phase-1 GINO-DEQ).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_passive_transport.ps1" -Fresh

param(
    [switch] $Fresh,
    [switch] $OomSafe = $true
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$KinCkpt = Join-Path $RepoRoot "outputs\kinematics\kinematics_best.pth"
if (-not (Test-Path $KinCkpt)) {
    Write-Host "[ERR] Missing Stage-A ckpt: $KinCkpt" -ForegroundColor Red
    exit 1
}

$env:KINEMATICS_USE_HARD_BCS = "1"
$env:KINEMATICS_USE_WIDTH_PRIORS = "1"
$env:BIOCHEM_PRESET = "passive_transport"
$env:BIOCHEM_STOCK_DEFAULTS = "0"
$env:BIOCHEM_NO_TEACHER_DEFAULTS = "1"
$env:BIOCHEM_RUN_NOTE = "passive_transport_step2"
$env:BIOCHEM_VAL_TIME_STRIDE = "10"
$env:BIOCHEM_TEACHER_VAL_EVERY = "3"
$env:BIOCHEM_TEACHER_EPOCHS = "12"
$env:BIOCHEM_EPOCHS = "12"
$env:BIOCHEM_CLI_TEACHER_EPOCHS = "12"
# Full TBPTT backward through species (set 1 if 4GB OOM).
$env:BIOCHEM_DETACH_MACRO_STATE = "0"

Remove-Item Env:BIOCHEM_SUPERVISED_DATA_LEASH,Env:BIOCHEM_MU_TRAIN_WALL_ONLY,Env:BIOCHEM_MU_TRAIN_CLOT_ONLY -ErrorAction SilentlyContinue

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
    "--run-name", "passive_transport_step2"
)
if ($Fresh) {
    Write-Host "[NEW] Fresh passive-transport run (AE+ODE+teacher)." -ForegroundColor Cyan
}
Write-Host "[i] PASSIVE: Data_Bio TBPTT + GT [u,v,p]; ADR metrics only; LR=5e-4; scale-grad-on-cap." -ForegroundColor Cyan

python @pyArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
