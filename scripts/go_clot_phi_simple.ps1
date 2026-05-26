# Wall-local clot phi: dgamma mask + minimal hybrid (linear baseline, MLP step-up).
#   powershell -File .\scripts\go_clot_phi_simple.ps1 -Fresh
#   powershell -File .\scripts\go_clot_phi_simple.ps1 -Fresh -Model linear
#   powershell -File .\scripts\go_clot_phi_simple.ps1 -Fresh -Model mlp

param(
    [switch] $VizOnly,
    [switch] $Fresh,
    [ValidateSet("linear", "mlp")]
    [string] $Model = "mlp"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_clot_phi_shared_env.ps1")
$env:CLOT_PHI_MODEL = $Model
$env:CLOT_PHI_SWEEP_DIR = ""
$env:CLOT_PHI_SWEEP_LEG = ""

if ($Model -eq "linear") {
    $env:CLOT_PHI_HIDDEN = "16"
    $env:CLOT_PHI_DROPOUT = "0"
    $env:CLOT_PHI_MU_LOG_LAMBDA = "2.0"
    $env:CLOT_PHI_DICE_LAMBDA = "0.25"
    $env:CLOT_PHI_EPOCHS = "50"
    $env:CLOT_PHI_LR = "5e-3"
    $env:CLOT_PHI_WEIGHT_DECAY = "1e-5"
} else {
    # MLP default (round-2 winner joint_blend_gtsp): h32/d2 + species + joint bio + physics blend.
    $env:CLOT_PHI_HIDDEN = "32"
    $env:CLOT_PHI_MLP_DEPTH = "2"
    $env:CLOT_PHI_DROPOUT = "0.15"
    $env:CLOT_PHI_MU_LOG_LAMBDA = "1.5"
    $env:CLOT_PHI_DICE_LAMBDA = "0.2"
    $env:CLOT_PHI_EPOCHS = "60"
    $env:CLOT_PHI_LR = "1e-3"
    $env:CLOT_PHI_WEIGHT_DECAY = "1e-4"
    $env:CLOT_PHI_SPECIES_FEATURES = "1"
    $env:CLOT_PHI_JOINT_BIO = "1"
    $env:CLOT_PHI_BIO_LAMBDA = "0.25"
    $env:CLOT_PHI_JOINT_USE_PRED_SPECIES = "0"
    $env:CLOT_PHI_PHYSICS_BLEND = "1"
    $env:CLOT_PHI_PHYSICS_BLEND_ALPHA = "0.55"
    $env:CLOT_PHI_PHYSICS_MU_RATIO_MAX = "4"
    $env:CLOT_PHI_PHYSICS_GELATION_GATE = "1"
    $env:CLOT_PHI_SPECIES_HIDDEN = "32"
}

if ($Fresh -and -not $VizOnly) {
    Remove-Item -Force -ErrorAction SilentlyContinue "outputs\biochem\clot_phi_best.pth"
    Remove-Item -Force -ErrorAction SilentlyContinue "outputs\biochem\clot_phi_train_log.jsonl"
}

Write-Host "[i]  clot_phi arch=$Model hidden=$env:CLOT_PHI_HIDDEN dropout=$env:CLOT_PHI_DROPOUT lr=$env:CLOT_PHI_LR epochs=$env:CLOT_PHI_EPOCHS" -ForegroundColor Cyan

$BestCkpt = Join-Path $RepoRoot "outputs\biochem\clot_phi_best.pth"
$ArchCkpt = Join-Path $RepoRoot "outputs\biochem\clot_phi_best_$Model.pth"

if (-not $VizOnly) {
    python -m src.training.train_clot_phi_simple
    if (Test-Path $BestCkpt) {
        Copy-Item -Force $BestCkpt $ArchCkpt
        Write-Host "[OK]  arch checkpoint -> outputs\biochem\clot_phi_best_$Model.pth" -ForegroundColor Green
    }
}

python -m src.evaluation.viz_clot_phi_simple --anchor patient007 --checkpoint outputs/biochem/clot_phi_best.pth
