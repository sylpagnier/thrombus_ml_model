param(
    [switch] $Fresh,
    [int] $Epochs = 60
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_clot_phi_shared_env.ps1")

# Use teacher-rolled species anchors to match inference conditions.
$env:CLOT_PHI_ANCHOR_DIR = "outputs/biochem/anchors_teacher_species"

# Regression-only viscosity-map fit (disable BCE/dice classification loss terms).
$env:CLOT_PHI_REGRESSION_ONLY = "1"
$env:CLOT_PHI_MODEL = "mlp"
$env:CLOT_PHI_HYBRID = "1"
$env:CLOT_PHI_SOFT_LABELS = "0"
$env:CLOT_PHI_BALANCED = "0"
$env:CLOT_PHI_DICE_LAMBDA = "0.0"
$env:CLOT_PHI_MU_LOG_LAMBDA = "2.0"
$env:CLOT_PHI_HIDDEN = "64"
$env:CLOT_PHI_MLP_DEPTH = "2"
$env:CLOT_PHI_DROPOUT = "0.10"
$env:CLOT_PHI_LR = "1e-3"
$env:CLOT_PHI_WEIGHT_DECAY = "1e-4"
$env:CLOT_PHI_EPOCHS = "$Epochs"

# Keep only physically logical bounds by making cap effectively non-binding.
$env:CLOT_PHI_MU_CAP_SI = "10000"
$env:CLOT_PHI_MU_SOLID_SI = "10000"

# Optional joint species auxiliary to stabilize latent dynamics.
$env:CLOT_PHI_JOINT_BIO = "1"
$env:CLOT_PHI_BIO_LAMBDA = "0.10"
$env:CLOT_PHI_SPECIES_HIDDEN = "32"
$env:CLOT_PHI_JOINT_USE_PRED_SPECIES = "1"

# No hand-mixed physics blend for pure learned regression target.
$env:CLOT_PHI_PHYSICS_BLEND = "0"

if ($Fresh) {
    Remove-Item -Force -ErrorAction SilentlyContinue "outputs\biochem\clot_phi_best.pth"
    Remove-Item -Force -ErrorAction SilentlyContinue "outputs\biochem\clot_phi_train_log.jsonl"
}

Write-Host "[NEW] Regression-only clot-phi viscosity map training..." -ForegroundColor Cyan
python -m src.training.train_clot_phi_simple
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -m src.evaluation.viz_clot_phi_simple --anchor patient007 --checkpoint outputs/biochem/clot_phi_best.pth
