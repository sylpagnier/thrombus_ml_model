# Cache teacher-rolled species onto anchors, then train clot-phi on those graphs.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_clot_phi_with_teacher_species.ps1 -Fresh
#

param(
    [switch] $Fresh,
    [switch] $FreshCache,
    [int] $Epochs = 60,
    [int] $DumpMinSteps = 6
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_clot_phi_shared_env.ps1")

$Teacher = "outputs/biochem/biochem_teacher_last.pth"
$OutAnchors = "outputs/biochem/anchors_teacher_species"

if (-not (Test-Path $Teacher)) {
    Write-Host "[ERR] Missing teacher ckpt: $Teacher" -ForegroundColor Red
    exit 1
}

if ($FreshCache) {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $OutAnchors
}
if ($Fresh) {
    Remove-Item -Force -ErrorAction SilentlyContinue "outputs\biochem\clot_phi_best.pth"
    Remove-Item -Force -ErrorAction SilentlyContinue "outputs\biochem\clot_phi_train_log.jsonl"
}

Write-Host "[NEW] Dumping teacher species onto anchors..." -ForegroundColor Cyan
$dumpArgs = @(
    (Join-Path $PSScriptRoot "dump_teacher_species_to_anchors.py"),
    "--teacher", $Teacher,
    "--out-dir", $OutAnchors,
    "--device", "cuda",
    "--time-stride", "24",
    "--min-steps", "$DumpMinSteps"
)
if ($FreshCache) { $dumpArgs += "--force" }
python @dumpArgs

$env:CLOT_PHI_ANCHOR_DIR = $OutAnchors

# Use the deployment-like clot-phi default (no GT species features, predicted species, physics blend).
$env:CLOT_PHI_MODEL = "mlp"
$env:CLOT_PHI_HIDDEN = "32"
$env:CLOT_PHI_MLP_DEPTH = "2"
$env:CLOT_PHI_DROPOUT = "0.15"
$env:CLOT_PHI_MU_LOG_LAMBDA = "1.5"
$env:CLOT_PHI_DICE_LAMBDA = "0.2"
$env:CLOT_PHI_EPOCHS = "$Epochs"
$env:CLOT_PHI_LR = "1e-3"
$env:CLOT_PHI_WEIGHT_DECAY = "1e-4"
$env:CLOT_PHI_HYBRID = "1"
$env:CLOT_PHI_SOFT_LABELS = "1"
$env:CLOT_PHI_BALANCED = "1"
$env:CLOT_PHI_POS_WEIGHT_CAP = "8"
$env:CLOT_PHI_MINIMAL_FEATURES = "1"
$env:CLOT_PHI_SPECIES_FEATURES = "0"
$env:CLOT_PHI_JOINT_BIO = "1"
$env:CLOT_PHI_BIO_LAMBDA = "0.25"
$env:CLOT_PHI_ANCHOR_BALANCED = "1"
$env:CLOT_PHI_BIO_FI_WEIGHT = "2.0"
$env:CLOT_PHI_BIO_MAT_WEIGHT = "2.0"
$env:CLOT_PHI_JOINT_USE_PRED_SPECIES = "1"
$env:CLOT_PHI_PHYSICS_BLEND = "1"
$env:CLOT_PHI_PHYSICS_BLEND_ALPHA = "0.75"
$env:CLOT_PHI_PHYSICS_MU_RATIO_MAX = "4"
$env:CLOT_PHI_PHYSICS_GELATION_GATE = "1"
$env:CLOT_PHI_SPECIES_HIDDEN = "32"
$env:CLOT_PHI_THRESH_SI = "0.045"
$env:CLOT_PHI_TIME_STRIDE_AUTO = "1"

Write-Host "[NEW] Training clot-phi on teacher-species anchors..." -ForegroundColor Cyan
python -m src.training.train_clot_phi_simple
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -m src.evaluation.viz_clot_phi_simple --anchor patient007 --checkpoint outputs/biochem/clot_phi_best.pth

