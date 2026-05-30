param(
    [string] $AnchorDir,
    [string] $LegName = "tmp_leg",
    [int] $Epochs = 20,
    [double] $BioFiWeight = 2.0,
    [double] $BioMatWeight = 2.0
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (-not $AnchorDir) {
    Write-Host "[ERR] AnchorDir is required." -ForegroundColor Red
    exit 1
}

. (Join-Path $PSScriptRoot "_clot_phi_shared_env.ps1")

$env:CLOT_PHI_ANCHOR_DIR = $AnchorDir
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
$env:CLOT_PHI_BIO_FI_WEIGHT = "$BioFiWeight"
$env:CLOT_PHI_BIO_MAT_WEIGHT = "$BioMatWeight"
$env:CLOT_PHI_JOINT_USE_PRED_SPECIES = "1"
$env:CLOT_PHI_PHYSICS_BLEND = "1"
$env:CLOT_PHI_PHYSICS_BLEND_ALPHA = "0.75"
$env:CLOT_PHI_PHYSICS_MU_RATIO_MAX = "4"
$env:CLOT_PHI_PHYSICS_GELATION_GATE = "1"
$env:CLOT_PHI_SPECIES_HIDDEN = "32"
$env:CLOT_PHI_THRESH_SI = "0.045"
$env:CLOT_PHI_TIME_STRIDE_AUTO = "1"
$env:CLOT_PHI_SWEEP_DIR = "outputs/biochem/passive_species_focus_compare"
$env:CLOT_PHI_SWEEP_LEG = $LegName

Write-Host "[NEW] Training clot-phi leg=$LegName from $AnchorDir (epochs=$Epochs)" -ForegroundColor Cyan
python -m src.training.train_clot_phi_simple
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$ckpt = "outputs/biochem/passive_species_focus_compare/$LegName/clot_phi_best.pth"
$out = "outputs/biochem/passive_species_focus_compare/$LegName/multi_anchor.jsonl"
python scripts/eval_clot_phi_multi_anchor.py --checkpoint $ckpt --out $out
exit $LASTEXITCODE
