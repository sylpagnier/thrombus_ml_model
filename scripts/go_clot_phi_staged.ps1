# Staged clot-phi: (A) regression mu on teacher anchors, (B) freeze mu branch + train phi.
#
#   powershell -File .\scripts\go_clot_phi_staged.ps1 -Fresh

param(
    [switch] $Fresh,
    [int] $EpochsA = 40,
    [int] $EpochsB = 40
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_clot_phi_shared_env.ps1")

$env:CLOT_PHI_ANCHOR_DIR = "outputs/biochem/anchors_teacher_species"
$StageA = "outputs/biochem/clot_phi_stage_a.pth"
$Best = "outputs/biochem/clot_phi_best.pth"

if ($Fresh) {
    Remove-Item -Force -ErrorAction SilentlyContinue $StageA, $Best, "outputs\biochem\clot_phi_train_log.jsonl"
}

Write-Host "[NEW] Stage A: regression-only mu (physical cap)..." -ForegroundColor Cyan
$env:CLOT_PHI_REGRESSION_ONLY = "1"
$env:CLOT_PHI_MODEL = "mlp"
$env:CLOT_PHI_HYBRID = "1"
$env:CLOT_PHI_SOFT_LABELS = "0"
$env:CLOT_PHI_BALANCED = "0"
$env:CLOT_PHI_DICE_LAMBDA = "0"
$env:CLOT_PHI_MU_LOG_LAMBDA = "2.0"
$env:CLOT_PHI_HIDDEN = "32"
$env:CLOT_PHI_MLP_DEPTH = "2"
$env:CLOT_PHI_DROPOUT = "0.15"
$env:CLOT_PHI_LR = "1e-3"
$env:CLOT_PHI_WEIGHT_DECAY = "1e-4"
$env:CLOT_PHI_EPOCHS = "$EpochsA"
$env:CLOT_PHI_MU_CAP_SI = "10"
$env:CLOT_PHI_MU_SOLID_SI = "10"
$env:CLOT_PHI_JOINT_BIO = "0"
$env:CLOT_PHI_PHYSICS_BLEND = "0"
Remove-Item Env:CLOT_PHI_INIT_CHECKPOINT,Env:CLOT_PHI_FREEZE_MU_BRANCH -ErrorAction SilentlyContinue

python -m src.training.train_clot_phi_simple
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Copy-Item -Force $Best $StageA

Write-Host "[NEW] Stage B: freeze mu branch, train phi (hybrid classifier)..." -ForegroundColor Cyan
$env:CLOT_PHI_REGRESSION_ONLY = "0"
$env:CLOT_PHI_SOFT_LABELS = "1"
$env:CLOT_PHI_BALANCED = "1"
$env:CLOT_PHI_DICE_LAMBDA = "0.2"
$env:CLOT_PHI_MU_LOG_LAMBDA = "1.0"
$env:CLOT_PHI_MU_CAP_SI = "0.10"
$env:CLOT_PHI_MU_SOLID_SI = "0.10"
$env:CLOT_PHI_THRESH_SI = "0.045"
$env:CLOT_PHI_JOINT_BIO = "1"
$env:CLOT_PHI_BIO_LAMBDA = "0.25"
$env:CLOT_PHI_JOINT_USE_PRED_SPECIES = "1"
$env:CLOT_PHI_PHYSICS_BLEND = "1"
$env:CLOT_PHI_PHYSICS_BLEND_ALPHA = "0.75"
$env:CLOT_PHI_PHYSICS_MU_RATIO_MAX = "4"
$env:CLOT_PHI_PHYSICS_GELATION_GATE = "1"
$env:CLOT_PHI_INIT_CHECKPOINT = $StageA
$env:CLOT_PHI_FREEZE_MU_BRANCH = "1"
$env:CLOT_PHI_EPOCHS = "$EpochsB"

python -m src.training.train_clot_phi_simple
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python scripts/eval_clot_phi_multi_anchor.py --checkpoint outputs/biochem/clot_phi_best.pth `
    --out outputs/biochem/clot_phi_staged_multi_anchor.jsonl
python -m src.evaluation.viz_clot_phi_simple --anchor patient007 --checkpoint outputs/biochem/clot_phi_best.pth
