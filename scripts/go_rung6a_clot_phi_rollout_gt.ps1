# Rung 6a: serial clot-phi MLP with carry state; GT [u,v] each step (no kine error).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_rung6a_clot_phi_rollout_gt.ps1" -Fresh

param(
    [switch] $Fresh,
    [int] $Epochs = 60
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_clot_phi_shared_env.ps1")
$env:CLOT_PHI_DGAMMA_FEATURE_TIME = "current"

# Rung 6a rollout (GT velocity teacher forcing on flow)
$env:CLOT_PHI_ROLLOUT = "1"
$env:CLOT_PHI_ROLLOUT_DETACH = "1"
$env:CLOT_PHI_VEL_SOURCE = "gt"
$env:CLOT_PHI_CARRY_PHI = "1"
$env:CLOT_PHI_CARRY_LOG_MU = "1"

# Same stack as rung 4 (GT species + joint blend)
$env:CLOT_PHI_MODEL = "mlp"
$env:CLOT_PHI_HIDDEN = "32"
$env:CLOT_PHI_MLP_DEPTH = "2"
$env:CLOT_PHI_DROPOUT = "0.15"
$env:CLOT_PHI_EPOCHS = "$Epochs"
$env:CLOT_PHI_LR = "1e-3"
$env:CLOT_PHI_WEIGHT_DECAY = "1e-4"
$env:CLOT_PHI_MU_LOG_LAMBDA = "1.5"
$env:CLOT_PHI_DICE_LAMBDA = "0.2"
$env:CLOT_PHI_MINIMAL_FEATURES = "1"
$env:CLOT_PHI_SPECIES_FEATURES = "1"
$env:CLOT_PHI_HYBRID = "1"
$env:CLOT_PHI_BALANCED = "1"
$env:CLOT_PHI_PHYSICS_ORACLE = "0"
$env:CLOT_PHI_JOINT_BIO = "1"
$env:CLOT_PHI_BIO_LAMBDA = "0.25"
$env:CLOT_PHI_JOINT_USE_PRED_SPECIES = "0"
$env:CLOT_PHI_PHYSICS_BLEND = "1"
$env:CLOT_PHI_PHYSICS_BLEND_ALPHA = "0.55"
$env:CLOT_PHI_PHYSICS_MU_RATIO_MAX = "4"
$env:CLOT_PHI_PHYSICS_GELATION_GATE = "1"
$env:CLOT_PHI_SWEEP_DIR = "outputs/biochem/clot_phi_ladder"
$env:CLOT_PHI_SWEEP_LEG = "rollout_gt_rung6a"

$LegDir = Join-Path $RepoRoot "outputs/biochem/clot_phi_ladder/rollout_gt_rung6a"
$Ckpt = Join-Path $LegDir "clot_phi_best.pth"

if ($Fresh) {
    Remove-Item -Force -ErrorAction SilentlyContinue $Ckpt, (Join-Path $LegDir "clot_phi_train_log.jsonl")
}

Write-Host "[NEW] rung6a clot-phi rollout (GT vel + carry) epochs=$Epochs" -ForegroundColor Cyan
python -m src.training.train_clot_phi_simple
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$EvalOut = Join-Path $RepoRoot "outputs/biochem/rung6a_rollout_gt/multi_anchor.jsonl"
New-Item -ItemType Directory -Force -Path (Split-Path $EvalOut) | Out-Null
python scripts/eval_clot_phi_multi_anchor.py --checkpoint $Ckpt --out $EvalOut

python -m src.evaluation.viz_clot_phi_simple `
    --anchor patient007 `
    --checkpoint $Ckpt `
    --time-index -1 `
    --plot-mode scatter `
    --out outputs/biochem/clot_phi_viz_rung6a_p007_tfinal.png

Write-Host "[OK]  rung6a done ckpt=$Ckpt" -ForegroundColor Green
