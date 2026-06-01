# Rung 6b: serial clot-phi + Stage-A GINO-DEQ per step (mu_prior -> new [u,v]).
# Slower than 6a: one steady DEQ solve per time index per graph.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_rung6b_clot_phi_rollout_kine.ps1" -Fresh
# Optional stability while kine is imperfect:
#   $env:CLOT_PHI_KINE_TF = "0.3"   # blend 30% GT velocity into features

param(
    [switch] $Fresh,
    [int] $Epochs = 60,
    [double] $KineTf = 0.0
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_clot_phi_shared_env.ps1")
$env:CLOT_PHI_DGAMMA_FEATURE_TIME = "current"

$env:CLOT_PHI_ROLLOUT = "1"
$env:CLOT_PHI_ROLLOUT_DETACH = "1"
$env:CLOT_PHI_VEL_SOURCE = "kinematics"
$env:CLOT_PHI_CARRY_PHI = "1"
$env:CLOT_PHI_CARRY_LOG_MU = "1"
$env:CLOT_PHI_KINE_TF = "$KineTf"
$env:CLOT_PHI_KINE_CKPT = "outputs/kinematics/kinematics_best.pth"

# Rung 5-style: predicted species from dumped anchors (optional; comment out for GT graphs)
# $env:CLOT_PHI_ANCHOR_DIR = "outputs/biochem/passive_species_clotband_focus/anchors_clotband_36"

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
$env:CLOT_PHI_JOINT_BIO = "1"
$env:CLOT_PHI_BIO_LAMBDA = "0.25"
$env:CLOT_PHI_JOINT_USE_PRED_SPECIES = "0"
$env:CLOT_PHI_PHYSICS_BLEND = "1"
$env:CLOT_PHI_PHYSICS_BLEND_ALPHA = "0.55"
$env:CLOT_PHI_PHYSICS_MU_RATIO_MAX = "4"
$env:CLOT_PHI_PHYSICS_GELATION_GATE = "1"
$env:CLOT_PHI_SWEEP_DIR = "outputs/biochem/clot_phi_ladder"
$env:CLOT_PHI_SWEEP_LEG = "rollout_kine_rung6b"

$LegDir = Join-Path $RepoRoot "outputs/biochem/clot_phi_ladder/rollout_kine_rung6b"
$Ckpt = Join-Path $LegDir "clot_phi_best.pth"

if ($Fresh) {
    Remove-Item -Force -ErrorAction SilentlyContinue $Ckpt, (Join-Path $LegDir "clot_phi_train_log.jsonl")
}

Write-Host "[NEW] rung6b rollout vel=kinematics kine_tf=$KineTf epochs=$Epochs" -ForegroundColor Cyan
python -m src.training.train_clot_phi_simple
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$EvalOut = Join-Path $RepoRoot "outputs/biochem/rung6b_rollout_kine/multi_anchor.jsonl"
New-Item -ItemType Directory -Force -Path (Split-Path $EvalOut) | Out-Null
python scripts/eval_clot_phi_multi_anchor.py --checkpoint $Ckpt --out $EvalOut

Write-Host "[OK]  rung6b done ckpt=$Ckpt" -ForegroundColor Green
