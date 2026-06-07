# R2B: R2 rollout + Bridge A (GT log-mu carry warm-up -> fade -> pred carry).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_forecast_r2b.ps1" -Fresh
#   powershell ... -NoInitFromD    # cold start (in_dim must match carry recipe)

param(
    [switch] $Fresh,
    [switch] $NoInitFromD,
    [int] $Epochs = 60,
    [int] $WarmupEpochs = 15,
    [int] $FadeEpochs = 10,
    [int] $WarmupSteps = 0
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_clot_phi_shared_env.ps1")
$env:CLOT_PHI_DGAMMA_FEATURE_TIME = "current"

# Rollout (not one_step forecast): carry log(mu) only -> in_dim=4 matches R1D
Remove-Item Env:CLOT_FORECAST_MODE -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_FORECAST_INPUT_MU -ErrorAction SilentlyContinue
$env:CLOT_FORECAST_MASK = "deploy_pred"
$env:CLOT_PHI_ROLLOUT = "1"
$env:CLOT_PHI_ROLLOUT_DETACH = "1"
$env:CLOT_PHI_VEL_SOURCE = "gt"
$env:CLOT_PHI_CARRY_PHI = "0"
$env:CLOT_PHI_CARRY_LOG_MU = "1"

# Bridge A: train-only GT log(mu @ ti) in carry slot, then linear fade to pred carry
$env:CLOT_PHI_CARRY_GT_WARMUP_EPOCHS = "$WarmupEpochs"
$env:CLOT_PHI_CARRY_GT_FADE_EPOCHS = "$FadeEpochs"
$env:CLOT_PHI_CARRY_GT_WARMUP_STEPS = "$WarmupSteps"

# Deploy neighbor commit band (pred phi + mu seeds)
$env:BIOCHEM_MLP_NEIGHBOR_SEED = "pred_clot"
$env:BIOCHEM_MLP_NEIGHBOR_REQUIRE_PHI = "0"
$env:BIOCHEM_MLP_MU_MAP_PHI_THRESH = "0.5"
$env:BIOCHEM_MLP_DEPLOY_DGAMMA_SLICE = "0"
Remove-Item Env:CLOT_PHI_INIT_CHECKPOINT -ErrorAction SilentlyContinue

$InitD = Join-Path $RepoRoot "outputs\biochem\clot_forecast_ladder\r1_prong_d\clot_phi_best.pth"
if (-not $NoInitFromD -and (Test-Path $InitD)) {
    $env:CLOT_PHI_INIT_CHECKPOINT = $InitD
}

# R1D-matched MLP stack (no species / joint bio / physics blend)
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
$env:CLOT_PHI_SPECIES_FEATURES = "0"
$env:CLOT_PHI_JOINT_BIO = "0"
$env:CLOT_PHI_PHYSICS_BLEND = "0"
$env:CLOT_PHI_HYBRID = "1"
$env:CLOT_PHI_BALANCED = "1"
$env:CLOT_PHI_TIME_STRIDE = "1"
$env:CLOT_PHI_TIME_STRIDE_AUTO = "0"
$env:CLOT_PHI_SWEEP_DIR = "outputs/biochem/clot_forecast_ladder"
$env:CLOT_PHI_SWEEP_LEG = "r2b_bridge_a_gt_carry"

$LegDir = Join-Path $RepoRoot "outputs/biochem/clot_forecast_ladder/r2b_bridge_a_gt_carry"
$Ckpt = Join-Path $LegDir "clot_phi_best.pth"

if ($Fresh) {
    Remove-Item -Force -ErrorAction SilentlyContinue $Ckpt, (Join-Path $LegDir "clot_phi_train_log.jsonl")
}

Write-Host "[NEW] R2B forecast rollout (Bridge A GT carry warm-up) epochs=$Epochs" -ForegroundColor Cyan
Write-Host "[i]  mask=$env:CLOT_FORECAST_MASK carry_gt_epochs=$WarmupEpochs fade_epochs=$FadeEpochs carry_gt_steps=$WarmupSteps init=$($env:CLOT_PHI_INIT_CHECKPOINT)" -ForegroundColor DarkGray

python -m src.training.train_clot_phi_simple
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$EvalOut = Join-Path $LegDir "multi_anchor.jsonl"
python scripts/eval_clot_phi_multi_anchor.py --checkpoint $Ckpt --out $EvalOut
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$VizOut = Join-Path $RepoRoot "outputs/biochem/clot_forecast_ladder/viz/r2b_bridge_a_patient007_tfinal.png"
New-Item -ItemType Directory -Force -Path (Split-Path $VizOut) | Out-Null
python -m src.evaluation.viz_clot_phi_simple `
    --anchor patient007 `
    --checkpoint $Ckpt `
    --time-index -1 `
    --plot-mode scatter `
    --out $VizOut

Write-Host "[OK]  R2B done ckpt=$Ckpt" -ForegroundColor Green
