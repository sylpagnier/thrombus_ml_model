# R2-simple: phi-only rollout + fixed clot mu (no mu carry, no hybrid delta head).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_forecast_r2_simple.ps1" -Fresh
#
# Predict phi only; mu_eff = log_blend(mu_c, phi, mu_solid). Rollout carries phi_prev only.

param(
    [switch] $Fresh,
    [int] $Epochs = 60
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_clot_phi_shared_env.ps1")
$env:CLOT_PHI_DGAMMA_FEATURE_TIME = "current"

# Phi-only fixed-mu rollout (deploy-faithful band)
Remove-Item Env:CLOT_FORECAST_MODE -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_FORECAST_INPUT_MU -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_PHI_INIT_CHECKPOINT -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_PHI_CARRY_GT_WARMUP_EPOCHS -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_PHI_CARRY_GT_FADE_EPOCHS -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_PHI_CARRY_GT_WARMUP_STEPS -ErrorAction SilentlyContinue

$env:CLOT_FORECAST_MASK = "deploy_pred"
$env:CLOT_PHI_FIXED_MU_FROM_PHI = "1"
$env:CLOT_PHI_HYBRID = "0"
$env:CLOT_PHI_ROLLOUT = "1"
$env:CLOT_PHI_ROLLOUT_DETACH = "1"
$env:CLOT_PHI_VEL_SOURCE = "gt"
$env:CLOT_PHI_CARRY_PHI = "1"
$env:CLOT_PHI_CARRY_LOG_MU = "0"
$env:CLOT_PHI_MU_SOLID_SI = "0.10"
$env:CLOT_PHI_MU_LOG_LAMBDA = "0"

# Deploy neighbor commit band (pred phi + mu from carried phi)
$env:BIOCHEM_MLP_NEIGHBOR_SEED = "pred_clot"
$env:BIOCHEM_MLP_NEIGHBOR_REQUIRE_PHI = "0"
$env:BIOCHEM_MLP_MU_MAP_PHI_THRESH = "0.5"
$env:BIOCHEM_MLP_DEPLOY_DGAMMA_SLICE = "0"

# MLP stack (cold start; in_dim=4 = base + phi_carry, not R1D hybrid)
$env:CLOT_PHI_MODEL = "mlp"
$env:CLOT_PHI_HIDDEN = "32"
$env:CLOT_PHI_MLP_DEPTH = "2"
$env:CLOT_PHI_DROPOUT = "0.15"
$env:CLOT_PHI_EPOCHS = "$Epochs"
$env:CLOT_PHI_LR = "1e-3"
$env:CLOT_PHI_WEIGHT_DECAY = "1e-4"
$env:CLOT_PHI_DICE_LAMBDA = "0.2"
$env:CLOT_PHI_MINIMAL_FEATURES = "1"
$env:CLOT_PHI_SPECIES_FEATURES = "0"
$env:CLOT_PHI_JOINT_BIO = "0"
$env:CLOT_PHI_PHYSICS_BLEND = "0"
$env:CLOT_PHI_BALANCED = "1"
$env:CLOT_PHI_TIME_STRIDE = "1"
$env:CLOT_PHI_TIME_STRIDE_AUTO = "0"
$env:CLOT_PHI_SWEEP_DIR = "outputs/biochem/clot_forecast_ladder"
$env:CLOT_PHI_SWEEP_LEG = "r2_simple_phi_rollout"

$LegDir = Join-Path $RepoRoot "outputs/biochem/clot_forecast_ladder/r2_simple_phi_rollout"
$Ckpt = Join-Path $LegDir "clot_phi_best.pth"

if ($Fresh) {
    Remove-Item -Force -ErrorAction SilentlyContinue $Ckpt, (Join-Path $LegDir "clot_phi_train_log.jsonl")
}

Write-Host "[NEW] R2-simple phi rollout (fixed mu_solid=0.10, phi carry only) epochs=$Epochs" -ForegroundColor Cyan
Write-Host "[i]  mask=$env:CLOT_FORECAST_MASK hybrid=0 carry_phi=1 carry_log_mu=0 in_dim=4 (cold start)" -ForegroundColor DarkGray

python -m src.training.train_clot_phi_simple
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$EvalOut = Join-Path $LegDir "multi_anchor.jsonl"
python scripts/eval_clot_phi_multi_anchor.py --checkpoint $Ckpt --out $EvalOut
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$VizOut = Join-Path $RepoRoot "outputs/biochem/clot_forecast_ladder/viz/r2_simple_phi_rollout_patient007_tfinal.png"
New-Item -ItemType Directory -Force -Path (Split-Path $VizOut) | Out-Null
python -m src.evaluation.viz_clot_phi_simple `
    --anchor patient007 `
    --checkpoint $Ckpt `
    --time-index -1 `
    --plot-mode scatter `
    --out $VizOut

Write-Host "[OK]  R2-simple done ckpt=$Ckpt" -ForegroundColor Green
