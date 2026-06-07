# R2a: one-step phi-only + fixed clot mu (no rollout, no mu carry).
#
# Gate before R2b rollout: p007 band F1 + one-step clot_shape on deploy_pred.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_forecast_r2a_one_step.ps1" -Fresh

param(
    [switch] $Fresh,
    [int] $Epochs = 40
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_clot_phi_shared_env.ps1")
$env:CLOT_PHI_DGAMMA_FEATURE_TIME = "current"

# One-step forecast (no rollout)
$env:CLOT_FORECAST_MODE = "one_step"
$env:CLOT_FORECAST_PAIR_STRIDE = "1"
$env:CLOT_FORECAST_MASK = "deploy_pred"
$env:CLOT_FORECAST_INPUT_MU = "0"
Remove-Item Env:CLOT_PHI_INIT_CHECKPOINT -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_PHI_CARRY_GT_WARMUP_EPOCHS -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_PHI_CARRY_GT_FADE_EPOCHS -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_PHI_CARRY_GT_WARMUP_STEPS -ErrorAction SilentlyContinue

$env:CLOT_PHI_FIXED_MU_FROM_PHI = "1"
$env:CLOT_PHI_HYBRID = "0"
$env:CLOT_PHI_ROLLOUT = "0"
Remove-Item Env:CLOT_PHI_CARRY_PHI -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_PHI_CARRY_LOG_MU -ErrorAction SilentlyContinue
$env:CLOT_PHI_MU_SOLID_SI = "0.10"
$env:CLOT_PHI_MU_LOG_LAMBDA = "0"

# Deploy neighbor commit band (model phi @ t_in)
$env:BIOCHEM_MLP_NEIGHBOR_SEED = "pred_clot"
$env:BIOCHEM_MLP_NEIGHBOR_REQUIRE_PHI = "0"
$env:BIOCHEM_MLP_MU_MAP_PHI_THRESH = "0.5"
$env:BIOCHEM_MLP_DEPLOY_DGAMMA_SLICE = "0"

# MLP phi-only (in_dim=3 minimal features)
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
$env:CLOT_PHI_SWEEP_LEG = "r2a_one_step_phi"

$LegDir = Join-Path $RepoRoot "outputs/biochem/clot_forecast_ladder/r2a_one_step_phi"
$Ckpt = Join-Path $LegDir "clot_phi_best.pth"

if ($Fresh) {
    Remove-Item -Force -ErrorAction SilentlyContinue $Ckpt, (Join-Path $LegDir "clot_phi_train_log.jsonl")
}

Write-Host "[NEW] R2a one-step phi-only (fixed mu_solid=0.10, no rollout) epochs=$Epochs" -ForegroundColor Cyan
Write-Host "[i]  forecast=one_step mask=$env:CLOT_FORECAST_MASK hybrid=0 in_dim=3" -ForegroundColor DarkGray

python -m src.training.train_clot_phi_simple
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$EvalOut = Join-Path $LegDir "multi_anchor.jsonl"
python scripts/eval_clot_phi_multi_anchor.py --checkpoint $Ckpt --out $EvalOut
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$VizOut = Join-Path $RepoRoot "outputs/biochem/clot_forecast_ladder/viz/r2a_one_step_phi_patient007_tfinal.png"
New-Item -ItemType Directory -Force -Path (Split-Path $VizOut) | Out-Null
python -m src.evaluation.viz_clot_phi_simple `
    --anchor patient007 `
    --checkpoint $Ckpt `
    --time-index -1 `
    --plot-mode scatter `
    --out $VizOut

Write-Host "[OK]  R2a done ckpt=$Ckpt" -ForegroundColor Green
