# R1: one-step clot forecast prongs (GT flow, no rollout, no GNODE).
#
#   powershell ... -Prong A -Fresh    # MLP, no mu_t input
#   powershell ... -Prong B -Fresh    # MLP + log(mu_t) input
#   powershell ... -Prong C -Fresh    # 1-hop MPNN + log(mu_t)
#   powershell ... -Prong D -Fresh    # B backbone + deploy_pred loss band (model phi @ t_in)

param(
    [ValidateSet("A", "B", "C", "D")]
    [string] $Prong = "A",
    [switch] $Fresh,
    [switch] $NoInitFromB,
    [int] $Epochs = 40
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_clot_phi_shared_env.ps1")
$env:CLOT_PHI_DGAMMA_FEATURE_TIME = "current"

# Forecast ladder R1: one-step pairs, GT flow only
$env:CLOT_FORECAST_MODE = "one_step"
$env:CLOT_FORECAST_PAIR_STRIDE = "1"
$env:CLOT_FORECAST_MASK = "target"
$env:CLOT_PHI_ROLLOUT = "0"
$env:CLOT_PHI_VEL_SOURCE = "gt"
Remove-Item Env:CLOT_PHI_CARRY_PHI -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_PHI_CARRY_LOG_MU -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_PHI_INIT_CHECKPOINT -ErrorAction SilentlyContinue

$env:CLOT_PHI_MODEL = "mlp"
$env:CLOT_FORECAST_INPUT_MU = "0"
$Leg = "r1_prong_a"
if ($Prong -eq "B") {
    $env:CLOT_FORECAST_INPUT_MU = "1"
    $Leg = "r1_prong_b"
} elseif ($Prong -eq "C") {
    $env:CLOT_PHI_MODEL = "mpnn"
    $env:CLOT_FORECAST_INPUT_MU = "1"
    $Leg = "r1_prong_c"
} elseif ($Prong -eq "D") {
    $env:CLOT_FORECAST_INPUT_MU = "1"
    $env:CLOT_FORECAST_MASK = "deploy_pred"
    $Leg = "r1_prong_d"
    # Deploy neighbor commit: pred phi @ t_in, mu(t_in) seeds; keep growth frontier (no phi gate on band).
    $env:BIOCHEM_MLP_NEIGHBOR_SEED = "pred_clot"
    $env:BIOCHEM_MLP_NEIGHBOR_REQUIRE_PHI = "0"
    $env:BIOCHEM_MLP_MU_MAP_PHI_THRESH = "0.5"
    $env:BIOCHEM_MLP_DEPLOY_DGAMMA_SLICE = "0"
    $InitB = Join-Path $RepoRoot "outputs\biochem\clot_forecast_ladder\r1_prong_b\clot_phi_best.pth"
    if (-not $NoInitFromB -and (Test-Path $InitB)) {
        $env:CLOT_PHI_INIT_CHECKPOINT = $InitB
    }
}

$env:CLOT_PHI_HIDDEN = "32"
$env:CLOT_PHI_MLP_DEPTH = "2"
$env:CLOT_PHI_DROPOUT = "0.15"
$env:CLOT_PHI_EPOCHS = "$Epochs"
$env:CLOT_PHI_LR = "1e-3"
$env:CLOT_PHI_WEIGHT_DECAY = "1e-4"
$env:CLOT_PHI_MU_LOG_LAMBDA = "1.5"
$env:CLOT_PHI_DICE_LAMBDA = "0.2"
$env:CLOT_PHI_MINIMAL_FEATURES = "1"
$env:CLOT_PHI_HYBRID = "1"
$env:CLOT_PHI_BALANCED = "1"
$env:CLOT_PHI_SPECIES_FEATURES = "0"
$env:CLOT_PHI_JOINT_BIO = "0"
$env:CLOT_PHI_PHYSICS_BLEND = "0"
$env:CLOT_PHI_TIME_STRIDE = "1"
$env:CLOT_PHI_TIME_STRIDE_AUTO = "0"
$env:CLOT_PHI_SWEEP_DIR = "outputs/biochem/clot_forecast_ladder"
$env:CLOT_PHI_SWEEP_LEG = $Leg

$LegDir = Join-Path $RepoRoot "outputs/biochem/clot_forecast_ladder/$Leg"
$Ckpt = Join-Path $LegDir "clot_phi_best.pth"

if ($Fresh) {
    Remove-Item -Force -ErrorAction SilentlyContinue $Ckpt, (Join-Path $LegDir "clot_phi_train_log.jsonl")
}

Write-Host "[NEW] R1 prong $Prong one-step forecast (GT flow) epochs=$Epochs" -ForegroundColor Cyan
Write-Host "[i]  model=$env:CLOT_PHI_MODEL input_mu=$env:CLOT_FORECAST_INPUT_MU mask=$env:CLOT_FORECAST_MASK" -ForegroundColor DarkGray

python -m src.training.train_clot_phi_simple
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$EvalOut = Join-Path $LegDir "multi_anchor.jsonl"
python scripts/eval_clot_phi_multi_anchor.py --checkpoint $Ckpt --out $EvalOut
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python scripts/summarize_clot_forecast_r1.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[OK]  R1 prong $Prong done ckpt=$Ckpt" -ForegroundColor Green
