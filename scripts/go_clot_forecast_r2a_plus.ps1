# R2a-plus: one-step phi-only + log(GT mu @ t_in) + fixed clot mu (no rollout).
#
# Adds current-state mu feature (R1B/D style) without hybrid regression.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_forecast_r2a_plus.ps1" -Fresh

param(
    [switch] $Fresh,
    [int] $Epochs = 40
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_clot_phi_shared_env.ps1")
. (Join-Path $PSScriptRoot "_clot_forecast_r2a_plus_base.ps1")
$env:CLOT_PHI_DGAMMA_FEATURE_TIME = "current"
$env:CLOT_FORECAST_MASK = "deploy_pred"
$env:CLOT_PHI_SWEEP_DIR = "outputs/biochem/clot_forecast_ladder"
$env:CLOT_PHI_SWEEP_LEG = "r2a_plus_one_step_phi"
$env:CLOT_PHI_EPOCHS = "$Epochs"
Remove-Item Env:CLOT_PHI_INIT_CHECKPOINT -ErrorAction SilentlyContinue

$LegDir = Join-Path $RepoRoot "outputs/biochem/clot_forecast_ladder/r2a_plus_one_step_phi"
$Ckpt = Join-Path $LegDir "clot_phi_best.pth"

if ($Fresh) {
    Remove-Item -Force -ErrorAction SilentlyContinue $Ckpt, (Join-Path $LegDir "clot_phi_train_log.jsonl")
}

Write-Host "[NEW] R2a-plus one-step phi + log(mu@t_in), fixed mu_solid=0.10 epochs=$Epochs" -ForegroundColor Cyan
Write-Host "[i]  forecast=one_step mask=$env:CLOT_FORECAST_MASK hybrid=0 in_dim=4" -ForegroundColor DarkGray

python -m src.training.train_clot_phi_simple
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$EvalOut = Join-Path $LegDir "multi_anchor.jsonl"
python scripts/eval_clot_phi_multi_anchor.py --checkpoint $Ckpt --out $EvalOut
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$VizOut = Join-Path $RepoRoot "outputs/biochem/clot_forecast_ladder/viz/r2a_plus_one_step_phi_patient007_tfinal.png"
New-Item -ItemType Directory -Force -Path (Split-Path $VizOut) | Out-Null
python -m src.evaluation.viz_clot_phi_simple `
    --anchor patient007 `
    --checkpoint $Ckpt `
    --time-index -1 `
    --plot-mode scatter `
    --out $VizOut

Write-Host "[OK]  R2a-plus done ckpt=$Ckpt" -ForegroundColor Green
