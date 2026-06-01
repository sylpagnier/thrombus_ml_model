# L2-heavy Carreau finetune after go_kinematics_production_allfix.ps1 (lower LR, more L2 sampling).
# Resumes production_allfix checkpoint; writes to same KINEMATICS_OUTPUT_DIR.

param(
  [string]$Resume = "latest",
  [double]$FinetuneLr = 1e-5,
  [int]$AdamEpochs = 40,
  [int]$HardMiningStart = 20
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$env:KINEMATICS_PHYS_GAT_PRIORS_MULTIPLY_BEFORE_ADDITIVE = "1"
$env:KINEMATICS_BC_ENVELOPE = "1"
$env:KINEMATICS_BC_LAMBDA = "10.0"
$env:KINEMATICS_WSS_FUSE = "1"
$env:KINEMATICS_FOURIER_LEARNABLE = "1"
Remove-Item Env:KINEMATICS_SKIP_LBFGS -ErrorAction SilentlyContinue
$env:KINEMATICS_OUTPUT_DIR = "outputs/kinematics/production_allfix"
$env:KINEMATICS_VAL_EVERY = "1"
Remove-Item Env:KINEMATICS_GRAPH_CAP -ErrorAction SilentlyContinue

Write-Host ("[kin-prod-ft] resume={0} lr={1} adam_epochs={2} geometry=l2_heavy" -f $Resume, $FinetuneLr, $AdamEpochs)

python -m src.training.train_kinematics_predictor `
  --no-prompt `
  --resume $Resume `
  --geometry-phase l2_heavy `
  --hard-mining-start-epoch $HardMiningStart `
  --finetune-lr $FinetuneLr `
  --adam-epochs $AdamEpochs

Write-Host "[kin-prod-ft] done."
