# Production Stage-A kinematics: allfix architecture + April-scale curriculum (100 ep, LBFGS).
# Goal: best val Rel L2 on full graph cohort (no cap by default).
#
# Prereq: graphs under data/processed/graphs_kinematics/{newtonian,carreau}
# Backfill if needed: python -m src.data_gen.backfill_kinematics_geometry_level
#
# After completion (optional polish):
#   powershell -File .\scripts\go_kinematics_production_allfix_finetune.ps1
#
# Outputs: outputs/kinematics/production_allfix/kinematics_best.pth
# Promote: Copy-Item outputs\kinematics\production_allfix\kinematics_best.pth outputs\kinematics\kinematics_best.pth

param(
  [switch]$Fresh,
  [int]$Epochs = 100,
  [int]$AdamEpochs = 85,
  [int]$Stage1End = 40,
  [int]$Stage2End = 60,
  [int]$GraphCap = 0,
  [int]$Seed = 42,
  [switch]$Quiet
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

# Architecture (allfix winner vs 30-ep baseline)
$env:KINEMATICS_PHYS_GAT_PRIORS_MULTIPLY_BEFORE_ADDITIVE = "1"
$env:KINEMATICS_BC_ENVELOPE = "1"
$env:KINEMATICS_BC_LAMBDA = "10.0"
$env:KINEMATICS_WSS_FUSE = "1"
$env:KINEMATICS_FOURIER_LEARNABLE = "1"

# Production: enable L-BFGS handoff (April reference adam_epochs=85)
Remove-Item Env:KINEMATICS_SKIP_LBFGS -ErrorAction SilentlyContinue

# Logging
if ($Quiet) {
  $env:KINEMATICS_QUIET = "1"
  $env:KINEMATICS_VAL_PROGRESS = "0"
  $env:KINEMATICS_TQDM = "0"
} else {
  Remove-Item Env:KINEMATICS_QUIET -ErrorAction SilentlyContinue
  Remove-Item Env:KINEMATICS_VAL_PROGRESS -ErrorAction SilentlyContinue
  Remove-Item Env:KINEMATICS_TQDM -ErrorAction SilentlyContinue
}
$env:KINEMATICS_VAL_EVERY = "1"

$env:KINEMATICS_OUTPUT_DIR = "outputs/kinematics/production_allfix"
if ($GraphCap -gt 0) {
  $env:KINEMATICS_GRAPH_CAP = "$GraphCap"
} else {
  Remove-Item Env:KINEMATICS_GRAPH_CAP -ErrorAction SilentlyContinue
}

$capLabel = if ($GraphCap -gt 0) { "$GraphCap" } else { "all" }
Write-Host ("[kin-prod] epochs={0} adam={1} stage1_end={2} stage2_end={3} graphs={4} seed={5} lbfgs=on allfix=on" -f `
  $Epochs, $AdamEpochs, $Stage1End, $Stage2End, $capLabel, $Seed)

$trainArgs = @(
  "-m", "src.training.train_kinematics_predictor",
  "--no-prompt",
  "--epochs", "$Epochs",
  "--adam-epochs", "$AdamEpochs",
  "--stage1-end-epoch", "$Stage1End",
  "--stage2-end-epoch", "$Stage2End",
  "--l0l1-only-epochs", "0",
  "--hard-mining-start-epoch", "16",
  "--accum-steps", "2",
  "--shuffle-graphs",
  "--graph-load-seed", "$Seed"
)
if ($Fresh) { $trainArgs += "--fresh" }
if ($Quiet) { $trainArgs += "--quiet" }

& python @trainArgs

Write-Host "[kin-prod] done. best -> outputs/kinematics/production_allfix/kinematics_best.pth"
