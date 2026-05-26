# A0_april_ratio + shuffled graphs + ALL proposed fixes (candidate architecture A/B).
# Compare printed validation Rel L2 vs `go_kinematics_a0_shuffle.ps1`.
#
# Fixes implemented behind env toggles:
# - KINEMATICS_PHYS_GAT_PRIORS_MULTIPLY_BEFORE_ADDITIVE=1
# - KINEMATICS_BC_ENVELOPE=1 (soft hard-BC envelope)
# - KINEMATICS_WSS_FUSE=1 (fuse WSS decoder with z + u,v,p and mu)
# - KINEMATICS_FOURIER_LEARNABLE=1 (learnable Fourier frequency bands)
#
# Uses Adam-only (KINEMATICS_SKIP_LBFGS=1) for stability on short sweeps.

param(
  [int]$Epochs = 30,
  [int]$GraphCap = 2000,
  [int]$Seed = 42,
  [switch]$NoGraphCap
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

# Candidate toggles
$env:KINEMATICS_PHYS_GAT_PRIORS_MULTIPLY_BEFORE_ADDITIVE = "1"
$env:KINEMATICS_BC_ENVELOPE = "1"
$env:KINEMATICS_BC_LAMBDA = "10.0"
$env:KINEMATICS_WSS_FUSE = "1"
$env:KINEMATICS_FOURIER_LEARNABLE = "1"

# Stable / quiet
$env:KINEMATICS_QUIET = "1"
$env:KINEMATICS_VAL_PROGRESS = "0"
$env:KINEMATICS_TQDM = "0"
$env:KINEMATICS_SKIP_LBFGS = "1"

# Isolate outputs
$env:KINEMATICS_OUTPUT_DIR = "outputs/kinematics/a0_shuffle_allfix"

if ($NoGraphCap) {
  Remove-Item Env:KINEMATICS_GRAPH_CAP -ErrorAction SilentlyContinue
} else {
  $env:KINEMATICS_GRAPH_CAP = "$GraphCap"
}

$stage1End = [int][math]::Round(40 * $Epochs / 100.0)
$stage2End = [int][math]::Round(60 * $Epochs / 100.0)
if ($stage1End -lt 1) { $stage1End = 1 }
if ($stage2End -le $stage1End) { $stage2End = $stage1End + 1 }
if ($stage2End -ge $Epochs) { $stage2End = $Epochs - 1 }
$adamEpochs = $Epochs - 1

Write-Host ("[a0+shuffle+allfix] epochs={0} stage1_end={1} stage2_end={2} graphcap={3} seed={4}" -f $Epochs, $stage1End, $stage2End, $env:KINEMATICS_GRAPH_CAP, $Seed)

python -m src.training.train_kinematics_predictor --fresh --no-prompt --quiet `
  --epochs $Epochs --adam-epochs $adamEpochs `
  --stage1-end-epoch $stage1End --stage2-end-epoch $stage2End `
  --l0l1-only-epochs 0 `
  --hard-mining-start-epoch 16 `
  --shuffle-graphs --graph-load-seed $Seed

Write-Host "[a0+shuffle+allfix] done."

