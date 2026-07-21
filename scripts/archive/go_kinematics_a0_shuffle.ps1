# A0_april_ratio + shuffle recipe (stable, quiet logs).
#
# Runs Adam-only (no L-BFGS) because sweep legs on short schedules sometimes
# produce non-finite validation (NaN) and fail to promote checkpoints.
#
# This script uses the same defaults as the winning A0 leg:
# - geometry curriculum: enabled
# - l0l1-only-epochs: 0
# - stage timing scaled from April 40/60 within the provided epoch budget
# - graph sampling cap: KINEMATICS_GRAPH_CAP=2000 (matches April best scale and avoids RAM blow-ups)

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

$env:KINEMATICS_QUIET = "1"
$env:KINEMATICS_VAL_PROGRESS = "0"
$env:KINEMATICS_TQDM = "0"
$env:KINEMATICS_SKIP_LBFGS = "1"

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

Write-Host ("[a0+shuffle] epochs={0} stage1_end={1} stage2_end={2} graphcap={3} seed={4}" -f $Epochs, $stage1End, $stage2End, $env:KINEMATICS_GRAPH_CAP, $Seed)

python -m src.training.train_kinematics_predictor --fresh --no-prompt --quiet `
  --epochs $Epochs --adam-epochs $adamEpochs `
  --stage1-end-epoch $stage1End --stage2-end-epoch $stage2End `
  --l0l1-only-epochs 0 `
  --hard-mining-start-epoch 16 `
  --shuffle-graphs --graph-load-seed $Seed

Write-Host "[a0+shuffle] done."
