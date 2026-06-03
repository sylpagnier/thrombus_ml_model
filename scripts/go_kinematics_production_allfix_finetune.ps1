# Carreau L2-heavy finetune after production allfix (ep ~80 best).
# Targets lower div_u via stronger BC envelope + lower LR; Adam-only (LBFGS hurts on this recipe).
#
# One line (resume from best.pth, +40 Carreau epochs):
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_kinematics_production_allfix_finetune.ps1"
#
# Stronger continuity / BC focus:
#   ... -ContinuityFocus
#
# Experimental LBFGS polish (risky; 5 steps only, from best):
#   ... -TryLbfgs
#
# Writes: outputs/kinematics/production_allfix/kinematics_best.pth (updates if val improves)

param(
  [string]$Resume = "best",
  [double]$FinetuneLr = 5e-6,
  [int]$FinetuneEpochs = 40,
  [int]$HardMiningStart = 0,
  [switch]$ContinuityFocus,
  [switch]$TryLbfgs,
  [switch]$Quiet
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$env:KINEMATICS_PHYS_GAT_PRIORS_MULTIPLY_BEFORE_ADDITIVE = "1"
$env:KINEMATICS_BC_ENVELOPE = "1"
$env:KINEMATICS_WSS_FUSE = "1"
$env:KINEMATICS_FOURIER_LEARNABLE = "1"
$env:KINEMATICS_OUTPUT_DIR = "outputs/kinematics/production_allfix"
$env:KINEMATICS_VAL_EVERY = "1"
Remove-Item Env:KINEMATICS_GRAPH_CAP -ErrorAction SilentlyContinue

if ($ContinuityFocus) {
  $env:KINEMATICS_BC_LAMBDA = "25.0"
} else {
  $env:KINEMATICS_BC_LAMBDA = "15.0"
}

$outDir = Join-Path $RepoRoot "outputs/kinematics/production_allfix"
$bestCkpt = Join-Path $outDir "kinematics_best.pth"

function Get-KinResumeEpoch {
  param([string]$Path)
  if (-not (Test-Path $Path)) { return 0 }
  $py = @"
import torch
from pathlib import Path
p = Path(r'$Path')
c = torch.load(p, map_location='cpu', weights_only=False)
if isinstance(c, dict):
    for k in ('epoch', 'best_epoch'):
        if k in c:
            print(int(c[k]))
            raise SystemExit
print(0)
"@
  return [int](& python -c $py)
}

$resumePath = $bestCkpt
if ($Resume -eq "best") {
  if (-not (Test-Path $bestCkpt)) {
    throw "[kin-prod-ft] missing $bestCkpt -- run go_kinematics_production_allfix.ps1 first."
  }
  $resumePath = $bestCkpt
} elseif ($Resume -eq "latest") {
  $stateLatest = Join-Path $outDir "kinematics_state_latest.pth"
  $ckptLatest = Join-Path $outDir "kinematics_ckpt_latest.pth"
  if (Test-Path $stateLatest) { $resumePath = $stateLatest }
  elseif (Test-Path $ckptLatest) { $resumePath = $ckptLatest }
  else { throw "[kin-prod-ft] no latest checkpoint in $outDir" }
} else {
  $resumePath = $Resume
  if (-not (Test-Path $resumePath)) { throw "[kin-prod-ft] resume not found: $resumePath" }
}

$startEp = Get-KinResumeEpoch -Path $resumePath
$totalEpochs = $startEp + $FinetuneEpochs
$adamEpochs = if ($TryLbfgs) { $totalEpochs - 5 } else { $totalEpochs }

if ($TryLbfgs) {
  Remove-Item Env:KINEMATICS_SKIP_LBFGS -ErrorAction SilentlyContinue
  Write-Host "[kin-prod-ft] WARN TryLbfgs: last 5 epochs use LBFGS (experimental; may NaN)."
} else {
  $env:KINEMATICS_SKIP_LBFGS = "1"
}

if ($Quiet) {
  $env:KINEMATICS_QUIET = "1"
  $env:KINEMATICS_VAL_PROGRESS = "0"
  $env:KINEMATICS_TQDM = "0"
}

$weightData = if ($ContinuityFocus) { "350.0" } else { "500.0" }
$focusLabel = if ($ContinuityFocus) { "continuity+BC" } else { "balanced" }
Write-Host ("[kin-prod-ft] resume={0} start_ep={1} total_epochs={2} lr={3} focus={4} BC_lambda={5}" -f `
  $resumePath, ($startEp + 1), $totalEpochs, $FinetuneLr, $focusLabel, $env:KINEMATICS_BC_LAMBDA)

$trainArgs = @(
  "-m", "src.training.train_kinematics_predictor",
  "--no-prompt",
  "--resume", $resumePath,
  "--epochs", "$totalEpochs",
  "--adam-epochs", "$adamEpochs",
  "--stage1-end-epoch", "40",
  "--stage2-end-epoch", "60",
  "--geometry-phase", "l2_heavy",
  "--hard-mining-start-epoch", "$HardMiningStart",
  "--finetune-lr", "$FinetuneLr",
  "--weight-data", $weightData,
  "--shuffle-graphs",
  "--graph-load-seed", "42"
)
if ($TryLbfgs) {
  $trainArgs += @("--max-lbfgs-graphs", "2")
}
if ($Quiet) { $trainArgs += "--quiet" }

& python @trainArgs
if ($LASTEXITCODE -ne 0) {
  throw "[kin-prod-ft] training failed (exit $LASTEXITCODE)."
}

Write-Host "[kin-prod-ft] done. best -> outputs/kinematics/production_allfix/kinematics_best.pth"
