# Production Stage-A kinematics: default = full 3-phase loop (foundation + polish + clinical anchors).
#
# Default (foundation -> synthetic polish -> clinical geometry finetune -> promote):
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_kinematics_production_allfix.ps1"
#
# Foundation only (legacy / debugging):
#   ... -FoundationOnly
#
# Start foundation from scratch:
#   ... -Fresh
#
# Outputs: outputs/kinematics/production_allfix/kinematics_best.pth (phase 1-2)
#          outputs/kinematics/kinematics_best.pth (promoted after phase 3)

param(
  [switch]$Fresh,
  [switch]$FoundationOnly,
  [switch]$SkipSyntheticPolish,
  [switch]$SkipClinicalAnchors,
  [switch]$SkipPromote,
  [switch]$RequireClinical,
  [string]$Holdout = "patient007",
  [switch]$NoContinuityFocus,
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

$env:KINEMATICS_PHYS_GAT_PRIORS_MULTIPLY_BEFORE_ADDITIVE = "1"
$env:KINEMATICS_BC_ENVELOPE = "1"
$env:KINEMATICS_BC_LAMBDA = "10.0"
$env:KINEMATICS_WSS_FUSE = "1"
$env:KINEMATICS_FOURIER_LEARNABLE = "1"
$env:KINEMATICS_VAL_EVERY = "1"
$env:KINEMATICS_OUTPUT_DIR = "outputs/kinematics/production_allfix"
# Adam-only default: 20260601 production run -- LBFGS ep 86+ hurt val; ep 98-99 NaN (April same pattern).
$env:KINEMATICS_SKIP_LBFGS = "1"

if ($Quiet) {
  $env:KINEMATICS_QUIET = "1"
  $env:KINEMATICS_VAL_PROGRESS = "0"
  $env:KINEMATICS_TQDM = "0"
} else {
  Remove-Item Env:KINEMATICS_QUIET -ErrorAction SilentlyContinue
  Remove-Item Env:KINEMATICS_VAL_PROGRESS -ErrorAction SilentlyContinue
  Remove-Item Env:KINEMATICS_TQDM -ErrorAction SilentlyContinue
}

if ($GraphCap -gt 0) {
  $env:KINEMATICS_GRAPH_CAP = "$GraphCap"
} else {
  Remove-Item Env:KINEMATICS_GRAPH_CAP -ErrorAction SilentlyContinue
}

$outDir = Join-Path $RepoRoot "outputs/kinematics/production_allfix"
$skipLbfgsFlag = Join-Path $outDir ".skip_lbfgs_after_crash"
$stateLatest = Join-Path $outDir "kinematics_state_latest.pth"
$ckptLatest = Join-Path $outDir "kinematics_ckpt_latest.pth"

function Get-KinNextEpoch {
  param([string]$Dir)
  $py = @"
import sys
from pathlib import Path
import torch
root = Path(r'$Dir')
for name in ('kinematics_state_latest.pth', 'kinematics_ckpt_latest.pth'):
    p = root / name
    if not p.exists():
        continue
    c = torch.load(p, map_location='cpu', weights_only=False)
    if isinstance(c, dict) and 'epoch' in c:
        print(int(c['epoch']) + 1)
        sys.exit(0)
    m = __import__('re').search(r'kinematics_ckpt_(\d+)\.pth$', name)
    if m:
        print(int(m.group(1)))
        sys.exit(0)
print(0)
"@
  return [int](& python -c $py)
}

if ($Fresh) {
  Remove-Item $skipLbfgsFlag -ErrorAction SilentlyContinue
  Remove-Item $stateLatest, $ckptLatest -ErrorAction SilentlyContinue
  Get-ChildItem $outDir -Filter "kinematics_ckpt_*.pth" -ErrorAction SilentlyContinue | Remove-Item -Force
  Get-ChildItem $outDir -Filter "kinematics_state_*.pth" -ErrorAction SilentlyContinue | Remove-Item -Force
  Write-Host "[kin-prod] -Fresh: cleared production_allfix checkpoints."
}

$capLabel = if ($GraphCap -gt 0) { "$GraphCap" } else { "all" }
Write-Host ("[kin-prod] epochs={0} adam={1} stage1={2} stage2={3} graphs={4} seed={5}" -f `
  $Epochs, $AdamEpochs, $Stage1End, $Stage2End, $capLabel, $Seed)

$attempt = 0
$maxAttempts = 50

while ($attempt -lt $maxAttempts) {
  $attempt++
  Remove-Item Env:KINEMATICS_SKIP_LBFGS -ErrorAction SilentlyContinue
  if (Test-Path $skipLbfgsFlag) {
    $env:KINEMATICS_SKIP_LBFGS = "1"
    Write-Host "[kin-prod] LBFGS skip flag set (prior crash at Adam handoff); Adam-only for remaining epochs."
  } else {
    Remove-Item Env:KINEMATICS_SKIP_LBFGS -ErrorAction SilentlyContinue
  }

  $hasCkpt = (Test-Path $stateLatest) -or (Test-Path $ckptLatest)
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
  if ($Quiet) { $trainArgs += "--quiet" }
  if ($Fresh -and $attempt -eq 1 -and -not $hasCkpt) {
    $trainArgs += "--fresh"
    Write-Host "[kin-prod] attempt $attempt : fresh start."
  } elseif ($hasCkpt) {
    $trainArgs += @("--resume", "latest")
    $nextEp = Get-KinNextEpoch -Dir $outDir
    Write-Host "[kin-prod] attempt $attempt : resume latest (next epoch $nextEp)."
  } else {
    $trainArgs += "--fresh"
    Write-Host "[kin-prod] attempt $attempt : no checkpoint; fresh start."
  }

  & python @trainArgs
  $exit = $LASTEXITCODE
  if ($exit -eq 0) {
    Write-Host "[kin-prod] phase 1 finished OK -> outputs/kinematics/production_allfix/kinematics_best.pth"
    break
  }

  if (-not ((Test-Path $stateLatest) -or (Test-Path $ckptLatest))) {
    throw "[kin-prod] training failed (exit $exit) and no checkpoint to resume."
  }

  $nextEp = Get-KinNextEpoch -Dir $outDir
  if ($nextEp -ge $AdamEpochs) {
    New-Item -ItemType File -Path $skipLbfgsFlag -Force | Out-Null
    Write-Host "[kin-prod] failed near/after Adam epoch $AdamEpochs; next retry skips LBFGS (best.pth kept from Adam phase)."
  }
  Write-Host "[kin-prod] failed (exit $exit); retrying in 5s..."
  Start-Sleep -Seconds 5
  $Fresh = $false
}

if ($attempt -ge $maxAttempts) {
  throw "[kin-prod] exceeded $maxAttempts resume attempts."
}

if ($FoundationOnly) {
  Write-Host "[kin-prod] -FoundationOnly: skipping phases 2-3 (synthetic polish + clinical anchors)."
  exit 0
}

Write-Host "[kin-prod] chaining phases 2-3 (synthetic polish + clinical geometry finetune)..."
$ladderArgs = @(
  "-SkipFoundation",
  "-Holdout", $Holdout
)
if ($SkipSyntheticPolish) { $ladderArgs += "-SkipSyntheticPolish" }
if ($SkipClinicalAnchors) { $ladderArgs += "-SkipClinicalAnchors" }
if ($SkipPromote) { $ladderArgs += "-SkipPromote" }
if ($RequireClinical) { $ladderArgs += "-RequireClinical" }
if (-not $NoContinuityFocus) { $ladderArgs += "-ContinuityFocus" }
if ($Quiet) { $ladderArgs += "-Quiet" }

& (Join-Path $PSScriptRoot "go_kinematics_stage_a_ladder.ps1") @ladderArgs
if ($LASTEXITCODE -ne 0) {
  throw "[kin-prod] post-foundation ladder failed (exit $LASTEXITCODE)."
}
