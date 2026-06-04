# Stage-A default loop: (1) synthetic foundation -> (2) synthetic polish -> (3) clinical geometry finetune -> promote.
#
# Full run (default entry is go_kinematics_production_allfix.ps1, which calls this after phase 1):
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_kinematics_stage_a_ladder.ps1" -Fresh
#
# Resume after phase 1 already done:
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_kinematics_stage_a_ladder.ps1" -SkipFoundation

param(
  [switch]$Fresh,
  [switch]$SkipFoundation,
  [switch]$SkipSyntheticPolish,
  [switch]$SkipClinicalAnchors,
  [switch]$SkipPromote,
  [switch]$RequireClinical,
  [string]$Resume = "outputs/kinematics/production_allfix/kinematics_best.pth",
  [string]$Holdout = "patient007",
  [int]$SyntheticFinetuneEpochs = 40,
  [int]$ClinicalFinetuneEpochs = 25,
  [switch]$ContinuityFocus,
  [switch]$NoContinuityFocus,
  [switch]$Quiet
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Test-KinPatientAnchors {
  $anchorDir = Join-Path $RepoRoot "data\processed\graphs_kinematics_anchors\carreau"
  if (-not (Test-Path $anchorDir)) { return $false }
  return @(Get-ChildItem -Path $anchorDir -Filter "patient*.pt" -ErrorAction SilentlyContinue).Count -gt 0
}

$ckpt = $Resume
if (-not [System.IO.Path]::IsPathRooted($ckpt)) {
  $ckpt = Join-Path $RepoRoot $Resume
}

Write-Host "[ladder] Stage-A: foundation -> synthetic polish -> clinical anchors -> promote"

if (-not $SkipFoundation) {
  Write-Host "[ladder] === phase 1/3: synthetic foundation (3000 graphs, 100 ep) ==="
  $foundationArgs = @()
  if ($Fresh) { $foundationArgs += "-Fresh" }
  if ($Quiet) { $foundationArgs += "-Quiet" }
  & (Join-Path $PSScriptRoot "go_kinematics_production_allfix.ps1") @foundationArgs -FoundationOnly
  if ($LASTEXITCODE -ne 0) { throw "[ladder] phase 1 foundation failed." }
  $ckpt = Join-Path $RepoRoot "outputs\kinematics\production_allfix\kinematics_best.pth"
}

if (-not (Test-Path $ckpt)) {
  throw "[ladder] resume checkpoint missing: $ckpt (run phase 1 first or pass -Resume)."
}

if (-not $SkipSyntheticPolish) {
  Write-Host "[ladder] === phase 2/3: synthetic polish (ContinuityFocus finetune) ==="
  $ftArgs = @(
    "-Resume", $ckpt,
    "-FinetuneEpochs", "$SyntheticFinetuneEpochs"
  )
  $useContinuity = (-not $NoContinuityFocus) -or $ContinuityFocus
  if ($useContinuity) { $ftArgs += "-ContinuityFocus" }
  if ($Quiet) { $ftArgs += "-Quiet" }
  Write-Host ("[ladder] synthetic polish epochs={0}" -f $SyntheticFinetuneEpochs)
  & (Join-Path $PSScriptRoot "go_kinematics_production_allfix_finetune.ps1") @ftArgs
  if ($LASTEXITCODE -ne 0) { throw "[ladder] phase 2 synthetic polish failed." }
  $ckpt = Join-Path $RepoRoot "outputs\kinematics\production_allfix\kinematics_best.pth"
}

$hasClinical = Test-KinPatientAnchors
if ($SkipClinicalAnchors) {
  Write-Host "[ladder] phase 3 clinical anchors skipped (-SkipClinicalAnchors)."
} elseif (-not $hasClinical) {
  $msg = "[ladder] phase 3 skipped: no patient*.pt under data/processed/graphs_kinematics_anchors/carreau/"
  if ($RequireClinical) {
    throw $msg
  }
  Write-Host "[ladder] WARN $msg"
  Write-Host "[ladder] Add patient kine graphs and re-run with -SkipFoundation -SkipSyntheticPolish"
} else {
  Write-Host ("[ladder] === phase 3/3: clinical geometry finetune (holdout={0}, epochs={1}) ===" -f $Holdout, $ClinicalFinetuneEpochs)
  & (Join-Path $PSScriptRoot "go_kinematics_clinical_anchor_finetune.ps1") `
    -Resume $ckpt -Holdout $Holdout -FinetuneEpochs $ClinicalFinetuneEpochs
  if ($LASTEXITCODE -ne 0) { throw "[ladder] phase 3 clinical anchor finetune failed." }

  $clinicalBest = Join-Path $RepoRoot "outputs\kinematics\clinical_anchor_finetune\kinematics_best.pth"
  if (-not $SkipPromote) {
    Write-Host "[ladder] === promotion gates (patient + synthetic + synthetic L2) ==="
    & (Join-Path $PSScriptRoot "promote_kinematics_checkpoint.ps1") `
      -Checkpoint $clinicalBest `
      -Holdout $Holdout
    if ($LASTEXITCODE -ne 0) { throw "[ladder] promotion gates failed (use -SkipPromote to inspect)." }
    Write-Host "[ladder] promoted -> outputs/kinematics/kinematics_best.pth"
  } else {
    Write-Host "[ladder] clinical best (not promoted): $clinicalBest"
    Write-Host "  python scripts/check_kinematics_promotion_gates.py --checkpoint $clinicalBest"
  }
  exit 0
}

if (-not $SkipPromote) {
  Write-Host "[ladder] no clinical phase; promoting synthetic best -> outputs/kinematics/kinematics_best.pth"
  $dest = Join-Path $RepoRoot "outputs\kinematics\kinematics_best.pth"
  Copy-Item $ckpt $dest -Force
  Write-Host "[ladder] copied $ckpt -> $dest"
}
