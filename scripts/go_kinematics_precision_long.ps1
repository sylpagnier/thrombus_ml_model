# Long precision finetune: extended synthetic polish + clinical patient geometry + promote.
# Skips 100-ep foundation; resumes production/finetune best.
#
# Prereq (Comsol):
#   python scripts/reextract_anchor_comsol_graphs.py
#   Get-ChildItem data\processed\graphs_kinematics_anchors\carreau\patient*.pt
#
# One command:
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_kinematics_precision_long.ps1"

param(
  [string]$Resume = "outputs/kinematics/production_allfix/kinematics_best.pth",
  [string]$Holdout = "patient007",
  [int]$SyntheticFinetuneEpochs = 60,
  [int]$ClinicalFinetuneEpochs = 50,
  [switch]$SkipPromote,
  [switch]$Quiet
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$anchorDir = Join-Path $RepoRoot "data\processed\graphs_kinematics_anchors\carreau"
$nPatient = @(Get-ChildItem -Path $anchorDir -Filter "patient*.pt" -ErrorAction SilentlyContinue).Count
if ($nPatient -lt 1) {
  throw @"
[kin-precision] no patient*.pt under $anchorDir
Run first: python scripts/reextract_anchor_comsol_graphs.py
"@
}

Write-Host ("[kin-precision] patient kine graphs={0} | synth_ft_epochs={1} | clinical_epochs={2} | holdout={3}" -f `
  $nPatient, $SyntheticFinetuneEpochs, $ClinicalFinetuneEpochs, $Holdout)
Write-Host "[kin-precision] resume=$Resume (git pull recommended before long run)"

# Named args only (array splat can bind -ContinuityFocus to [int] params on PS 5.x).
$ladderParams = @{
  SkipFoundation          = $true
  RequireClinical         = $true
  ContinuityFocus         = $true
  Resume                  = $Resume
  Holdout                 = $Holdout
  SyntheticFinetuneEpochs = $SyntheticFinetuneEpochs
  ClinicalFinetuneEpochs  = $ClinicalFinetuneEpochs
}
if ($SkipPromote) { $ladderParams.SkipPromote = $true }
if ($Quiet) { $ladderParams.Quiet = $true }

& (Join-Path $PSScriptRoot "go_kinematics_stage_a_ladder.ps1") @ladderParams
if ($LASTEXITCODE -ne 0) {
  throw "[kin-precision] ladder failed (exit $LASTEXITCODE)."
}

Write-Host "[kin-precision] done."
