# Long precision finetune: extended synthetic polish + clinical patient geometry + promote.
# Skips 100-ep foundation; resumes production/finetune best.
#
# Prereq (Comsol): export missing anchors, then list graphs:
#   python scripts/reextract_anchor_comsol_graphs.py --stems patient001,patient002,patient003,patient004,patient006,patient007
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
  [switch]$SkipReextract,
  [string]$ExtractStems = "",
  [switch]$Quiet
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (-not $SkipReextract -and $ExtractStems.Trim()) {
  Write-Host "[kin-precision] extracting COMSOL anchors: $ExtractStems"
  python scripts/reextract_anchor_comsol_graphs.py --stems $ExtractStems.Trim()
  if ($LASTEXITCODE -ne 0) { throw "[kin-precision] reextract failed (exit $LASTEXITCODE)." }
}

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

$ladderArgs = @{
  SkipFoundation          = $true
  RequireClinical         = $true
  ContinuityFocus         = $true
  Resume                  = $Resume
  Holdout                 = $Holdout
  SyntheticFinetuneEpochs = $SyntheticFinetuneEpochs
  ClinicalFinetuneEpochs  = $ClinicalFinetuneEpochs
}
if ($SkipPromote) { $ladderArgs.SkipPromote = $true }
if ($Quiet) { $ladderArgs.Quiet = $true }

& (Join-Path $PSScriptRoot "go_kinematics_stage_a_ladder.ps1") @ladderArgs
if ($LASTEXITCODE -ne 0) {
  throw "[kin-precision] ladder failed (exit $LASTEXITCODE)."
}

Write-Host "[kin-precision] done."
