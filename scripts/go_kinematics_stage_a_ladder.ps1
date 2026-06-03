# Stage-A ladder: foundation (optional) -> synthetic polish (optional) -> clinical anchors + dual promotion gates.
#
# Example (production ep-80 already trained):
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_kinematics_stage_a_ladder.ps1" `
#     -SkipFoundation -Resume outputs/kinematics/production_allfix/kinematics_ckpt_81.pth
#
# Full run from scratch:
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_kinematics_stage_a_ladder.ps1" -Fresh

param(
  [switch]$Fresh,
  [switch]$SkipFoundation,
  [switch]$SkipSyntheticPolish,
  [switch]$SkipPromote,
  [string]$Resume = "outputs/kinematics/production_allfix/kinematics_ckpt_81.pth",
  [string]$Holdout = "patient007",
  [switch]$ContinuityFocus
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$ckpt = $Resume
if (-not [System.IO.Path]::IsPathRooted($ckpt)) {
  $ckpt = Join-Path $RepoRoot $Resume
}

if (-not $SkipFoundation) {
  $foundationArgs = @()
  if ($Fresh) { $foundationArgs += "-Fresh" }
  & (Join-Path $PSScriptRoot "go_kinematics_production_allfix.ps1") @foundationArgs
  if ($LASTEXITCODE -ne 0) { throw "[ladder] foundation failed." }
  $ckpt = Join-Path $RepoRoot "outputs\kinematics\production_allfix\kinematics_best.pth"
}

if (-not $SkipSyntheticPolish) {
  $ftArgs = @("-Resume", $ckpt)
  if ($ContinuityFocus) { $ftArgs += "-ContinuityFocus" }
  & (Join-Path $PSScriptRoot "go_kinematics_production_allfix_finetune.ps1") @ftArgs
  if ($LASTEXITCODE -ne 0) { throw "[ladder] synthetic polish failed." }
  $ckpt = Join-Path $RepoRoot "outputs\kinematics\production_allfix\kinematics_best.pth"
}

& (Join-Path $PSScriptRoot "go_kinematics_clinical_anchor_finetune.ps1") -Resume $ckpt -Holdout $Holdout
if ($LASTEXITCODE -ne 0) { throw "[ladder] clinical anchor finetune failed." }

$clinicalBest = Join-Path $RepoRoot "outputs\kinematics\clinical_anchor_finetune\kinematics_best.pth"
if (-not $SkipPromote) {
  & (Join-Path $PSScriptRoot "promote_kinematics_checkpoint.ps1") `
    -Checkpoint $clinicalBest `
    -Holdout $Holdout
  if ($LASTEXITCODE -ne 0) { throw "[ladder] promotion gates failed." }
  Write-Host "[ladder] promoted -> outputs/kinematics/kinematics_best.pth"
} else {
  Write-Host "[ladder] clinical best (not promoted): $clinicalBest"
  Write-Host "  python scripts/check_kinematics_promotion_gates.py --checkpoint $clinicalBest"
}
