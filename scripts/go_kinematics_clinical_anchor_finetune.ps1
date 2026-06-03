# Stage-A phase 3: clinical patient anchor finetune (Carreau kine graphs under graphs_kinematics_anchors/).
# Resumes production healthy checkpoint; held-out patients stay in val; synthetic regularizer via cap.
#
# Prereq: patient*.pt under data/processed/graphs_kinematics_anchors/carreau/
#
# Example:
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_kinematics_clinical_anchor_finetune.ps1"
#   ... -Resume outputs\kinematics\production_allfix\kinematics_best.pth -Holdout patient007,patient003

param(
  [string]$Resume = "outputs/kinematics/production_allfix/kinematics_best.pth",
  [string]$Holdout = "patient007",
  [int]$FinetuneEpochs = 25,
  [double]$FinetuneLr = 5e-6,
  [int]$SyntheticCap = 120,
  [double]$ClinicalBoost = 10.0,
  [string]$OutDir = "outputs/kinematics/clinical_anchor_finetune"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

# Allfix architecture (must match production)
$env:KINEMATICS_PHYS_GAT_PRIORS_MULTIPLY_BEFORE_ADDITIVE = "1"
$env:KINEMATICS_BC_ENVELOPE = "1"
$env:KINEMATICS_BC_LAMBDA = "10.0"
$env:KINEMATICS_WSS_FUSE = "1"
$env:KINEMATICS_FOURIER_LEARNABLE = "1"
$env:KINEMATICS_SKIP_LBFGS = "1"
$env:KINEMATICS_QUIET = "1"
$env:KINEMATICS_VAL_PROGRESS = "0"
$env:KINEMATICS_VAL_EVERY = "1"

$env:KINEMATICS_INCLUDE_PATIENT_ANCHORS = "1"
$env:KINEMATICS_VAL_HOLDOUT_PATIENT_STEMS = $Holdout
$env:KINEMATICS_CLINICAL_ANCHOR_BOOST = "$ClinicalBoost"
$env:KINEMATICS_OUTPUT_DIR = $OutDir
$env:KINEMATICS_GRAPH_CAP = "$SyntheticCap"

# Val: patient holdout + fixed synthetic holdout (L2 floor). Best ckpt uses dual gates.
$env:KINEMATICS_SYNTHETIC_VAL_RATIO = "0.15"
$env:KINEMATICS_SYNTHETIC_VAL_MIN = "20"
$env:KINEMATICS_SYNTHETIC_VAL_MIN_L2 = "6"
$env:KINEMATICS_DUAL_PROMOTION_GATES = "1"
$env:KINEMATICS_GATE_MAX_PATIENT_REL_L2 = "0.25"
$env:KINEMATICS_GATE_MAX_SYNTHETIC_REL_L2 = "0.20"
$env:KINEMATICS_GATE_MAX_SYNTHETIC_L2_REL_L2 = "0.22"

$resumePath = $Resume
if (-not [System.IO.Path]::IsPathRooted($resumePath)) {
  $resumePath = Join-Path $RepoRoot $Resume
}
if (-not (Test-Path $resumePath)) {
  throw "[kin-clinical-ft] resume checkpoint missing: $resumePath"
}

Write-Host ("[kin-clinical-ft] resume={0} holdout={1} epochs={2} lr={3} synth_cap={4} out={5}" -f `
  $resumePath, $Holdout, $FinetuneEpochs, $FinetuneLr, $SyntheticCap, $OutDir)

python scripts/finetune_kine_patient_anchors.py `
  --epochs $FinetuneEpochs `
  --lr $FinetuneLr `
  --synthetic-cap $SyntheticCap `
  --resume $resumePath `
  --out-dir $OutDir

if ($LASTEXITCODE -ne 0) {
  throw "[kin-clinical-ft] training failed (exit $LASTEXITCODE)."
}

Write-Host "[kin-clinical-ft] done. Run promotion gates before copying to global kinematics_best.pth:"
Write-Host "  python scripts/check_kinematics_promotion_gates.py --checkpoint $OutDir/kinematics_best.pth"
