# Small sweep on teacher-species anchors; checkpoint by multi-anchor mean score.
#
#   powershell -File .\scripts\go_clot_phi_teacher_sweep.ps1

param(
    [int] $Epochs = 45
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_clot_phi_shared_env.ps1")
$env:CLOT_PHI_ANCHOR_DIR = "outputs/biochem/anchors_teacher_species"
$env:CLOT_PHI_MODEL = "mlp"
$env:CLOT_PHI_HYBRID = "1"
$env:CLOT_PHI_SOFT_LABELS = "1"
$env:CLOT_PHI_BALANCED = "1"
$env:CLOT_PHI_DICE_LAMBDA = "0.2"
$env:CLOT_PHI_SPECIES_FEATURES = "0"
$env:CLOT_PHI_JOINT_BIO = "1"
$env:CLOT_PHI_BIO_LAMBDA = "0.25"
$env:CLOT_PHI_JOINT_USE_PRED_SPECIES = "1"
$env:CLOT_PHI_PHYSICS_BLEND = "1"
$env:CLOT_PHI_PHYSICS_MU_RATIO_MAX = "4"
$env:CLOT_PHI_PHYSICS_GELATION_GATE = "1"
$env:CLOT_PHI_MLP_DEPTH = "2"
$env:CLOT_PHI_DROPOUT = "0.15"
$env:CLOT_PHI_LR = "1e-3"
$env:CLOT_PHI_WEIGHT_DECAY = "1e-4"
$env:CLOT_PHI_EPOCHS = "$Epochs"
$env:CLOT_PHI_POS_WEIGHT_CAP = "8"

$SweepRoot = "outputs/biochem/sweep_clot_phi_teacher"
New-Item -ItemType Directory -Force -Path $SweepRoot | Out-Null

$Legs = @(
    @{ Name = "m15_a75_t045"; MuLog = "1.5"; Alpha = "0.75"; Thresh = "0.045"; Hidden = "32" },
    @{ Name = "m20_a75_t045"; MuLog = "2.0"; Alpha = "0.75"; Thresh = "0.045"; Hidden = "32" },
    @{ Name = "m20_a55_t045"; MuLog = "2.0"; Alpha = "0.55"; Thresh = "0.045"; Hidden = "32" },
    @{ Name = "m20_a75_t055"; MuLog = "2.0"; Alpha = "0.75"; Thresh = "0.055"; Hidden = "48" }
)

$BestMean = -999.0
$BestLeg = ""

foreach ($Leg in $Legs) {
    $LegDir = Join-Path $SweepRoot $Leg.Name
    New-Item -ItemType Directory -Force -Path $LegDir | Out-Null
    $env:CLOT_PHI_MU_LOG_LAMBDA = $Leg.MuLog
    $env:CLOT_PHI_PHYSICS_BLEND_ALPHA = $Leg.Alpha
    $env:CLOT_PHI_THRESH_SI = $Leg.Thresh
    $env:CLOT_PHI_HIDDEN = $Leg.Hidden
    $Ckpt = Join-Path $LegDir "clot_phi_best.pth"
    Remove-Item -Force -ErrorAction SilentlyContinue $Ckpt, "outputs\biochem\clot_phi_best.pth", "outputs\biochem\clot_phi_train_log.jsonl"

    Write-Host "[NEW] leg=$($Leg.Name) mu_log=$($Leg.MuLog) alpha=$($Leg.Alpha) thr=$($Leg.Thresh) h=$($Leg.Hidden)" -ForegroundColor Cyan
    python -m src.training.train_clot_phi_simple
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Copy-Item -Force "outputs\biochem\clot_phi_best.pth" $Ckpt

    $EvalOut = Join-Path $LegDir "multi_anchor_eval.jsonl"
    python scripts/eval_clot_phi_multi_anchor.py --checkpoint $Ckpt --out $EvalOut
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $MeanLine = python -c "import json; from pathlib import Path; p=Path(r'$EvalOut'); rows=[json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]; f1=sum(r['val']['clot_f1'] for r in rows)/len(rows); print(f'{f1:.4f}')"
    $MeanF1 = [double]$MeanLine
    Write-Host "[i]  leg $($Leg.Name) mean_f1=$MeanF1" -ForegroundColor Yellow
    if ($MeanF1 -gt $BestMean) {
        $BestMean = $MeanF1
        $BestLeg = $Leg.Name
        Copy-Item -Force $Ckpt "outputs\biochem\clot_phi_best_teacher_sweep.pth"
    }
}

Write-Host "[OK]  best leg=$BestLeg mean_f1=$BestMean -> outputs\biochem\clot_phi_best_teacher_sweep.pth" -ForegroundColor Green
