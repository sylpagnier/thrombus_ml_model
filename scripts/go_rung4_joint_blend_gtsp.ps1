# Rung 4b: GT species + joint_blend_gtsp (5 minimal feats + FI/Mat + physics blend).
# Requires: dot-source shared mask + CLOT_PHI_DGAMMA_FEATURE_TIME=current (3b stack).
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_rung4_joint_blend_gtsp.ps1" -Fresh
#   powershell ... -EvalOnly   # multi-anchor only (existing ckpt)

param(
    [switch] $Fresh,
    [switch] $EvalOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_clot_phi_shared_env.ps1")
$env:CLOT_PHI_DGAMMA_FEATURE_TIME = "current"

$LegName = "joint_blend_gtsp_rung4"
$LegDir = Join-Path $RepoRoot "outputs/biochem/clot_phi_ladder/$LegName"
$Ckpt = Join-Path $LegDir "clot_phi_best.pth"
$EvalOut = Join-Path $RepoRoot "outputs/biochem/rung4_joint_blend_gtsp/multi_anchor.jsonl"

# joint_blend_gtsp recipe (biology round2)
$env:CLOT_PHI_MODEL = "mlp"
$env:CLOT_PHI_HIDDEN = "32"
$env:CLOT_PHI_MLP_DEPTH = "2"
$env:CLOT_PHI_DROPOUT = "0.15"
$env:CLOT_PHI_EPOCHS = "60"
$env:CLOT_PHI_LR = "1e-3"
$env:CLOT_PHI_WEIGHT_DECAY = "1e-4"
$env:CLOT_PHI_MU_LOG_LAMBDA = "1.5"
$env:CLOT_PHI_DICE_LAMBDA = "0.2"
$env:CLOT_PHI_MINIMAL_FEATURES = "1"
$env:CLOT_PHI_SPECIES_FEATURES = "1"
$env:CLOT_PHI_HYBRID = "1"
$env:CLOT_PHI_BALANCED = "1"
$env:CLOT_PHI_PHYSICS_ORACLE = "0"
$env:CLOT_PHI_JOINT_BIO = "1"
$env:CLOT_PHI_BIO_LAMBDA = "0.25"
$env:CLOT_PHI_JOINT_USE_PRED_SPECIES = "0"
$env:CLOT_PHI_PHYSICS_BLEND = "1"
$env:CLOT_PHI_PHYSICS_BLEND_ALPHA = "0.55"
$env:CLOT_PHI_PHYSICS_MU_RATIO_MAX = "4"
$env:CLOT_PHI_PHYSICS_GELATION_GATE = "1"
$env:CLOT_PHI_SWEEP_DIR = "outputs/biochem/clot_phi_ladder"
$env:CLOT_PHI_SWEEP_LEG = $LegName

Write-Host "[i]  rung4 joint_blend_gtsp | hybrid=$env:CLOT_PHI_HYBRID minimal=$env:CLOT_PHI_MINIMAL_FEATURES dgamma_feat=$env:CLOT_PHI_DGAMMA_FEATURE_TIME" -ForegroundColor Cyan

if (-not $EvalOnly) {
    if ($Fresh) {
        Remove-Item -Force -ErrorAction SilentlyContinue $Ckpt, (Join-Path $LegDir "clot_phi_train_log.jsonl")
    }
    python -m src.training.train_clot_phi_simple
    if (-not (Test-Path $Ckpt)) { throw "Training did not write $Ckpt (check hybrid=1 in log banner)" }
}

New-Item -ItemType Directory -Force -Path (Split-Path $EvalOut) | Out-Null
python scripts/eval_clot_phi_multi_anchor.py --checkpoint $Ckpt --out $EvalOut

python -c @"
import json
from pathlib import Path
p = Path(r'$EvalOut')
rows = [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
for r in rows:
    v = r['val']
    print(f\"{r['anchor']:12} F1={v['clot_f1']:.3f} pred+={v.get('pred_pos_frac',0):.3f} logMAE={v.get('mu_log_mae',0):.3f}\")
f1 = [r['val']['clot_f1'] for r in rows]
print(f\"mean_f1={sum(f1)/len(f1):.3f} min_f1={min(f1):.3f} (gate >= 0.35)\")
"@

python -m src.evaluation.viz_clot_phi_simple --anchor patient007 --checkpoint $Ckpt --time-index -1 --plot-mode scatter

Write-Host "[OK]  rung4 done ckpt=$Ckpt" -ForegroundColor Green
