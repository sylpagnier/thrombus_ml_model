# Global biochem_gnn ~5h: all anchors, full horizon, guiding-metric checkpoint selection.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_biochem_gnn_global_guiding_5h.ps1
#   powershell ... -Fresh
#   powershell ... -SkipTrain

param(
    [int] $Epochs = 90,
    [int] $EarlyStop = 28,
    [double] $Lr = 1.5e-4,
    [switch] $Fresh,
    [switch] $SkipTrain
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$RunDir = Join-Path $RepoRoot "outputs/biochem/biochem_gnn/global_guiding_5h"
$SpeciesCkpt = Join-Path $RunDir "species/best.pth"
$BetaCkpt = Join-Path $RunDir "viscosity/beta.pth"
$StagingManifest = Join-Path $RepoRoot "data/reference/biochem_gnn_global_guiding_5h.json"

$InitWarm = "outputs/biochem/biochem_gnn/global_fulltime/species/best.pth"
if (-not (Test-Path (Join-Path $RepoRoot $InitWarm))) {
    $InitWarm = "outputs/biochem/species_gnn_deploy_baseline/species_gnn_best.pth"
}
if (-not $Fresh -and (Test-Path (Join-Path $RepoRoot "outputs/biochem/biochem_gnn/global_guiding_5h/species/best.pth"))) {
    $InitWarm = "outputs/biochem/biochem_gnn/global_guiding_5h/species/best.pth"
    Write-Host "[i] resume warm-start from $InitWarm" -ForegroundColor DarkGray
} else {
    Write-Host "[i] warm-start from $InitWarm" -ForegroundColor DarkGray
}

# Must set before horizon probe (python reads env).
$env:SPECIES_CONTINUOUS_DEPLOY_EVAL_FULL = "1"

function Get-BiochemAnchorHorizons {
    $py = @"
import json, torch
from pathlib import Path
from src.core_physics.species_pushforward_continuous import (
    discover_biochem_anchors,
    deploy_eval_time_index,
    train_t0_max_for_n_times,
)
rows = []
for anc in discover_biochem_anchors():
    p = Path('data/processed/graphs_biochem_anchors') / f'{anc}.pt'
    d = torch.load(p, map_location='cpu', weights_only=False)
    n = int(d.y.shape[0])
    last = deploy_eval_time_index(n)
    rows.append({'anchor': anc, 'n_steps': n, 'deploy_eval_t': last, 'train_t0_max': train_t0_max_for_n_times(n)})
p007 = next((r for r in rows if r['anchor'] == 'patient007'), rows[0] if rows else {})
print(json.dumps({'anchors': [r['anchor'] for r in rows], 'n_anchors': len(rows), 'per_vessel': rows, 'p007': p007}))
"@
    return python -c $py | ConvertFrom-Json
}

$hz = Get-BiochemAnchorHorizons
Write-Host "[i] train anchors on disk: $($hz.n_anchors) -> $($hz.anchors -join ', ')" -ForegroundColor DarkGray
Write-Host "[i] per-vessel t0 caps: $(($hz.per_vessel | ForEach-Object { \"$($_.anchor):t0=$($_.train_t0_max)/deploy=$($_.deploy_eval_t)\" }) -join '; ')" -ForegroundColor DarkGray

# --- guiding-metric-first deploy recipe ---
$env:SPECIES_CONTINUOUS_GROWTH_ONLY_LOSS = "1"
$env:SPECIES_CONTINUOUS_DUAL_HEAD = "1"
$env:SPECIES_CONTINUOUS_PHYSICS_READOUT = "0"
$env:SPECIES_KIN_PER_VESSEL_NORM = "1"
$env:SPECIES_CONTINUOUS_SATURATION_GATE = "1"
$env:SPECIES_CONTINUOUS_MATURE_FP_EXEMPT = "1"
$env:SPECIES_CONTINUOUS_MATURE_FRAC = "0.95"
$env:SPECIES_CONTINUOUS_SATURATION_SCALE = "80"
$env:SPECIES_CONTINUOUS_TEMPORAL_GATE = "1"
$env:SPECIES_CONTINUOUS_TEMPORAL_LAMBDA_MIN = "0.5"
$env:SPECIES_CONTINUOUS_TEMPORAL_LAMBDA_MAX = "1.5"
$env:SPECIES_CONTINUOUS_VEL_DECAY = "1"
$env:SPECIES_CONTINUOUS_TEACHER_NOISE = "0.015"
$env:SPECIES_CONTINUOUS_TEACHER_FP_FRAC = "0.06"
$env:SPECIES_CONTINUOUS_TEACHER_BLUR = "0.22"
$env:SPECIES_CONTINUOUS_TBPTT_TAIL = "8"
$env:SPECIES_CONTINUOUS_CURRICULUM_UNROLL = "1"
$env:SPECIES_CONTINUOUS_CLOSED_LOOP_INIT = "0.35"
$env:SPECIES_CONTINUOUS_FINAL_STATE_WEIGHT = "0.35"
$env:SPECIES_CONTINUOUS_FINAL_STATE_ALL_BAND = "1"
$env:SPECIES_CONTINUOUS_SPEED_FP_WEIGHT = "4.0"
$env:SPECIES_PUSHFORWARD_UNROLL = "12"
$env:SPECIES_PUSHFORWARD_MAX_UNROLL = "200"
$env:SPECIES_CONTINUOUS_DEPLOY_HORIZON = "200"
$env:SPECIES_PUSHFORWARD_TRAIN_T0_PER_VESSEL = "1"
$env:SPECIES_CONTINUOUS_DEPLOY_EVAL_DUAL = "1"
$env:SPECIES_CONTINUOUS_DEPLOY_DUAL_FULL_W = "0.70"
$env:SPECIES_TRAIN_DEPLOY_EVAL_FLOW = "kinematics"
$env:SPECIES_TRAIN_VEL_SOURCE = "gt"
$env:SPECIES_DEPLOY_HORIZON_ALL_PACKS = "1"
$env:SPECIES_DEPLOY_HORIZON_AUX_CAP = "72"
$env:SPECIES_CONTINUOUS_SCORE_CLOUT_W = "0.92"
$env:SPECIES_CONTINUOUS_CLOUT_SCORE = "guiding"
$env:CLOT_GUIDE_RELAX_HOPS = "2"
$env:CLOT_GUIDE_F_BETA = "0.5"
$env:CLOT_GUIDE_IOU_W = "0.45"
$env:CLOT_GUIDE_F05_W = "0.55"
$env:SPECIES_ROLLOUT_DEPLOY_FAITHFUL = "1"
$env:SPECIES_ROLLOUT_IC_SOURCE = "resting"
$env:SPECIES_ROLLOUT_PIN_OTHER = "rest"
$env:SPECIES_ROLLOUT_VEL_SOURCE = "kinematics"
$env:SPECIES_VISCOSITY_CALIB = "1"

if (-not $SkipTrain) {
    Write-Host "[NEW] global biochem_gnn guiding 5h ($Epochs ep, lr=$Lr, $($hz.n_anchors) anchors, dual t53+t$($hz.p007.deploy_eval_t))" -ForegroundColor Cyan
    $pyArgs = @(
        "-m", "src.training.train_biochem_gnn",
        "--step", "deploy",
        "--all-anchors",
        "--val-anchor", "patient007",
        "--epochs", "$Epochs",
        "--early-stop", "$EarlyStop",
        "--lr", "$Lr",
        "--init", $InitWarm,
        "--species-out", $SpeciesCkpt,
        "--beta-out", $BetaCkpt
    )
    if ($Fresh) { $pyArgs += "--fresh" }
    Invoke-PythonRcCheck -Label "biochem_gnn global guiding 5h" -PyArgs $pyArgs
}

Write-Host "[NEW] eval deploy_frozen p007 @ t53 + t$($hz.p007.deploy_eval_t)" -ForegroundColor Cyan
Invoke-PythonRcCheck -Label "eval guiding p007" -PyArgs @(
    "scripts/eval_biochem_gnn_deploy_ab.py",
    "--manifest", $StagingManifest,
    "--modes", "deploy_frozen",
    "--anchors", "patient007",
    "--times", "53,$($hz.p007.deploy_eval_t)",
    "--out", "outputs/biochem/biochem_gnn/global_guiding_5h_eval_p007.json"
)

Write-Host "[NEW] write staging manifest" -ForegroundColor Cyan
python -c @"
import json
from pathlib import Path
from src.biochem_gnn.config import rel_path
m = {
    'name': 'biochem_gnn_global_guiding_5h',
    'version': 1,
    'baseline': {
        'species_gnn_ckpt': rel_path(Path(r'$SpeciesCkpt')),
        'viscosity_beta': rel_path(Path(r'$BetaCkpt')),
        'kinematics_ckpt': 'outputs/kinematics/kinematics_best.pth',
        'train_val_anchor': 'patient007',
        'flow_modes': 'kinematics',
        'gamma_mode': 'max',
        'deploy_horizon': 'full',
        'clot_score': 'guiding',
    },
}
p = Path(r'$StagingManifest')
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(m, indent=2), encoding='utf-8')
print('[OK]', p)
"@

Write-Host "[NEW] viz patient007 clot map (deploy kine)" -ForegroundColor Cyan
Invoke-PythonRcCheck -Label "viz p007 guiding" -PyArgs @(
    "scripts/viz_species_gnn_deploy.py",
    "--anchor", "patient007",
    "--flow", "kinematics",
    "--manifest", $StagingManifest
)

Write-Host "[OK] run dir: $RunDir" -ForegroundColor Green
Write-Host "[OK] manifest: $StagingManifest" -ForegroundColor Green
