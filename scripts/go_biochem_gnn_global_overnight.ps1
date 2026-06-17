# Overnight global biochem_gnn (~8-10h): resume full-horizon species + beta + viz.
#
# Resume from outputs/biochem/biochem_gnn/global_fulltime/species/best.pth (4h run).
# Tuned for 4GB GPU: GT vel during train, pred kine at deploy eval only.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_biochem_gnn_global_overnight.ps1
#   powershell ... -SkipTrain   # viz/eval only

param(
    [int] $Epochs = 120,
    [int] $EarlyStop = 30,
    [switch] $Fresh,
    [switch] $SkipTrain
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

$RunDir = Join-Path $RepoRoot "outputs/biochem/biochem_gnn/global_fulltime"
$SpeciesCkpt = Join-Path $RunDir "species/best.pth"
$BetaCkpt = Join-Path $RunDir "viscosity/beta.pth"
$StagingManifest = Join-Path $RepoRoot "data/reference/biochem_gnn_global_fulltime.json"

# Resume 4h winner unless -Fresh (then fall back to deploy baseline warm-start).
$InitWarm = "outputs/biochem/biochem_gnn/global_fulltime/species/best.pth"
if ($Fresh -or -not (Test-Path (Join-Path $RepoRoot $InitWarm))) {
    $InitWarm = "outputs/biochem/species_gnn_deploy_baseline/species_gnn_best.pth"
}
Write-Host "[i] warm-start: $InitWarm" -ForegroundColor DarkGray

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

# Same deploy recipe as 4h; longer schedule + slightly lower lr for finetune stability.
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
$env:SPECIES_CONTINUOUS_TEACHER_NOISE = "0.02"
$env:SPECIES_CONTINUOUS_TEACHER_FP_FRAC = "0.08"
$env:SPECIES_CONTINUOUS_TEACHER_BLUR = "0.25"
$env:SPECIES_CONTINUOUS_TBPTT_TAIL = "8"
$env:SPECIES_CONTINUOUS_CURRICULUM_UNROLL = "1"
$env:SPECIES_CONTINUOUS_CLOSED_LOOP_INIT = "0.45"
$env:SPECIES_CONTINUOUS_FINAL_STATE_WEIGHT = "0.40"
$env:SPECIES_CONTINUOUS_FINAL_STATE_ALL_BAND = "1"
$env:SPECIES_CONTINUOUS_SPEED_FP_WEIGHT = "4.0"
$env:SPECIES_PUSHFORWARD_UNROLL = "12"
$env:SPECIES_PUSHFORWARD_MAX_UNROLL = "200"
$env:SPECIES_CONTINUOUS_DEPLOY_HORIZON = "200"
$env:SPECIES_PUSHFORWARD_TRAIN_T0_PER_VESSEL = "1"
$env:SPECIES_CONTINUOUS_DEPLOY_EVAL_FULL = "1"
$env:SPECIES_TRAIN_DEPLOY_EVAL_FLOW = "kinematics"
$env:SPECIES_TRAIN_VEL_SOURCE = "gt"
$env:SPECIES_DEPLOY_HORIZON_ALL_PACKS = "1"
$env:SPECIES_DEPLOY_HORIZON_AUX_CAP = "72"
# Slightly less single-time dominance; still clot-first.
$env:SPECIES_CONTINUOUS_SCORE_CLOUT_W = "0.65"
$env:SPECIES_CONTINUOUS_CLOUT_SCORE = "guiding"
$env:CLOT_GUIDE_RELAX_HOPS = "2"
$env:CLOT_GUIDE_F_BETA = "0.5"
$env:SPECIES_ROLLOUT_DEPLOY_FAITHFUL = "1"
$env:SPECIES_ROLLOUT_IC_SOURCE = "resting"
$env:SPECIES_ROLLOUT_PIN_OTHER = "rest"
$env:SPECIES_ROLLOUT_VEL_SOURCE = "kinematics"
$env:SPECIES_VISCOSITY_CALIB = "1"

if (-not $SkipTrain) {
    Write-Host "[NEW] overnight global biochem_gnn ($Epochs ep, resume, full horizon)" -ForegroundColor Cyan
    $pyArgs = @(
        "-m", "src.training.train_biochem_gnn",
        "--step", "deploy",
        "--all-anchors",
        "--val-anchor", "patient007",
        "--epochs", "$Epochs",
        "--early-stop", "$EarlyStop",
        "--init", $InitWarm,
        "--species-out", $SpeciesCkpt,
        "--beta-out", $BetaCkpt
    )
    if ($Fresh) { $pyArgs += "--fresh" }
    Invoke-PythonRcCheck -Label "biochem_gnn global overnight" -PyArgs $pyArgs
}

Write-Host "[NEW] eval deploy_frozen p007 @ t53 + t200" -ForegroundColor Cyan
Invoke-PythonRcCheck -Label "eval global" -PyArgs @(
    "scripts/eval_biochem_gnn_deploy_ab.py",
    "--manifest", $StagingManifest,
    "--modes", "deploy_frozen",
    "--anchors", "patient007",
    "--times", "53,200",
    "--out", "outputs/biochem/biochem_gnn/global_fulltime_eval_overnight.json"
)

Write-Host "[NEW] viz patient007 + synthetic seed 17" -ForegroundColor Cyan
& (Join-Path $PSScriptRoot "go_biochem_gnn_global_4h.ps1") -SkipTrain

Write-Host "[OK] run dir: $RunDir" -ForegroundColor Green
