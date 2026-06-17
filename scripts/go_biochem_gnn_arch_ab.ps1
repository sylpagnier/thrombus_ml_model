# Fair A/B: GraphSAGE pushforward vs GINO-derivative pushforward (same deploy recipe).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_biochem_gnn_arch_ab.ps1
#   powershell ... -Fresh
#   powershell ... -SkipTrain -Leg sage
#   powershell ... -Smoke   # 8 ep quick sanity (not for winner pick)

param(
    [int] $Epochs = 75,
    [int] $EarlyStop = 24,
    [double] $Lr = 1.5e-4,
    [ValidateSet("both", "sage", "gnode")]
    [string] $Leg = "both",
    [switch] $Fresh,
    [switch] $SkipTrain,
    [switch] $Smoke
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

if ($Smoke) {
    $Epochs = 8
    $EarlyStop = 4
    Write-Host "[i] smoke mode: $Epochs ep" -ForegroundColor DarkGray
}

$RunRoot = Join-Path $RepoRoot "outputs/biochem/biochem_gnn/arch_ab"
$SageDir = Join-Path $RunRoot "sage"
$GnodeDir = Join-Path $RunRoot "gnode"
$SageCkpt = Join-Path $SageDir "species/best.pth"
$GnodeCkpt = Join-Path $GnodeDir "species/best.pth"
$SageBeta = Join-Path $SageDir "viscosity/beta.pth"
$GnodeBeta = Join-Path $GnodeDir "viscosity/beta.pth"
$SharedBeta = "outputs/biochem/biochem_gnn/global_guiding_5h/viscosity/beta.pth"
if (-not (Test-Path (Join-Path $RepoRoot $SharedBeta))) {
    $SharedBeta = "outputs/biochem/species_gnn_deploy_baseline/viscosity_beta.pth"
}
Write-Host "[i] shared gelation beta: $SharedBeta" -ForegroundColor DarkGray
$SageManifest = Join-Path $RepoRoot "data/reference/biochem_gnn_arch_ab_sage.json"
$GnodeManifest = Join-Path $RepoRoot "data/reference/biochem_gnn_arch_ab_gnode.json"
$SummaryOut = Join-Path $RunRoot "arch_ab_summary.json"

$InitWarm = "outputs/biochem/biochem_gnn/global_guiding_5h/species/best.pth"
foreach ($cand in @(
    "outputs/biochem/biochem_gnn/global_fulltime/species/best.pth",
    "outputs/biochem/species_gnn_deploy_baseline/species_gnn_best.pth",
    "outputs/biochem/species_snapshot_s33/best.pth"
)) {
    if (-not (Test-Path (Join-Path $RepoRoot $InitWarm))) {
        if (Test-Path (Join-Path $RepoRoot $cand)) { $InitWarm = $cand }
    }
}
Write-Host "[i] shared warm-start: $InitWarm" -ForegroundColor DarkGray

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

function Set-ArchAbTrainEnv {
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
}

function Train-ArchLeg {
    param(
        [string] $Arch,
        [string] $SpeciesOut,
        [string] $InitPath
    )
    Set-ArchAbTrainEnv
    $env:SPECIES_PUSHFORWARD_ARCH = $Arch
    Write-Host "[NEW] train arch=$Arch ($Epochs ep, lr=$Lr, init=$InitPath)" -ForegroundColor Cyan
    $pyArgs = @(
        "-m", "src.training.train_biochem_gnn",
        "--step", "species",
        "--all-anchors",
        "--val-anchor", "patient007",
        "--epochs", "$Epochs",
        "--early-stop", "$EarlyStop",
        "--lr", "$Lr",
        "--init", $InitPath,
        "--arch", $Arch,
        "--species-out", $SpeciesOut
    )
    if ($Fresh) { $pyArgs += "--fresh" }
    Invoke-PythonRcCheck -Label "arch_ab train $Arch" -PyArgs $pyArgs
}

function Resolve-LegInit {
    param([string] $LegCkpt)
    if (-not $Fresh -and (Test-Path $LegCkpt)) {
        return $LegCkpt
    }
    return (Join-Path $RepoRoot $InitWarm)
}

function Write-ArchManifest {
    param(
        [string] $Name,
        [string] $ManifestPath,
        [string] $SpeciesCkptPath,
        [string] $BetaCkptPath,
        [string] $Arch
    )
    python -c @"
import json
from pathlib import Path
from src.biochem_gnn.config import rel_path
m = {
    'name': '$Name',
    'version': 1,
    'baseline': {
        'species_gnn_ckpt': rel_path(Path(r'$SpeciesCkptPath')),
        'viscosity_beta': rel_path(Path(r'$BetaCkptPath')),
        'kinematics_ckpt': 'outputs/kinematics/kinematics_best.pth',
        'train_val_anchor': 'patient007',
        'flow_modes': 'kinematics',
        'gamma_mode': 'max',
        'deploy_horizon': 'full',
        'clot_score': 'guiding',
        'pushforward_arch': '$Arch',
    },
}
p = Path(r'$ManifestPath')
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(m, indent=2), encoding='utf-8')
print('[OK]', p)
"@
}

function Eval-ArchLeg {
    param(
        [string] $Arch,
        [string] $ManifestPath,
        [string] $OutJson,
        [string] $Times = "53,200"
    )
    Write-Host "[NEW] eval deploy_frozen all anchors (arch=$Arch, times=$Times)" -ForegroundColor Cyan
    Invoke-PythonRcCheck -Label "arch_ab eval $Arch" -PyArgs @(
        "scripts/eval_biochem_gnn_deploy_ab.py",
        "--manifest", $ManifestPath,
        "--modes", "deploy_frozen",
        "--times", $Times,
        "--out", $OutJson
    )
}

$hz = Get-BiochemAnchorHorizons
Write-Host "[i] anchors: $($hz.n_anchors) -> $($hz.anchors -join ', ')" -ForegroundColor DarkGray
Write-Host "[i] p007 deploy t=$($hz.p007.deploy_eval_t)" -ForegroundColor DarkGray

$legs = @()
if ($Leg -eq "both") { $legs = @("sage", "gnode") } else { $legs = @($Leg) }

if (-not $SkipTrain) {
    foreach ($arch in $legs) {
        if ($arch -eq "sage") {
            $init = Resolve-LegInit -LegCkpt $SageCkpt
            Train-ArchLeg -Arch sage -SpeciesOut $SageCkpt -InitPath $init
        } else {
            $init = Resolve-LegInit -LegCkpt $GnodeCkpt
            Train-ArchLeg -Arch gnode -SpeciesOut $GnodeCkpt -InitPath $init
        }
        python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"
    }
}

$betaPath = Join-Path $RepoRoot $SharedBeta
foreach ($arch in $legs) {
    if ($arch -eq "sage") {
        $destBeta = $SageBeta
    } else {
        $destBeta = $GnodeBeta
    }
    $destDir = Split-Path -Parent $destBeta
    if (-not (Test-Path $destDir)) { New-Item -ItemType Directory -Path $destDir -Force | Out-Null }
    if (Test-Path $betaPath) {
        Copy-Item -Force $betaPath $destBeta
    }
}

foreach ($arch in $legs) {
    if ($arch -eq "sage") {
        Write-ArchManifest -Name "biochem_gnn_arch_ab_sage" -ManifestPath $SageManifest `
            -SpeciesCkptPath $SageCkpt -BetaCkptPath $SageBeta -Arch sage
        Eval-ArchLeg -Arch sage -ManifestPath $SageManifest `
            -OutJson (Join-Path $SageDir "eval_deploy_frozen.json") `
            -Times "53,$($hz.p007.deploy_eval_t)"
    } else {
        Write-ArchManifest -Name "biochem_gnn_arch_ab_gnode" -ManifestPath $GnodeManifest `
            -SpeciesCkptPath $GnodeCkpt -BetaCkptPath $GnodeBeta -Arch gnode
        Eval-ArchLeg -Arch gnode -ManifestPath $GnodeManifest `
            -OutJson (Join-Path $GnodeDir "eval_deploy_frozen.json") `
            -Times "53,$($hz.p007.deploy_eval_t)"
    }
}

Write-Host "[NEW] summarize A/B" -ForegroundColor Cyan
Invoke-PythonRcCheck -Label "arch_ab summarize" -PyArgs @(
    "scripts/summarize_biochem_gnn_arch_ab.py",
    "--sage-eval", (Join-Path $SageDir "eval_deploy_frozen.json"),
    "--gnode-eval", (Join-Path $GnodeDir "eval_deploy_frozen.json"),
    "--out", $SummaryOut
)

Write-Host "[OK] run root: $RunRoot" -ForegroundColor Green
Write-Host "[OK] summary: $SummaryOut" -ForegroundColor Green
