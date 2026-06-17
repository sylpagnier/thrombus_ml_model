# Gate A/B: global sigmoid vs Fourier-tau vs spatial MLP adhesion gate.
#
# Arm A: BIOCHEM_ADHESION_GATE=global_sigmoid  (current baseline)
# Arm B: BIOCHEM_ADHESION_GATE=fourier_tau     (sinusoidal tau encoding per node)
# Arm C: BIOCHEM_ADHESION_GATE=spatial_mlp     (node-level MLP incubation time)
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_biochem_gnn_gate_ab.ps1
#   powershell ... -Fresh
#   powershell ... -Legs "A,B"       # run subset of legs
#   powershell ... -Epochs 50 -Smoke # quick sanity (8 ep)
#   powershell ... -SkipTrain        # eval-only from existing checkpoints

param(
    [int]    $Epochs    = 60,
    [int]    $EarlyStop = 20,
    [double] $Lr        = 1.5e-4,
    [string] $Legs      = "A,B,C",  # comma-separated: A | B | C
    [switch] $Fresh,
    [switch] $SkipTrain,
    [switch] $Smoke
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED   = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

if ($Smoke) {
    $Epochs    = 8
    $EarlyStop = 4
    Write-Host "[i] smoke mode: $Epochs ep" -ForegroundColor DarkGray
}

$RunRoot = Join-Path $RepoRoot "outputs/biochem/biochem_gnn/gate_ab"
$SummaryOut = Join-Path $RunRoot "gate_ab_summary.json"

# Discover shared warm-start (try latest strong checkpoint first)
$InitWarm = "outputs/biochem/biochem_gnn/global_guiding_5h/species/best.pth"
foreach ($cand in @(
    "outputs/biochem/biochem_gnn/global_fulltime/species/best.pth",
    "outputs/biochem/biochem_gnn/arch_ab/sage/species/best.pth",
    "outputs/biochem/species_gnn_deploy_baseline/species_gnn_best.pth"
)) {
    if (-not (Test-Path (Join-Path $RepoRoot $InitWarm))) {
        if (Test-Path (Join-Path $RepoRoot $cand)) { $InitWarm = $cand }
    }
}
Write-Host "[i] warm-start: $InitWarm" -ForegroundColor DarkGray

# Shared beta (skip full calibration to avoid 4GB OOM)
$SharedBeta = "outputs/biochem/biochem_gnn/global_guiding_5h/viscosity/beta.pth"
foreach ($cand in @(
    "outputs/biochem/biochem_gnn/arch_ab/sage/viscosity/beta.pth",
    "outputs/biochem/species_gnn_deploy_baseline/viscosity_beta.pth"
)) {
    if (-not (Test-Path (Join-Path $RepoRoot $SharedBeta))) {
        if (Test-Path (Join-Path $RepoRoot $cand)) { $SharedBeta = $cand }
    }
}
Write-Host "[i] shared beta: $SharedBeta" -ForegroundColor DarkGray

# Discover all anchor patients
$AnchorDir = Join-Path $RepoRoot "outputs/biochem"
$AnchorPaths = @(Get-ChildItem -Path $AnchorDir -Filter "biochem_anchor_patient*.pt" -File |
    Select-Object -ExpandProperty FullName | Sort-Object)
if ($AnchorPaths.Count -eq 0) {
    $AnchorPaths = @()
}
Write-Host "[i] anchors found: $($AnchorPaths.Count)" -ForegroundColor DarkGray

# Deploy time for p007
$p007DeployTime = "10000"
$p007TsFile = Join-Path $RepoRoot "outputs/biochem/patient007_deploy_t_s.txt"
if (Test-Path $p007TsFile) {
    $p007DeployTime = (Get-Content $p007TsFile -Raw).Trim()
}
Write-Host "[i] p007 deploy t_s: $p007DeployTime" -ForegroundColor DarkGray

# -------------------------------------------------------------------
# Leg definitions: key -> (gate_mode, dir_suffix)
# -------------------------------------------------------------------
$LegDefs = @{
    "A" = @{ Mode = "global_sigmoid"; Label = "baseline_sigmoid" }
    "B" = @{ Mode = "fourier_tau";    Label = "fourier_tau" }
    "C" = @{ Mode = "spatial_mlp";    Label = "spatial_mlp" }
}

$RequestedLegs = $Legs -split "," | ForEach-Object { $_.Trim().ToUpper() }

# -------------------------------------------------------------------
function Set-GateAbTrainEnv {
    param([string]$GateMode, [int]$Ep, [int]$ES, [double]$LR)
    $env:BIOCHEM_ADHESION_GATE               = $GateMode
    $env:BIOCHEM_PUSHFORWARD_SPECIES_SCOPE  = "fi_mat"
    $env:SPECIES_PUSHFORWARD_ARCH            = "sage"   # keep arch fixed; only gate changes
    $env:SPECIES_CONTINUOUS_DEPLOY_EVAL_FULL = "1"
    $env:BIOCHEM_GELATION_PRIOR_GATE         = "1"
    $env:BIOCHEM_DETACH_MACRO_STATE          = "1"
    $env:BIOCHEM_ODE_GRADIENT_CHECKPOINT     = "1"
    $env:BIOCHEM_TRAINING_LOG                = "1"
    $env:TRAIN_EPOCHS                        = "$Ep"
    $env:TRAIN_EARLY_STOP_PATIENCE           = "$ES"
    $env:TRAIN_LR                            = "$LR"
    $env:SPECIES_TRAIN_VEL_SOURCE            = "gt"
    $env:SPECIES_ROLLOUT_DEPLOY_FAITHFUL     = "1"
    $env:SPECIES_ROLLOUT_VEL_SOURCE          = "kinematics"
    $env:SPECIES_ROLLOUT_PIN_OTHER           = "rest"
    $env:SPECIES_ROLLOUT_IC_SOURCE           = "resting"
}

function Train-GateLeg {
    param([string]$LegKey, [string]$GateMode, [string]$LegDir)
    $speciesDir  = Join-Path $LegDir "species"
    $viscDir     = Join-Path $LegDir "viscosity"
    $legBest     = Join-Path $speciesDir "best.pth"
    New-Item -ItemType Directory -Force -Path $speciesDir | Out-Null
    New-Item -ItemType Directory -Force -Path $viscDir    | Out-Null

    # Copy shared beta to avoid OOM during calibration
    $legBeta = Join-Path $viscDir "beta.pth"
    if (-not (Test-Path $legBeta)) {
        if (Test-Path (Join-Path $RepoRoot $SharedBeta)) {
            Copy-Item (Join-Path $RepoRoot $SharedBeta) $legBeta -Force
            Write-Host "[i] [$LegKey] copied shared beta -> $legBeta" -ForegroundColor DarkGray
        }
    }

    # Resolve init checkpoint
    $legInit = ""
    if (-not $Fresh -and (Test-Path $legBest)) {
        $legInit = $legBest
        Write-Host "[i] [$LegKey] resuming from $legBest" -ForegroundColor DarkGray
    } elseif (Test-Path (Join-Path $RepoRoot $InitWarm)) {
        $legInit = Join-Path $RepoRoot $InitWarm
        Write-Host "[i] [$LegKey] warm-starting from $legInit" -ForegroundColor DarkGray
    }

    Set-GateAbTrainEnv -GateMode $GateMode -Ep $Epochs -ES $EarlyStop -LR $Lr
    $env:BIOCHEM_GNN_TRAIN_OUTPUT_DIR = $speciesDir

    $cmd = @(
        "-m", "src.training.train_biochem_gnn",
        "--step",  "species",
        "--arch",  "sage",
        "--epochs", "$Epochs",
        "--lr",    "$Lr",
        "--species-out", (Join-Path $speciesDir "best.pth")
    )
    if ($legInit -ne "") { $cmd += @("--init", $legInit) }

    Write-Host "[run] [$LegKey] gate=$GateMode training ($Epochs ep)" -ForegroundColor Cyan
    Invoke-PythonRcCheck -Label "gate_ab train [$LegKey]" -PyArgs $cmd
    Write-Host "[OK] [$LegKey] training done" -ForegroundColor Green
}

function Write-GateManifest {
    param([string]$LegKey, [string]$GateMode, [string]$LegDir)
    $speciesDir  = Join-Path $LegDir "species"
    $viscDir     = Join-Path $LegDir "viscosity"
    $legBest     = Join-Path $speciesDir "best.pth"
    $legBeta     = Join-Path $viscDir "beta.pth"

    if (-not (Test-Path $legBest)) {
        Write-Host "[WARN] [$LegKey] no checkpoint at $legBest; skipping manifest" -ForegroundColor Yellow
        return ""
    }

    $mfPath = Join-Path $RepoRoot "data/reference/biochem_gnn_gate_ab_$($LegKey.ToLower()).json"
    $betaPath = if (Test-Path $legBeta) { $legBeta } else { Join-Path $RepoRoot $SharedBeta }
    $null = Invoke-PythonRc -Quiet @("-c", @"
import json
from pathlib import Path
from src.biochem_gnn.config import rel_path
m = {
    'name': 'biochem_gnn_gate_ab_$($LegKey.ToLower())',
    'version': 1,
    'baseline': {
        'species_gnn_ckpt': rel_path(Path(r'$legBest')),
        'viscosity_beta': rel_path(Path(r'$betaPath')),
        'kinematics_ckpt': 'outputs/kinematics/kinematics_best.pth',
        'train_val_anchor': 'patient007',
        'flow_modes': 'kinematics',
        'gamma_mode': 'max',
        'deploy_horizon': 'full',
        'clot_score': 'guiding',
        'pushforward_arch': 'sage',
        'gate_mode': '$GateMode',
    },
}
p = Path(r'$mfPath')
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(m, indent=2), encoding='utf-8')
"@)
    Write-Host "[i] [$LegKey] manifest -> $mfPath" -ForegroundColor DarkGray
    return $mfPath
}

function Eval-GateLeg {
    param([string]$LegKey, [string]$GateMode, [string]$LegDir, [string]$ManifestPath)
    $evalDir = Join-Path $LegDir "eval"
    New-Item -ItemType Directory -Force -Path $evalDir | Out-Null
    $evalOut = Join-Path $evalDir "deploy_ab_eval.json"
    $times = "53,$p007DeployTime"

    $env:BIOCHEM_ADHESION_GATE = $GateMode

    Write-Host "[run] [$LegKey] eval deploy_frozen (times=$times)" -ForegroundColor Cyan
    Invoke-PythonRcCheck -Label "gate_ab eval [$LegKey]" -PyArgs @(
        "scripts/eval_biochem_gnn_deploy_ab.py",
        "--manifest", $ManifestPath,
        "--modes", "deploy_frozen",
        "--times", $times,
        "--out", $evalOut
    )
    Write-Host "[OK] [$LegKey] eval done -> $evalOut" -ForegroundColor Green
}

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
Write-Host ""
Write-Host "=== Gate A/B: global_sigmoid vs fourier_tau vs spatial_mlp ===" -ForegroundColor White
Write-Host "Requested legs: $RequestedLegs" -ForegroundColor DarkGray

New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

$ManifestPaths = @{}

foreach ($LegKey in $RequestedLegs) {
    if (-not $LegDefs.ContainsKey($LegKey)) {
        Write-Host "[WARN] unknown leg '$LegKey'; skipping" -ForegroundColor Yellow
        continue
    }
    $def      = $LegDefs[$LegKey]
    $gateMode = $def.Mode
    $legDir   = Join-Path $RunRoot $def.Label

    Write-Host ""
    Write-Host "--- Leg $LegKey ($gateMode) ---" -ForegroundColor Yellow

    if (-not $SkipTrain) {
        Train-GateLeg -LegKey $LegKey -GateMode $gateMode -LegDir $legDir
    }

    $mfPath = Write-GateManifest -LegKey $LegKey -GateMode $gateMode -LegDir $legDir
    if ($mfPath -ne "") {
        $ManifestPaths[$LegKey] = $mfPath
        Eval-GateLeg -LegKey $LegKey -GateMode $gateMode -LegDir $legDir -ManifestPath $mfPath
    }
}

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------
Write-Host ""
Write-Host "--- Summarising gate A/B results ---" -ForegroundColor Yellow
$evalDirs = @{}
foreach ($k in $ManifestPaths.Keys) {
    $def    = $LegDefs[$k]
    $legDir = Join-Path $RunRoot $def.Label
    $evalDirs[$k] = Join-Path $legDir "eval"
}

if ($evalDirs.Count -gt 0) {
    $dirsJson = $evalDirs | ConvertTo-Json -Compress
    Invoke-PythonRc `
        "scripts/summarize_biochem_gnn_gate_ab.py",
        "--eval-dirs", $dirsJson,
        "--output", $SummaryOut
    Write-Host "[OK] summary -> $SummaryOut" -ForegroundColor Green
} else {
    Write-Host "[WARN] no eval dirs to summarise" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Gate A/B done ===" -ForegroundColor White
