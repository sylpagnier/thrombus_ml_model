# Phase 5: adaptive temporal gate (global species mass -> lambda in [0.5, 1.5]).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_species_snapshot_s34.ps1" -Fresh -AllAnchors

param(
    [string] $ValAnchor = "patient007",
    [switch] $AllAnchors,
    [string] $Anchors = "",
    [int] $Unroll = 10,
    [int] $Epochs = 50,
    [string] $Ckpt = "outputs/biochem/species_snapshot_s34/best.pth",
    [string] $InitS33 = "outputs/biochem/species_snapshot_s33/best.pth",
    [switch] $SkipTrain,
    [switch] $Fresh,
    [switch] $VizOnly,
    [int] $EarlyStop = 35
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

# s33 recipe + Phase 5 temporal calibration
$env:SPECIES_CONTINUOUS_GROWTH_ONLY_LOSS = "1"
$env:SPECIES_CONTINUOUS_DUAL_HEAD = "1"
$env:SPECIES_CONTINUOUS_PHYSICS_READOUT = "0"
$env:SPECIES_KIN_PER_VESSEL_NORM = "1"
$env:SPECIES_CONTINUOUS_SATURATION_GATE = "1"
$env:SPECIES_CONTINUOUS_MATURE_FP_EXEMPT = "1"
$env:SPECIES_CONTINUOUS_MATURE_FRAC = if ($env:SPECIES_CONTINUOUS_MATURE_FRAC) { $env:SPECIES_CONTINUOUS_MATURE_FRAC } else { "0.95" }
$env:SPECIES_CONTINUOUS_SATURATION_SCALE = if ($env:SPECIES_CONTINUOUS_SATURATION_SCALE) { $env:SPECIES_CONTINUOUS_SATURATION_SCALE } else { "80" }
$env:SPECIES_CONTINUOUS_TEMPORAL_GATE = "1"
$env:SPECIES_CONTINUOUS_TEMPORAL_LAMBDA_MIN = if ($env:SPECIES_CONTINUOUS_TEMPORAL_LAMBDA_MIN) { $env:SPECIES_CONTINUOUS_TEMPORAL_LAMBDA_MIN } else { "0.5" }
$env:SPECIES_CONTINUOUS_TEMPORAL_LAMBDA_MAX = if ($env:SPECIES_CONTINUOUS_TEMPORAL_LAMBDA_MAX) { $env:SPECIES_CONTINUOUS_TEMPORAL_LAMBDA_MAX } else { "1.5" }
$env:SPECIES_CONTINUOUS_VEL_DECAY = "1"
$env:SPECIES_CONTINUOUS_TEACHER_NOISE = if ($env:SPECIES_CONTINUOUS_TEACHER_NOISE) { $env:SPECIES_CONTINUOUS_TEACHER_NOISE } else { "0.02" }
$env:SPECIES_CONTINUOUS_TEACHER_FP_FRAC = if ($env:SPECIES_CONTINUOUS_TEACHER_FP_FRAC) { $env:SPECIES_CONTINUOUS_TEACHER_FP_FRAC } else { "0.08" }
$env:SPECIES_CONTINUOUS_TEACHER_BLUR = if ($env:SPECIES_CONTINUOUS_TEACHER_BLUR) { $env:SPECIES_CONTINUOUS_TEACHER_BLUR } else { "0.25" }
$env:SPECIES_CONTINUOUS_TBPTT_TAIL = if ($env:SPECIES_CONTINUOUS_TBPTT_TAIL) { $env:SPECIES_CONTINUOUS_TBPTT_TAIL } else { "5" }
$env:SPECIES_CONTINUOUS_CLOSED_LOOP_INIT = if ($env:SPECIES_CONTINUOUS_CLOSED_LOOP_INIT) { $env:SPECIES_CONTINUOUS_CLOSED_LOOP_INIT } else { "0.45" }
$env:SPECIES_CONTINUOUS_FINAL_STATE_WEIGHT = if ($env:SPECIES_CONTINUOUS_FINAL_STATE_WEIGHT) { $env:SPECIES_CONTINUOUS_FINAL_STATE_WEIGHT } else { "0.35" }
$env:SPECIES_CONTINUOUS_FINAL_STATE_ALL_BAND = "1"
$env:SPECIES_CONTINUOUS_CURRICULUM_UNROLL = "1"
$env:SPECIES_PUSHFORWARD_UNROLL = "$Unroll"
$env:SPECIES_PUSHFORWARD_MAX_UNROLL = if ($env:SPECIES_PUSHFORWARD_MAX_UNROLL) { $env:SPECIES_PUSHFORWARD_MAX_UNROLL } else { "53" }
$env:SPECIES_CONTINUOUS_DEPLOY_HORIZON = if ($env:SPECIES_CONTINUOUS_DEPLOY_HORIZON) { $env:SPECIES_CONTINUOUS_DEPLOY_HORIZON } else { "53" }
$env:SPECIES_CONTINUOUS_SPEED_FP_WEIGHT = if ($env:SPECIES_CONTINUOUS_SPEED_FP_WEIGHT) { $env:SPECIES_CONTINUOUS_SPEED_FP_WEIGHT } else { "4.0" }
$env:SPECIES_CONTINUOUS_HUBER_BETA = if ($env:SPECIES_CONTINUOUS_HUBER_BETA) { $env:SPECIES_CONTINUOUS_HUBER_BETA } else { "0.5" }
$env:SPECIES_CONTINUOUS_CHANNEL_WEIGHT_MAT = if ($env:SPECIES_CONTINUOUS_CHANNEL_WEIGHT_MAT) { $env:SPECIES_CONTINUOUS_CHANNEL_WEIGHT_MAT } else { "4.0" }
$env:SPECIES_CONTINUOUS_DELTA_VALUE_SCALE = if ($env:SPECIES_CONTINUOUS_DELTA_VALUE_SCALE) { $env:SPECIES_CONTINUOUS_DELTA_VALUE_SCALE } else { "150000" }
$env:SPECIES_CONTINUOUS_DELTA_THRESH = if ($env:SPECIES_CONTINUOUS_DELTA_THRESH) { $env:SPECIES_CONTINUOUS_DELTA_THRESH } else { "5e-6" }
$env:SPECIES_CONTINUOUS_FP_WEIGHT = if ($env:SPECIES_CONTINUOUS_FP_WEIGHT) { $env:SPECIES_CONTINUOUS_FP_WEIGHT } else { "8" }
$env:SPECIES_CONTINUOUS_FP_THRESH = if ($env:SPECIES_CONTINUOUS_FP_THRESH) { $env:SPECIES_CONTINUOUS_FP_THRESH } else { "2e-5" }
$env:SPECIES_CONTINUOUS_SPATIAL_LOSS_WEIGHT = if ($env:SPECIES_CONTINUOUS_SPATIAL_LOSS_WEIGHT) { $env:SPECIES_CONTINUOUS_SPATIAL_LOSS_WEIGHT } else { "1.0" }
$env:SPECIES_PUSHFORWARD_TRAIN_T0_MAX = if ($env:SPECIES_PUSHFORWARD_TRAIN_T0_MAX) { $env:SPECIES_PUSHFORWARD_TRAIN_T0_MAX } else { "35" }
Remove-Item Env:SPECIES_PUSHFORWARD_TRAIN_T0_MIN -ErrorAction SilentlyContinue

$ckptPath = Join-Path $RepoRoot $Ckpt
if ($Fresh -and (Test-Path $ckptPath)) {
    Remove-Item $ckptPath -Force
    $jsonSide = Join-Path (Split-Path $ckptPath) "best.json"
    if (Test-Path $jsonSide) { Remove-Item $jsonSide -Force }
    $logPath = Join-Path (Split-Path $ckptPath) "train_log.jsonl"
    if (Test-Path $logPath) { Remove-Item $logPath -Force }
}

if (-not $SkipTrain -and -not $VizOnly) {
    Write-Host "[NEW] Train species s34 (temporal gate)" -ForegroundColor Cyan
    $trainArgs = @(
        "-m", "src.training.train_species_pushforward_continuous",
        "--phase", "s34",
        "--val-anchor", $ValAnchor,
        "--epochs", "$Epochs",
        "--unroll", "$Unroll",
        "--early-stop", "$EarlyStop",
        "--init-s26", $InitS33,
        "--out", $Ckpt
    )
    if ($AllAnchors) {
        $trainArgs += "--all-anchors"
    } elseif ($Anchors.Trim()) {
        $trainArgs += @("--anchors", $Anchors)
    } else {
        $trainArgs += @("--anchors", "patient001,patient002,patient003,patient004,patient006,patient007")
    }
    Invoke-PythonRcCheck -Label "species s34 train" -PyArgs $trainArgs
}

if (-not $SkipTrain -or $VizOnly) {
    Write-Host "[NEW] Viz s34 species ladder" -ForegroundColor Cyan
    Invoke-PythonRcCheck -Label "species s34 species ladder" -PyArgs @(
        "scripts/viz_species_gnn_species_ladder.py",
        "--anchor", $ValAnchor,
        "--ckpt", $Ckpt
    )
    Invoke-PythonRcCheck -Label "species s34 multi-anchor eval" -PyArgs @(
        "scripts/eval_species_gnn_multi_anchor.py", "--ckpt", $Ckpt
    )
    Invoke-PythonRcCheck -Label "species s34 clot ladder" -PyArgs @(
        "scripts/viz_species_gnn_clot_ladder.py",
        "--anchor", $ValAnchor,
        "--ckpt", $Ckpt
    )
}

Write-Host "[OK] ckpt=$Ckpt" -ForegroundColor Green
