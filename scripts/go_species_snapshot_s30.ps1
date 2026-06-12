# Phase 3: closed-loop gelation readout + tau window + multi-vessel train.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_species_snapshot_s30.ps1" -Fresh
#   powershell ... -AllAnchors -VizOnly -ClotViz

param(
    [string] $ValAnchor = "patient007",
    [switch] $AllAnchors,
    [string] $Anchors = "",
    [int] $Unroll = 5,
    [int] $Epochs = 100,
    [string] $Ckpt = "outputs/biochem/species_snapshot_s30/best.pth",
    [string] $InitS26 = "outputs/biochem/species_snapshot_s26/best.pth",
    [switch] $SkipTrain,
    [switch] $Fresh,
    [switch] $VizOnly,
    [switch] $ClotViz
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"
$env:SPECIES_CONTINUOUS_GROWTH_ONLY_LOSS = "1"
$env:SPECIES_CONTINUOUS_PHYSICS_READOUT = "1"

# s26 combo winner + phase 3 physics/tau defaults
$env:SPECIES_CONTINUOUS_HUBER_BETA = if ($env:SPECIES_CONTINUOUS_HUBER_BETA) { $env:SPECIES_CONTINUOUS_HUBER_BETA } else { "0.5" }
$env:SPECIES_CONTINUOUS_CHANNEL_WEIGHT_MAT = if ($env:SPECIES_CONTINUOUS_CHANNEL_WEIGHT_MAT) { $env:SPECIES_CONTINUOUS_CHANNEL_WEIGHT_MAT } else { "4.0" }
$env:SPECIES_CONTINUOUS_DELTA_VALUE_SCALE = if ($env:SPECIES_CONTINUOUS_DELTA_VALUE_SCALE) { $env:SPECIES_CONTINUOUS_DELTA_VALUE_SCALE } else { "150000" }
$env:SPECIES_CONTINUOUS_DELTA_THRESH = if ($env:SPECIES_CONTINUOUS_DELTA_THRESH) { $env:SPECIES_CONTINUOUS_DELTA_THRESH } else { "5e-6" }
$env:SPECIES_CONTINUOUS_FP_WEIGHT = if ($env:SPECIES_CONTINUOUS_FP_WEIGHT) { $env:SPECIES_CONTINUOUS_FP_WEIGHT } else { "0" }
$env:SPECIES_CONTINUOUS_FP_THRESH = if ($env:SPECIES_CONTINUOUS_FP_THRESH) { $env:SPECIES_CONTINUOUS_FP_THRESH } else { "2e-5" }
$env:SPECIES_CONTINUOUS_UNDERPRED_WEIGHT = if ($env:SPECIES_CONTINUOUS_UNDERPRED_WEIGHT) { $env:SPECIES_CONTINUOUS_UNDERPRED_WEIGHT } else { "0" }
$env:SPECIES_CONTINUOUS_FINAL_STATE_WEIGHT = if ($env:SPECIES_CONTINUOUS_FINAL_STATE_WEIGHT) { $env:SPECIES_CONTINUOUS_FINAL_STATE_WEIGHT } else { "0" }
$env:SPECIES_CONTINUOUS_PHI_LOSS_WEIGHT = if ($env:SPECIES_CONTINUOUS_PHI_LOSS_WEIGHT) { $env:SPECIES_CONTINUOUS_PHI_LOSS_WEIGHT } else { "3.0" }
$env:SPECIES_CONTINUOUS_MU_LOSS_WEIGHT = if ($env:SPECIES_CONTINUOUS_MU_LOSS_WEIGHT) { $env:SPECIES_CONTINUOUS_MU_LOSS_WEIGHT } else { "0.5" }
$env:SPECIES_GELATION_FRONTIER_BOOST = if ($env:SPECIES_GELATION_FRONTIER_BOOST) { $env:SPECIES_GELATION_FRONTIER_BOOST } else { "3.0" }
$env:SPECIES_PUSHFORWARD_TRAIN_T0_MIN = if ($env:SPECIES_PUSHFORWARD_TRAIN_T0_MIN) { $env:SPECIES_PUSHFORWARD_TRAIN_T0_MIN } else { "17" }
$env:SPECIES_PUSHFORWARD_TRAIN_T0_MAX = if ($env:SPECIES_PUSHFORWARD_TRAIN_T0_MAX) { $env:SPECIES_PUSHFORWARD_TRAIN_T0_MAX } else { "32" }
$env:SPECIES_PUSHFORWARD_TAU_CENTER = if ($env:SPECIES_PUSHFORWARD_TAU_CENTER) { $env:SPECIES_PUSHFORWARD_TAU_CENTER } else { "25" }
$env:SPECIES_PUSHFORWARD_TAU_SIGMA = if ($env:SPECIES_PUSHFORWARD_TAU_SIGMA) { $env:SPECIES_PUSHFORWARD_TAU_SIGMA } else { "6" }
$env:SPECIES_PUSHFORWARD_INPUT_NOISE = if ($env:SPECIES_PUSHFORWARD_INPUT_NOISE) { $env:SPECIES_PUSHFORWARD_INPUT_NOISE } else { "0.02" }
$env:SPECIES_PUSHFORWARD_UNROLL = "$Unroll"

$ckptPath = Join-Path $RepoRoot $Ckpt
if ($Fresh -and (Test-Path $ckptPath)) {
    Remove-Item $ckptPath -Force
    $jsonSide = Join-Path (Split-Path $ckptPath) "best.json"
    if (Test-Path $jsonSide) { Remove-Item $jsonSide -Force }
    $logPath = Join-Path (Split-Path $ckptPath) "train_log.jsonl"
    if (Test-Path $logPath) { Remove-Item $logPath -Force }
}

if (-not $SkipTrain -and -not $VizOnly) {
    Write-Host "[NEW] Train species s30 (physics readout)" -ForegroundColor Cyan
    $trainArgs = @(
        "-m", "src.training.train_species_pushforward_continuous",
        "--phase", "s30",
        "--val-anchor", $ValAnchor,
        "--epochs", "$Epochs",
        "--unroll", "$Unroll",
        "--init-s26", $InitS26,
        "--out", $Ckpt
    )
    if ($AllAnchors) {
        $trainArgs += "--all-anchors"
    } elseif ($Anchors.Trim()) {
        $trainArgs += @("--anchors", $Anchors)
    } else {
        $trainArgs += @("--anchors", "patient001,patient002,patient003,patient004,patient006,patient007")
    }
    Invoke-PythonRcCheck -Label "species s30 train" -PyArgs $trainArgs
}

if (-not $SkipTrain -or $VizOnly) {
    Write-Host "[NEW] Viz species ladder (clot time grid)" -ForegroundColor Cyan
    Invoke-PythonRcCheck -Label "species s30 species ladder" -PyArgs @(
        "scripts/viz_species_gnn_species_ladder.py",
        "--anchor", $ValAnchor,
        "--ckpt", $Ckpt
    )
    if ($ClotViz) {
        Write-Host "[NEW] Clot ladder viz (GT | s0 | s30)" -ForegroundColor Cyan
        $clotArgs = @(
            "scripts/viz_species_gnn_clot_ladder.py",
            "--anchor", $ValAnchor,
            "--ckpt", $Ckpt
        )
        Invoke-PythonRcCheck -Label "species s30 clot ladder" -PyArgs $clotArgs
    }
}

Write-Host "[OK] ckpt=$Ckpt" -ForegroundColor Green
