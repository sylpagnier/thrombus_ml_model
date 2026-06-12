# Phase 3.5: dual-head spatial*magnitude + per-vessel kin norm, pure species loss (no physics).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_species_snapshot_s31.ps1" -Fresh -AllAnchors -ClotViz

param(
    [string] $ValAnchor = "patient007",
    [switch] $AllAnchors,
    [string] $Anchors = "",
    [int] $Unroll = 5,
    [int] $Epochs = 100,
    [string] $Ckpt = "outputs/biochem/species_snapshot_s31/best.pth",
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
$env:SPECIES_CONTINUOUS_DUAL_HEAD = "1"
$env:SPECIES_CONTINUOUS_PHYSICS_READOUT = "0"
$env:SPECIES_KIN_PER_VESSEL_NORM = "1"

# s26 combo + dual-head defaults
$env:SPECIES_CONTINUOUS_HUBER_BETA = if ($env:SPECIES_CONTINUOUS_HUBER_BETA) { $env:SPECIES_CONTINUOUS_HUBER_BETA } else { "0.5" }
$env:SPECIES_CONTINUOUS_CHANNEL_WEIGHT_MAT = if ($env:SPECIES_CONTINUOUS_CHANNEL_WEIGHT_MAT) { $env:SPECIES_CONTINUOUS_CHANNEL_WEIGHT_MAT } else { "4.0" }
$env:SPECIES_CONTINUOUS_DELTA_VALUE_SCALE = if ($env:SPECIES_CONTINUOUS_DELTA_VALUE_SCALE) { $env:SPECIES_CONTINUOUS_DELTA_VALUE_SCALE } else { "150000" }
$env:SPECIES_CONTINUOUS_DELTA_THRESH = if ($env:SPECIES_CONTINUOUS_DELTA_THRESH) { $env:SPECIES_CONTINUOUS_DELTA_THRESH } else { "5e-6" }
$env:SPECIES_CONTINUOUS_FP_WEIGHT = if ($env:SPECIES_CONTINUOUS_FP_WEIGHT) { $env:SPECIES_CONTINUOUS_FP_WEIGHT } else { "8" }
$env:SPECIES_CONTINUOUS_FP_THRESH = if ($env:SPECIES_CONTINUOUS_FP_THRESH) { $env:SPECIES_CONTINUOUS_FP_THRESH } else { "2e-5" }
$env:SPECIES_CONTINUOUS_FINAL_STATE_WEIGHT = if ($env:SPECIES_CONTINUOUS_FINAL_STATE_WEIGHT) { $env:SPECIES_CONTINUOUS_FINAL_STATE_WEIGHT } else { "0" }
$env:SPECIES_CONTINUOUS_SPATIAL_LOSS_WEIGHT = if ($env:SPECIES_CONTINUOUS_SPATIAL_LOSS_WEIGHT) { $env:SPECIES_CONTINUOUS_SPATIAL_LOSS_WEIGHT } else { "1.0" }
$env:SPECIES_PUSHFORWARD_TRAIN_T0_MAX = if ($env:SPECIES_PUSHFORWARD_TRAIN_T0_MAX) { $env:SPECIES_PUSHFORWARD_TRAIN_T0_MAX } else { "22" }
$env:SPECIES_PUSHFORWARD_UNROLL = "$Unroll"
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
    Write-Host "[NEW] Train species s31 (dual-head + kin norm)" -ForegroundColor Cyan
    $trainArgs = @(
        "-m", "src.training.train_species_pushforward_continuous",
        "--phase", "s31",
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
    Invoke-PythonRcCheck -Label "species s31 train" -PyArgs $trainArgs
}

if (-not $SkipTrain -or $VizOnly) {
    Write-Host "[NEW] Viz s31 species ladder (clot time grid)" -ForegroundColor Cyan
    Invoke-PythonRcCheck -Label "species s31 species ladder" -PyArgs @(
        "scripts/viz_species_gnn_species_ladder.py",
        "--anchor", $ValAnchor,
        "--ckpt", $Ckpt
    )
    Invoke-PythonRcCheck -Label "species s31 multi-anchor eval" -PyArgs @(
        "scripts/eval_species_gnn_multi_anchor.py", "--ckpt", $Ckpt
    )
    Write-Host "[NEW] Clot ladder viz (GT | s0 | s31)" -ForegroundColor Cyan
    Invoke-PythonRcCheck -Label "species s31 clot ladder" -PyArgs @(
        "scripts/viz_species_gnn_clot_ladder.py",
        "--anchor", $ValAnchor,
        "--ckpt", $Ckpt
    )
}

Write-Host "[OK] ckpt=$Ckpt" -ForegroundColor Green
