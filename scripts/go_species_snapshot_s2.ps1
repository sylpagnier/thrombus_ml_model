# Phase 2 species pushforward: growth residual + 5-step unroll training + viz.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_species_snapshot_s2.ps1" -Fresh
#   powershell ... -Fresh -Anchor patient007 -Unroll 5 -Epochs 120

param(
    [string] $Anchor = "patient007",
    [int] $Unroll = 5,
    [int] $Stride = 1,
    [int] $Epochs = 120,
    [int] $T0 = 28,
    [string] $Ckpt = "outputs/biochem/species_snapshot_s2/best.pth",
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
$env:CUDA_VISIBLE_DEVICES = if ($env:CUDA_VISIBLE_DEVICES) { $env:CUDA_VISIBLE_DEVICES } else { "0" }
$env:SPECIES_PUSHFORWARD_UNROLL = "$Unroll"
$env:SPECIES_PUSHFORWARD_STEP_STRIDE = "$Stride"
$env:SPECIES_PUSHFORWARD_CKPT = $Ckpt
$env:SPECIES_PUSHFORWARD_FOCAL_ALPHA_FI = if ($env:SPECIES_PUSHFORWARD_FOCAL_ALPHA_FI) { $env:SPECIES_PUSHFORWARD_FOCAL_ALPHA_FI } else { "0.98" }
$env:SPECIES_PUSHFORWARD_FOCAL_ALPHA_MAT = if ($env:SPECIES_PUSHFORWARD_FOCAL_ALPHA_MAT) { $env:SPECIES_PUSHFORWARD_FOCAL_ALPHA_MAT } else { "0.98" }
$env:SPECIES_PUSHFORWARD_CHANNEL_WEIGHT_MAT = if ($env:SPECIES_PUSHFORWARD_CHANNEL_WEIGHT_MAT) { $env:SPECIES_PUSHFORWARD_CHANNEL_WEIGHT_MAT } else { "2.5" }
$env:SPECIES_PUSHFORWARD_GROWTH_THRESH_MAT = if ($env:SPECIES_PUSHFORWARD_GROWTH_THRESH_MAT) { $env:SPECIES_PUSHFORWARD_GROWTH_THRESH_MAT } else { "0.68" }
$env:SPECIES_PUSHFORWARD_TRAIN_T0_MAX = if ($env:SPECIES_PUSHFORWARD_TRAIN_T0_MAX) { $env:SPECIES_PUSHFORWARD_TRAIN_T0_MAX } else { "22" }
$env:SPECIES_PUSHFORWARD_INPUT_NOISE = if ($env:SPECIES_PUSHFORWARD_INPUT_NOISE) { $env:SPECIES_PUSHFORWARD_INPUT_NOISE } else { "0.02" }
$env:SPECIES_PUSHFORWARD_STEP_LOSS = if ($env:SPECIES_PUSHFORWARD_STEP_LOSS) { $env:SPECIES_PUSHFORWARD_STEP_LOSS } else { "linear" }
$env:SPECIES_PUSHFORWARD_SCORE_GROWTH_W = if ($env:SPECIES_PUSHFORWARD_SCORE_GROWTH_W) { $env:SPECIES_PUSHFORWARD_SCORE_GROWTH_W } else { "0.75" }

$ckptPath = Join-Path $RepoRoot $Ckpt
if ($Fresh -and (Test-Path $ckptPath)) {
    Remove-Item $ckptPath -Force
    $jsonSide = Join-Path (Split-Path $ckptPath) "best.json"
    if (Test-Path $jsonSide) { Remove-Item $jsonSide -Force }
    $logPath = Join-Path (Split-Path $ckptPath) "train_log.jsonl"
    if (Test-Path $logPath) { Remove-Item $logPath -Force }
}

if (-not $SkipTrain -and -not $VizOnly) {
    if (-not (Test-Path $ckptPath)) {
        Write-Host "[NEW] Train species pushforward s2 anchor=$Anchor unroll=$Unroll" -ForegroundColor Cyan
        $trainArgs = @(
            "-m", "src.training.train_species_snapshot_pushforward",
            "--anchor", $Anchor,
            "--unroll", "$Unroll",
            "--stride", "$Stride",
            "--epochs", "$Epochs",
            "--out", $Ckpt
        )
        Invoke-PythonRcCheck -Label "species pushforward s2 train" -PyArgs $trainArgs
    } else {
        Write-Host "[skip] checkpoint exists: $Ckpt (use -Fresh to retrain)" -ForegroundColor Yellow
    }
}

Write-Host "[NEW] Viz species pushforward s2 ($Anchor)" -ForegroundColor Cyan
$vizArgs = @(
    "scripts/viz_species_snapshot_s2.py",
    "--anchor", $Anchor,
    "--t0", "$T0",
    "--ckpt", $Ckpt
)
Invoke-PythonRcCheck -Label "species pushforward s2 viz" -PyArgs $vizArgs

Write-Host "[NEW] Viz species pushforward s2 timeline ($Anchor)" -ForegroundColor Cyan
$timelineArgs = @(
    "scripts/viz_species_snapshot_s2_timeline.py",
    "--anchor", $Anchor,
    "--t0", "$T0",
    "--ckpt", $Ckpt
)
Invoke-PythonRcCheck -Label "species pushforward s2 timeline viz" -PyArgs $timelineArgs

Write-Host "[NEW] Viz species pushforward s2 growth timeline ($Anchor)" -ForegroundColor Cyan
$growthArgs = @(
    "scripts/viz_species_snapshot_s2_growth_timeline.py",
    "--anchor", $Anchor,
    "--t0", "10",
    "--ckpt", $Ckpt
)
Invoke-PythonRcCheck -Label "species pushforward s2 growth timeline viz" -PyArgs $growthArgs

$growthPlateauArgs = @(
    "scripts/viz_species_snapshot_s2_growth_timeline.py",
    "--anchor", $Anchor,
    "--t0", "$T0",
    "--ckpt", $Ckpt,
    "--out", "outputs/biochem/viz/species_gnn/s2_${Anchor}_growth_t${T0}.png"
)
Invoke-PythonRcCheck -Label "species pushforward s2 growth plateau viz" -PyArgs $growthPlateauArgs

if ($ClotViz) {
    Write-Host "[NEW] Clot ladder viz (GT | s0 | s2)" -ForegroundColor Cyan
    $clotArgs = @(
        "scripts/viz_species_gnn_clot_ladder.py",
        "--anchor", $Anchor,
        "--ckpt", $Ckpt
    )
    Invoke-PythonRcCheck -Label "species s2 clot ladder" -PyArgs $clotArgs
}

Write-Host "[OK] ckpt=$Ckpt" -ForegroundColor Green
Write-Host "[OK] viz=outputs/biochem/viz/species_gnn/s2_${Anchor}.png" -ForegroundColor Green
$timelineActiveArgs = @(
    "scripts/viz_species_snapshot_s2_timeline.py",
    "--anchor", $Anchor,
    "--t0", "10",
    "--ckpt", $Ckpt
)
Invoke-PythonRcCheck -Label "species pushforward s2 timeline active viz" -PyArgs $timelineActiveArgs

Write-Host "[OK] timeline_active=outputs/biochem/viz/species_gnn/s2_${Anchor}_timeline_t10.png" -ForegroundColor Green
Write-Host "[OK] timeline_plateau=outputs/biochem/viz/species_gnn/s2_${Anchor}_timeline_t${T0}.png" -ForegroundColor Green
Write-Host "[OK] growth_active=outputs/biochem/viz/species_gnn/s2_${Anchor}_growth_t10.png" -ForegroundColor Green
Write-Host "[OK] growth_plateau=outputs/biochem/viz/species_gnn/s2_${Anchor}_growth_t${T0}.png" -ForegroundColor Green
