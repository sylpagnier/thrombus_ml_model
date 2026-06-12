# Phase 2.5 continuous log-delta pushforward + soft-commit memory.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_species_snapshot_s25.ps1" -Fresh
#   powershell ... -SweepQuick

param(
    [string] $Anchor = "patient007",
    [int] $Unroll = 5,
    [int] $Epochs = 120,
    [int] $T0Active = 10,
    [int] $T0Plateau = 28,
    [string] $Ckpt = "outputs/biochem/species_snapshot_s25/best.pth",
    [switch] $SkipTrain,
    [switch] $Fresh,
    [switch] $VizOnly,
    [switch] $SweepQuick,
    [switch] $Sweep,
    [switch] $ClotViz
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

# combo winner defaults (updated after sweep)
$env:SPECIES_CONTINUOUS_HUBER_BETA = if ($env:SPECIES_CONTINUOUS_HUBER_BETA) { $env:SPECIES_CONTINUOUS_HUBER_BETA } else { "1e-4" }
$env:SPECIES_CONTINUOUS_CHANNEL_WEIGHT_MAT = if ($env:SPECIES_CONTINUOUS_CHANNEL_WEIGHT_MAT) { $env:SPECIES_CONTINUOUS_CHANNEL_WEIGHT_MAT } else { "3.0" }
$env:SPECIES_CONTINUOUS_LOSS_SCALE = if ($env:SPECIES_CONTINUOUS_LOSS_SCALE) { $env:SPECIES_CONTINUOUS_LOSS_SCALE } else { "10000" }
$env:SPECIES_CONTINUOUS_FINAL_STATE_WEIGHT = if ($env:SPECIES_CONTINUOUS_FINAL_STATE_WEIGHT) { $env:SPECIES_CONTINUOUS_FINAL_STATE_WEIGHT } else { "0.5" }
$env:SPECIES_CONTINUOUS_MAX_SAT_LOG_MAT = if ($env:SPECIES_CONTINUOUS_MAX_SAT_LOG_MAT) { $env:SPECIES_CONTINUOUS_MAX_SAT_LOG_MAT } else { "0.003" }
$env:SPECIES_PUSHFORWARD_TRAIN_T0_MAX = if ($env:SPECIES_PUSHFORWARD_TRAIN_T0_MAX) { $env:SPECIES_PUSHFORWARD_TRAIN_T0_MAX } else { "22" }
$env:SPECIES_PUSHFORWARD_INPUT_NOISE = if ($env:SPECIES_PUSHFORWARD_INPUT_NOISE) { $env:SPECIES_PUSHFORWARD_INPUT_NOISE } else { "0.015" }
$env:SPECIES_PUSHFORWARD_UNROLL = "$Unroll"
$env:SPECIES_PUSHFORWARD_STEP_LOSS = if ($env:SPECIES_PUSHFORWARD_STEP_LOSS) { $env:SPECIES_PUSHFORWARD_STEP_LOSS } else { "linear" }

$ckptPath = Join-Path $RepoRoot $Ckpt
if ($Fresh -and (Test-Path $ckptPath)) {
    Remove-Item $ckptPath -Force
    $jsonSide = Join-Path (Split-Path $ckptPath) "best.json"
    if (Test-Path $jsonSide) { Remove-Item $jsonSide -Force }
    $logPath = Join-Path (Split-Path $ckptPath) "train_log.jsonl"
    if (Test-Path $logPath) { Remove-Item $logPath -Force }
}

if ($Sweep -or $SweepQuick) {
    $sweepArgs = @("scripts/sweep_species_pushforward_continuous_tune.py")
    if ($SweepQuick) { $sweepArgs += "--quick" }
    Invoke-PythonRcCheck -Label "species s25 sweep" -PyArgs $sweepArgs
}

if (-not $SkipTrain -and -not $VizOnly -and -not $Sweep -and -not $SweepQuick) {
    if (-not (Test-Path $ckptPath)) {
        Write-Host "[NEW] Train species continuous s25 anchor=$Anchor" -ForegroundColor Cyan
        $trainArgs = @(
            "-m", "src.training.train_species_pushforward_continuous",
            "--anchor", $Anchor,
            "--unroll", "$Unroll",
            "--epochs", "$Epochs",
            "--out", $Ckpt
        )
        Invoke-PythonRcCheck -Label "species s25 train" -PyArgs $trainArgs
    } else {
        Write-Host "[skip] checkpoint exists: $Ckpt (use -Fresh)" -ForegroundColor Yellow
    }
}

if (-not $Sweep -and -not $SweepQuick) {
    foreach ($pair in @(@($T0Active, "active"), @($T0Plateau, "plateau"))) {
        $t0 = $pair[0]; $tag = $pair[1]
        Write-Host "[NEW] Viz s25 timeline $tag t0=$t0" -ForegroundColor Cyan
        $vizArgs = @(
            "scripts/viz_species_snapshot_s25_timeline.py",
            "--anchor", $Anchor,
            "--t0", "$t0",
            "--ckpt", $Ckpt
        )
        Invoke-PythonRcCheck -Label "species s25 viz $tag" -PyArgs $vizArgs
    }

    if ($ClotViz) {
        Write-Host "[NEW] Clot ladder viz (GT | s0 | s25)" -ForegroundColor Cyan
        $clotArgs = @(
            "scripts/viz_species_gnn_clot_ladder.py",
            "--anchor", $Anchor,
            "--ckpt", $Ckpt
        )
        Invoke-PythonRcCheck -Label "species s25 clot ladder" -PyArgs $clotArgs
    }
}

Write-Host "[OK] ckpt=$Ckpt" -ForegroundColor Green
