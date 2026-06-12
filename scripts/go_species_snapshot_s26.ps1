# Phase 2.6: growth-only Huber on ceiling band (fixes zero-delta collapse).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_species_snapshot_s26.ps1" -Fresh
#   powershell ... -SweepQuick
#   powershell ... -VizOnly -ClotViz

param(
    [string] $Anchor = "patient007",
    [int] $Unroll = 5,
    [int] $Epochs = 120,
    [int] $T0Active = 10,
    [int] $T0Plateau = 28,
    [string] $Ckpt = "outputs/biochem/species_snapshot_s26/best.pth",
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
$env:SPECIES_CONTINUOUS_GROWTH_ONLY_LOSS = "1"

# s26 defaults (tune via sweep)
$env:SPECIES_CONTINUOUS_HUBER_BETA = if ($env:SPECIES_CONTINUOUS_HUBER_BETA) { $env:SPECIES_CONTINUOUS_HUBER_BETA } else { "0.5" }
$env:SPECIES_CONTINUOUS_CHANNEL_WEIGHT_MAT = if ($env:SPECIES_CONTINUOUS_CHANNEL_WEIGHT_MAT) { $env:SPECIES_CONTINUOUS_CHANNEL_WEIGHT_MAT } else { "4.0" }
$env:SPECIES_CONTINUOUS_LOSS_SCALE = if ($env:SPECIES_CONTINUOUS_LOSS_SCALE) { $env:SPECIES_CONTINUOUS_LOSS_SCALE } else { "1" }
$env:SPECIES_CONTINUOUS_DELTA_VALUE_SCALE = if ($env:SPECIES_CONTINUOUS_DELTA_VALUE_SCALE) { $env:SPECIES_CONTINUOUS_DELTA_VALUE_SCALE } else { "150000" }
$env:SPECIES_CONTINUOUS_FP_WEIGHT = if ($env:SPECIES_CONTINUOUS_FP_WEIGHT) { $env:SPECIES_CONTINUOUS_FP_WEIGHT } else { "8" }
$env:SPECIES_CONTINUOUS_FP_THRESH = if ($env:SPECIES_CONTINUOUS_FP_THRESH) { $env:SPECIES_CONTINUOUS_FP_THRESH } else { "2e-5" }
$env:SPECIES_CONTINUOUS_DELTA_THRESH = if ($env:SPECIES_CONTINUOUS_DELTA_THRESH) { $env:SPECIES_CONTINUOUS_DELTA_THRESH } else { "1e-5" }
$env:SPECIES_CONTINUOUS_DELTA_THRESH_FI = if ($env:SPECIES_CONTINUOUS_DELTA_THRESH_FI) { $env:SPECIES_CONTINUOUS_DELTA_THRESH_FI } else { "1e-5" }
$env:SPECIES_CONTINUOUS_DELTA_THRESH_MAT = if ($env:SPECIES_CONTINUOUS_DELTA_THRESH_MAT) { $env:SPECIES_CONTINUOUS_DELTA_THRESH_MAT } else { "5e-6" }
$env:SPECIES_CONTINUOUS_UNDERPRED_WEIGHT = if ($env:SPECIES_CONTINUOUS_UNDERPRED_WEIGHT) { $env:SPECIES_CONTINUOUS_UNDERPRED_WEIGHT } else { "0" }
$env:SPECIES_CONTINUOUS_FINAL_STATE_WEIGHT = if ($env:SPECIES_CONTINUOUS_FINAL_STATE_WEIGHT) { $env:SPECIES_CONTINUOUS_FINAL_STATE_WEIGHT } else { "0" }
$env:SPECIES_CONTINUOUS_MAX_SAT_LOG_MAT = if ($env:SPECIES_CONTINUOUS_MAX_SAT_LOG_MAT) { $env:SPECIES_CONTINUOUS_MAX_SAT_LOG_MAT } else { "0.003" }
$env:SPECIES_PUSHFORWARD_TRAIN_T0_MAX = if ($env:SPECIES_PUSHFORWARD_TRAIN_T0_MAX) { $env:SPECIES_PUSHFORWARD_TRAIN_T0_MAX } else { "22" }
$env:SPECIES_PUSHFORWARD_INPUT_NOISE = if ($env:SPECIES_PUSHFORWARD_INPUT_NOISE) { $env:SPECIES_PUSHFORWARD_INPUT_NOISE } else { "0.02" }
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
    $sweepArgs = @("scripts/sweep_species_pushforward_s26_tune.py")
    if ($SweepQuick) { $sweepArgs += "--quick" }
    Invoke-PythonRcCheck -Label "species s26 sweep" -PyArgs $sweepArgs
}

if (-not $SkipTrain -and -not $VizOnly -and -not $Sweep -and -not $SweepQuick) {
    if (-not (Test-Path $ckptPath)) {
        Write-Host "[NEW] Train species continuous s26 anchor=$Anchor" -ForegroundColor Cyan
        $trainArgs = @(
            "-m", "src.training.train_species_pushforward_continuous",
            "--phase", "s26",
            "--anchor", $Anchor,
            "--unroll", "$Unroll",
            "--epochs", "$Epochs",
            "--out", $Ckpt
        )
        Invoke-PythonRcCheck -Label "species s26 train" -PyArgs $trainArgs
    } else {
        Write-Host "[skip] checkpoint exists: $Ckpt (use -Fresh)" -ForegroundColor Yellow
    }
}

if (-not $Sweep -and -not $SweepQuick) {
    Write-Host "[NEW] Viz species ladder (clot time grid)" -ForegroundColor Cyan
    Invoke-PythonRcCheck -Label "species s26 species ladder" -PyArgs @(
        "scripts/viz_species_gnn_species_ladder.py",
        "--anchor", $Anchor,
        "--ckpt", $Ckpt
    )
    if ($ClotViz) {
        Write-Host "[NEW] Clot ladder viz (GT | s0 | GNN)" -ForegroundColor Cyan
        Invoke-PythonRcCheck -Label "species s26 clot ladder" -PyArgs @(
            "scripts/viz_species_gnn_clot_ladder.py",
            "--anchor", $Anchor,
            "--ckpt", $Ckpt
        )
    }
}

Write-Host "[OK] ckpt=$Ckpt" -ForegroundColor Green
