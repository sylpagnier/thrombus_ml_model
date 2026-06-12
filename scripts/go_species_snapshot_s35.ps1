# Phase 6: freeze s34 GNN; finetune global Mat beta for T=53 viscosity calibration.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_species_snapshot_s35.ps1" -Fresh -AllAnchors

param(
    [string] $GnnCkpt = "outputs/biochem/species_snapshot_s34/best.pth",
    [string] $ValAnchor = "patient007",
    [switch] $AllAnchors,
    [string] $Anchors = "",
    [int] $TimeIndex = 53,
    [float] $BetaInit = 1.5,
    [int] $Epochs = 300,
    [float] $Lr = 0.08,
    [float] $GrowthWeight = 12.0,  # upweight clot-growth nodes in log-MSE
    [string] $Out = "outputs/biochem/species_snapshot_s35/beta.pth",
    [switch] $SkipTrain,
    [switch] $Fresh,
    [switch] $VizOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"
$env:SPECIES_VISCOSITY_CALIB = "1"
$env:SPECIES_VISCOSITY_BETA_MIN = if ($env:SPECIES_VISCOSITY_BETA_MIN) { $env:SPECIES_VISCOSITY_BETA_MIN } else { "0.1" }
$env:SPECIES_VISCOSITY_BETA_MAX = if ($env:SPECIES_VISCOSITY_BETA_MAX) { $env:SPECIES_VISCOSITY_BETA_MAX } else { "2.0" }

$outPath = Join-Path $RepoRoot $Out
if ($Fresh -and (Test-Path $outPath)) {
    Remove-Item $outPath -Force
    $jsonSide = Join-Path (Split-Path $outPath) "beta.json"
    if (Test-Path $jsonSide) { Remove-Item $jsonSide -Force }
    $logPath = Join-Path (Split-Path $outPath) "train_log.jsonl"
    if (Test-Path $logPath) { Remove-Item $logPath -Force }
}

if (-not $SkipTrain -and -not $VizOnly) {
    Write-Host "[NEW] Train s35 viscosity beta calibration" -ForegroundColor Cyan
    $trainArgs = @(
        "scripts/train_clot_phi_calibration.py",
        "--gnn-ckpt", $GnnCkpt,
        "--time-index", "$TimeIndex",
        "--beta-init", "$BetaInit",
        "--epochs", "$Epochs",
        "--lr", "$Lr",
        "--growth-weight", "$GrowthWeight",
        "--out", $Out
    )
    if ($AllAnchors) {
        $trainArgs += "--all-anchors"
    } elseif ($Anchors.Trim()) {
        $trainArgs += @("--anchors", $Anchors)
    } else {
        $trainArgs += @("--anchors", "patient001,patient002,patient003,patient004,patient006,patient007")
    }
    Invoke-PythonRcCheck -Label "s35 beta train" -PyArgs $trainArgs
}

if (-not $SkipTrain -or $VizOnly) {
    Write-Host "[NEW] Viz s34 species + s35 mu eval" -ForegroundColor Cyan
    Invoke-PythonRcCheck -Label "species s34 ladder (GNN)" -PyArgs @(
        "scripts/viz_species_gnn_species_ladder.py",
        "--anchor", $ValAnchor,
        "--ckpt", $GnnCkpt
    )
    Invoke-PythonRcCheck -Label "s35 mu eval patient007" -PyArgs @(
        "scripts/eval_species_viscosity_calibration.py",
        "--anchor", $ValAnchor,
        "--gnn-ckpt", $GnnCkpt,
        "--calib", $Out
    )
    if ($AllAnchors -or $Anchors.Trim()) {
        Invoke-PythonRcCheck -Label "s35 mu multi-anchor" -PyArgs @(
            "scripts/eval_species_viscosity_calibration.py",
            "--gnn-ckpt", $GnnCkpt,
            "--calib", $Out,
            "--all-anchors"
        )
    }
}

Write-Host "[OK] calib=$Out gnn=$GnnCkpt" -ForegroundColor Green
