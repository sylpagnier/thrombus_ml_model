# Multi-vessel Phase 2.6 (frozen combo hyperparams, no physics readout).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_species_snapshot_s26_6v.ps1" -Fresh

param(
    [string] $ValAnchor = "patient007",
    [string] $InitS26 = "outputs/biochem/species_snapshot_s26/best.pth",
    [string] $Ckpt = "outputs/biochem/species_snapshot_s26_6v/best.pth",
    [int] $Epochs = 80,
    [switch] $Fresh,
    [switch] $EvalOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"
$env:SPECIES_CONTINUOUS_GROWTH_ONLY_LOSS = "1"
$env:SPECIES_CONTINUOUS_PHYSICS_READOUT = "0"
$env:SPECIES_CONTINUOUS_HUBER_BETA = "0.5"
$env:SPECIES_CONTINUOUS_CHANNEL_WEIGHT_MAT = "4.0"
$env:SPECIES_CONTINUOUS_DELTA_VALUE_SCALE = "150000"
$env:SPECIES_CONTINUOUS_DELTA_THRESH = "5e-6"
$env:SPECIES_CONTINUOUS_FP_WEIGHT = "8"
$env:SPECIES_CONTINUOUS_FP_THRESH = "2e-5"
$env:SPECIES_CONTINUOUS_FINAL_STATE_WEIGHT = "0"
Remove-Item Env:SPECIES_PUSHFORWARD_TRAIN_T0_MIN -ErrorAction SilentlyContinue
$env:SPECIES_PUSHFORWARD_TRAIN_T0_MAX = "22"

$ckptPath = Join-Path $RepoRoot $Ckpt
if ($Fresh -and (Test-Path $ckptPath)) {
    Remove-Item $ckptPath -Force
}

if (-not $EvalOnly) {
    Invoke-PythonRcCheck -Label "species s26 6v train" -PyArgs @(
        "-m", "src.training.train_species_pushforward_continuous",
        "--phase", "s26", "--all-anchors", "--val-anchor", $ValAnchor,
        "--epochs", "$Epochs", "--early-stop", "20", "--unroll", "5",
        "--init-s26", $InitS26, "--out", $Ckpt
    )
}

Invoke-PythonRcCheck -Label "species s26 6v multi-anchor eval" -PyArgs @(
    "scripts/eval_species_gnn_multi_anchor.py", "--ckpt", $Ckpt
)
Write-Host "[OK] ckpt=$Ckpt" -ForegroundColor Green
