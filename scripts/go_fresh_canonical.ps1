# Powershell script to train HemoRGP v7 Biochem GNN legs.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_fresh_canonical.ps1 -Fresh
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_fresh_canonical.ps1 -Fast          # Quick smoke test

param(
    [ValidateSet("all", "WC_v7_fresh_canonical", "WC_v7_clot_phi_mse", "WC_v7_high_precision")]
    [string] $Leg = "all",
    [string] $Anchors = "patient001,patient002,patient003,patient004,patient005,patient006,patient007,patient008,patient010,patient011",
    [switch] $Fresh,
    [switch] $Fast
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

# Default to 3h training budget per leg (epochs=60, early_stop=15)
$Epochs = 60
$EarlyStop = 15
$MaxWindows = 400

if ($Fast) {
    Write-Output "[i] Applying fast smoke test overrides..."
    $Epochs = 2
    $EarlyStop = 2
    $MaxWindows = 4
    $Anchors = "patient007"
}

$Legs = @()
if ($Leg -eq "all") {
    $Legs = @("WC_v7_fresh_canonical", "WC_v7_clot_phi_mse", "WC_v7_high_precision")
} else {
    $Legs = @($Leg)
}

foreach ($l in $Legs) {
    Write-Host "`n==================================================" -ForegroundColor Magenta
    Write-Host "[i] Starting training for Leg: $l" -ForegroundColor Magenta
    Write-Host "==================================================" -ForegroundColor Magenta
    
    $Ckpt = "outputs/biochem/biochem_gnn/$l/species/best.pth"
    $trainArgs = @(
        "-m", "src.training.train_species_pushforward_continuous",
        "--leg", $l,
        "--anchors", $Anchors,
        "--epochs", "$Epochs",
        "--early-stop", "$EarlyStop",
        "--max-windows", "$MaxWindows",
        "--recipe", "mat_growth_simple",
        "--out", $Ckpt
    )
    if ($Fresh) { $trainArgs += "--no-init" }
    
    Invoke-PythonRcCheck -Label "train $l" -PyArgs $trainArgs
    
    Write-Host "[i] Evaluating Leg: $l ..." -ForegroundColor Cyan
    $Ckpt = "outputs/biochem/biochem_gnn/$l/species/best.pth"
    $CompareJson = "outputs/biochem/biochem_gnn/$l/compare.json"
    $evalArgs = @(
        "scripts/eval_mat_growth_simple.py",
        "--ckpt", $Ckpt,
        "--out", $CompareJson,
        "--anchors", $Anchors
    )
    Invoke-PythonRcCheck -Label "eval $l" -PyArgs $evalArgs
}

Write-Host "`n[OK] go_fresh_canonical done." -ForegroundColor Green
