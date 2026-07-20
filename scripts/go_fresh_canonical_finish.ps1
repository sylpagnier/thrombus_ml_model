# Train legs 2 and 3, then evaluate all 3 legs at the end.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_fresh_canonical_finish.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_fresh_canonical_finish.ps1 -Fast  # smoke test

param(
    [switch] $Fast
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

$Anchors = "patient001,patient002,patient003,patient004,patient005,patient006,patient007,patient008,patient010,patient011"
$Epochs    = 60
$EarlyStop = 15
$MaxWindows = 400

if ($Fast) {
    Write-Output "[i] Fast smoke-test mode"
    $Epochs     = 2
    $EarlyStop  = 2
    $MaxWindows = 4
    $Anchors    = "patient007"
}

# -----------------------------------------------------------------------
# STEP 1: Train leg 2 and leg 3 (no per-leg eval)
# -----------------------------------------------------------------------
$TrainLegs = @("WC_v7_clot_phi_mse", "WC_v7_high_precision")

foreach ($l in $TrainLegs) {
    Write-Host "`n==================================================" -ForegroundColor Magenta
    Write-Host "[i] Training Leg: $l" -ForegroundColor Magenta
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
    Invoke-PythonRcCheck -Label "train $l" -PyArgs $trainArgs
}

# -----------------------------------------------------------------------
# STEP 2: Evaluate all 3 legs sequentially (single kine model load per leg)
# -----------------------------------------------------------------------
$EvalLegs = @("WC_v7_fresh_canonical", "WC_v7_clot_phi_mse", "WC_v7_high_precision")

foreach ($l in $EvalLegs) {
    Write-Host "`n==================================================" -ForegroundColor Cyan
    Write-Host "[i] Evaluating Leg: $l" -ForegroundColor Cyan
    Write-Host "==================================================" -ForegroundColor Cyan

    $Ckpt        = "outputs/biochem/biochem_gnn/$l/species/best.pth"
    $CompareJson = "outputs/biochem/biochem_gnn/$l/compare.json"

    if (-not (Test-Path $Ckpt)) {
        Write-Warning "[WARN] Checkpoint not found, skipping eval for $l : $Ckpt"
        continue
    }

    $evalArgs = @(
        "scripts/eval_mat_growth_simple.py",
        "--ckpt",    $Ckpt,
        "--out",     $CompareJson,
        "--anchors", $Anchors
    )
    Invoke-PythonRcCheck -Label "eval $l" -PyArgs $evalArgs
}

# -----------------------------------------------------------------------
# STEP 3: Summary table
# -----------------------------------------------------------------------
Write-Host "`n==================== SUMMARY ====================" -ForegroundColor Green
foreach ($l in $EvalLegs) {
    $CompareJson = "outputs/biochem/biochem_gnn/$l/compare.json"
    if (Test-Path $CompareJson) {
        $j = Get-Content $CompareJson | ConvertFrom-Json
        $score  = [math]::Round($j.simple.mean.deploy_clot_score,  3)
        $clotf1 = [math]::Round($j.simple.mean.deploy_clot_f1,     3)
        $matf1  = [math]::Round($j.simple.mean.deploy_mat_f1,      3)
        $delta  = [math]::Round($j.delta_simple_minus_baseline.deploy_clot_score, 3)
        Write-Host ("  {0,-30}  score={1,6}  clot_f1={2,6}  mat_f1={3,6}  delta_score={4,+7}" -f $l, $score, $clotf1, $matf1, $delta)
    } else {
        Write-Host "  $l : no compare.json found"
    }
}
Write-Host "[OK] go_fresh_canonical_finish done." -ForegroundColor Green
