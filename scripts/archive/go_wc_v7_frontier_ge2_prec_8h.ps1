# WC_v7 Frontier-ge2 precision lumen compound (~8 h GPU budget, orig10).
#
# Recipe (post FrontierLumen 6h spray / wall-floor miss):
#   freeze-backbone growth specialist
#   --supervise-mode frontier_ge2  (dilate(clot)&hop>=2)
#   --loss-mode loss_lumen_shape with precision tilt (FN~5, FP~2.5, underpred~3, w~4)
#   --compound-val on patient007 + wall clot F1 floor reject (A - 0.02)
#   deploy: wall-route compound vs frozen locked WC_v7
#
# Budget sketch (4GB-class GPU):
#   Arm-A floor + compound vals ~1.5-2 h
#   train ~4-4.5 h (16 ep / ES 6 / max-windows 56 / hops-k 5)
#   orig10 A + S eval ~1.5-2 h
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_wc_v7_frontier_ge2_prec_8h.ps1 -Fresh
#   powershell ... -Smoke -Fresh
#   powershell ... -Fast -Fresh
#   powershell ... -EvalOnly
#

param(
    [int] $Epochs = 16,
    [int] $EarlyStop = 6,
    [int] $MaxWindows = 56,
    [int] $HopsK = 5,
    [int] $CompoundValEvery = 2,
    [double] $LumenShapeWeight = 4.0,
    [double] $WallClotFloorDelta = 0.02,
    [string] $ValAnchor = "patient007",
    [string] $TrainAnchors = "patient001,patient002,patient003,patient004,patient005,patient006,patient007,patient008,patient010,patient011",
    [string] $WallCkpt = "outputs/biochem/biochem_gnn/locked/species_gnn_best.pth",
    [string] $RunRoot = "outputs/biochem/offwall_model/wc_v7_frontier_ge2_prec_8h",
    [switch] $Fast,
    [switch] $Smoke,
    [switch] $Fresh,
    [switch] $EvalOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

if ($Smoke) {
    $Epochs = 1
    $EarlyStop = 1
    $MaxWindows = 4
    $TrainAnchors = "patient007"
    $RunRoot = "outputs/biochem/offwall_model/wc_v7_frontier_ge2_prec_smoke"
    $EvalOnly = $false
    Write-Host "[i] SMOKE preset (train-only cheap-val; skips A/S deploy eval)" -ForegroundColor Yellow
} elseif ($Fast) {
    $Epochs = 6
    $EarlyStop = 3
    $MaxWindows = 24
    $CompoundValEvery = 2
    Write-Host "[i] FAST preset (~half budget)" -ForegroundColor Yellow
}

$OutDir = Join-Path $RepoRoot $RunRoot
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$WallPath = Join-Path $RepoRoot $WallCkpt
if (-not (Test-Path $WallPath)) {
    throw "Wall/canonical ckpt missing: $WallPath"
}

$deadline = (Get-Date).AddHours(8)
Write-Host "[NEW] go_wc_v7_frontier_ge2_prec_8h epochs=$Epochs early_stop=$EarlyStop max_windows=$MaxWindows" -ForegroundColor Cyan
Write-Host "[i] train_anchors=$TrainAnchors run_root=$RunRoot" -ForegroundColor DarkGray
Write-Host "[i] soft deadline ~8h -> $deadline" -ForegroundColor DarkGray
Write-Host "[i] recipe=frontier_ge2 freeze-backbone loss_lumen_shape w=$LumenShapeWeight compound-val every=$CompoundValEvery" -ForegroundColor DarkGray

$growthDir = Join-Path $OutDir "growth_frontier_ge2_prec"
$growthCkpt = Join-Path $growthDir "best.pth"
$evalA = Join-Path $OutDir "eval_A_canonical.json"
$evalS = Join-Path $OutDir "eval_S_frontier_ge2_prec.json"
$compare = Join-Path $OutDir "compare_frontier_ge2_prec.json"
New-Item -ItemType Directory -Force -Path $growthDir | Out-Null

if ($Fresh) {
    Remove-Item -Force $growthCkpt, $evalA, $evalS, $compare -ErrorAction SilentlyContinue
    Remove-Item -Force (Join-Path $growthDir "train_log.jsonl"), (Join-Path $growthDir "best.json") -ErrorAction SilentlyContinue
    Remove-Item -Force (Join-Path $growthDir "_compound_val_growth_tmp.pth") -ErrorAction SilentlyContinue
}

# Precision-tilted lumen shape (not the extreme FN=20 / FP=0.25 FrontierLumen recipe)
$env:SPECIES_LUMEN_SHAPE_FN_W = "5"
$env:SPECIES_LUMEN_SHAPE_FP_W = "2.5"
$env:SPECIES_CONTINUOUS_UNDERPRED_WEIGHT = "3.0"

if (-not $EvalOnly) {
    if ((Get-Date) -gt $deadline) {
        throw "8h budget already exceeded before train"
    }
    $gArgs = @(
        "-m", "src.training.train_offwall_growth",
        "--val-anchor", $ValAnchor,
        "--anchors", $TrainAnchors,
        "--epochs", "$Epochs",
        "--early-stop", "$EarlyStop",
        "--max-windows", "$MaxWindows",
        "--hops-k", "$HopsK",
        "--supervise-mode", "frontier_ge2",
        "--frontier-hops", "2",
        "--loss-mode", "loss_lumen_shape",
        "--lumen-shape-weight", "$LumenShapeWeight",
        "--ckpt-metric", "hop_ge2_balanced",
        "--freeze-backbone",
        "--mat-leg", "WC_v7_clot_phi_mse",
        "--init", $WallCkpt,
        "--out", $growthCkpt
    )
    if ($Smoke) {
        $gArgs += @("--cheap-val")
    } else {
        $gArgs += @(
            "--compound-val",
            "--wall-ckpt", $WallCkpt,
            "--wall-clot-floor-delta", "$WallClotFloorDelta",
            "--compound-val-every", "$CompoundValEvery"
        )
    }
    Write-Host "[i] Training Frontier-ge2 precision specialist..." -ForegroundColor Cyan
    Invoke-PythonRcCheck -Label "train frontier_ge2_prec" -PyArgs $gArgs
}

if (-not (Test-Path $growthCkpt)) {
    throw "Missing growth ckpt: $growthCkpt"
}

# Clear train-only loss env before deploy-faithful eval
Remove-Item Env:SPECIES_LUMEN_SHAPE_FN_W, Env:SPECIES_LUMEN_SHAPE_FP_W, Env:SPECIES_CONTINUOUS_UNDERPRED_WEIGHT -ErrorAction SilentlyContinue
$env:SPECIES_TWO_MODEL_MODE = "0"
Remove-Item Env:SPECIES_OFFWALL_MODEL_CKPT, Env:SPECIES_TWO_MODEL_ROUTE -ErrorAction SilentlyContinue

if ($Smoke) {
    Write-Host "[OK] SMOKE train path OK -> $growthCkpt" -ForegroundColor Green
    Write-Host "[i] Skipping A/S deploy eval in -Smoke (use full launch without -Smoke for cohort eval)" -ForegroundColor DarkGray
    exit 0
}

Write-Host "[i] Eval Arm A (canonical wall)..." -ForegroundColor Cyan
Invoke-PythonRcCheck -Label "eval Arm A" -PyArgs @(
    "scripts/eval_mat_growth_simple.py",
    "--ckpt", $WallCkpt,
    "--mat-leg", "WC_v7_clot_phi_mse",
    "--no-baseline",
    "--out", $evalA,
    "--anchors", $TrainAnchors
)

if ((Get-Date) -gt $deadline) {
    Write-Host "[WARN] past 8h soft deadline before compound eval; continuing eval anyway" -ForegroundColor Yellow
}

Write-Host "[i] Eval Arm S (compound wall + Frontier-ge2 prec)..." -ForegroundColor Cyan
Invoke-PythonRcCheck -Label "eval Arm S frontier_ge2_prec" -PyArgs @(
    "scripts/eval_mat_growth_simple.py",
    "--ckpt", $WallCkpt,
    "--mat-leg", "WC_v7_clot_phi_mse",
    "--no-baseline",
    "--out", $evalS,
    "--anchors", $TrainAnchors,
    "--offwall-ckpt", $growthCkpt,
    "--two-model-route", "wall",
    "--two-model-frontier-hops", "2"
)

Write-Host "[i] Summarize A vs S..." -ForegroundColor Cyan
Invoke-PythonRcCheck -Label "summarize frontier_ge2_prec 8h" -PyArgs @(
    "scripts/summarize_frontier_ge2_prec_8h.py",
    "--arm-a", $evalA,
    "--arm-s", $evalS,
    "--wall-clot-floor-delta", "$WallClotFloorDelta",
    "--out", $compare
)

Write-Host "[OK] Frontier-ge2 prec 8h done -> $OutDir" -ForegroundColor Green
Write-Host "[i] Gates: wall clot F1 >= A-$WallClotFloorDelta; hop_ge2 n_pred up; hop_ge2 strict F1 up vs 6h (0.017)" -ForegroundColor DarkGray
