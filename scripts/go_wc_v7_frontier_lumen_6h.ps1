# WC_v7 FrontierLumen scale-up (~6 h GPU budget, orig10).
#
# Recipe from limit-2h winner (lumen-only frontier loss):
#   freeze-backbone growth specialist
#   --supervise-mode frontier_lumen  (dilate(clot)&~wall)
#   --loss-mode loss_lumen_shape + strong FN/underpred
#   compound wall-route eval vs frozen locked WC_v7
#
# Budget sketch (4GB-class GPU):
#   train ~3-4 h (16 ep / ES 8 / max-windows 48 / cheap-val)
#   orig10 A + S eval ~1.5-2 h
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_wc_v7_frontier_lumen_6h.ps1 -Fresh
#   powershell ... -Smoke -Fresh
#   powershell ... -Fast -Fresh
#   powershell ... -EvalOnly
#

param(
    [int] $Epochs = 16,
    [int] $EarlyStop = 8,
    [int] $MaxWindows = 48,
    [double] $LumenShapeWeight = 8.0,
    [string] $ValAnchor = "patient007",
    [string] $TrainAnchors = "patient001,patient002,patient003,patient004,patient005,patient006,patient007,patient008,patient010,patient011",
    [string] $WallCkpt = "outputs/biochem/biochem_gnn/locked/species_gnn_best.pth",
    [string] $RunRoot = "outputs/biochem/offwall_model/wc_v7_frontier_lumen_6h",
    [double] $WallClotFloorDelta = 0.03,
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
    $RunRoot = "outputs/biochem/offwall_model/wc_v7_frontier_lumen_smoke"
    # Full deploy eval is ~10-15 min/anchor; smoke only validates the train path (<3 min).
    $EvalOnly = $false
    Write-Host "[i] SMOKE preset (train-only; skips A/S deploy eval)" -ForegroundColor Yellow
} elseif ($Fast) {
    $Epochs = 8
    $EarlyStop = 4
    $MaxWindows = 24
    Write-Host "[i] FAST preset (~half budget)" -ForegroundColor Yellow
}

$OutDir = Join-Path $RepoRoot $RunRoot
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$WallPath = Join-Path $RepoRoot $WallCkpt
if (-not (Test-Path $WallPath)) {
    throw "Wall/canonical ckpt missing: $WallPath"
}

$deadline = (Get-Date).AddHours(6)
Write-Host "[NEW] go_wc_v7_frontier_lumen_6h epochs=$Epochs early_stop=$EarlyStop max_windows=$MaxWindows" -ForegroundColor Cyan
Write-Host "[i] train_anchors=$TrainAnchors run_root=$RunRoot" -ForegroundColor DarkGray
Write-Host "[i] soft deadline ~6h -> $deadline" -ForegroundColor DarkGray
Write-Host "[i] recipe=frontier_lumen freeze-backbone loss_lumen_shape w=$LumenShapeWeight" -ForegroundColor DarkGray

$growthDir = Join-Path $OutDir "growth_frontier_lumen"
$growthCkpt = Join-Path $growthDir "best.pth"
$evalA = Join-Path $OutDir "eval_A_canonical.json"
$evalS = Join-Path $OutDir "eval_S_frontier_lumen.json"
$compare = Join-Path $OutDir "compare_frontier_lumen.json"
New-Item -ItemType Directory -Force -Path $growthDir | Out-Null

if ($Fresh) {
    Remove-Item -Force $growthCkpt, $evalA, $evalS, $compare -ErrorAction SilentlyContinue
    Remove-Item -Force (Join-Path $growthDir "train_log.jsonl"), (Join-Path $growthDir "best.json") -ErrorAction SilentlyContinue
}

# Extreme lumen push (same family as limit-2h FrontierLumen; slightly softer Dice weight default 8)
$env:SPECIES_LUMEN_SHAPE_FN_W = "20"
$env:SPECIES_LUMEN_SHAPE_FP_W = "0.25"
$env:SPECIES_CONTINUOUS_UNDERPRED_WEIGHT = "8.0"

if (-not $EvalOnly) {
    if ((Get-Date) -gt $deadline) {
        throw "6h budget already exceeded before train"
    }
    $gArgs = @(
        "-m", "src.training.train_offwall_growth",
        "--val-anchor", $ValAnchor,
        "--anchors", $TrainAnchors,
        "--epochs", "$Epochs",
        "--early-stop", "$EarlyStop",
        "--max-windows", "$MaxWindows",
        "--hops-k", "4",
        "--supervise-mode", "frontier_lumen",
        "--frontier-hops", "2",
        "--loss-mode", "loss_lumen_shape",
        "--lumen-shape-weight", "$LumenShapeWeight",
        "--ckpt-metric", "hop_ge2_balanced",
        "--freeze-backbone",
        "--cheap-val",
        "--mat-leg", "WC_v7_clot_phi_mse",
        "--init", $WallCkpt,
        "--out", $growthCkpt
    )
    Write-Host "[i] Training FrontierLumen specialist..." -ForegroundColor Cyan
    Invoke-PythonRcCheck -Label "train frontier_lumen" -PyArgs $gArgs
}

if (-not (Test-Path $growthCkpt)) {
    throw "Missing growth ckpt: $growthCkpt"
}

# Clear train-only extreme env before deploy-faithful eval
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
    Write-Host "[WARN] past 6h soft deadline before compound eval; continuing eval anyway" -ForegroundColor Yellow
}

Write-Host "[i] Eval Arm S (compound wall + FrontierLumen)..." -ForegroundColor Cyan
Invoke-PythonRcCheck -Label "eval Arm S frontier_lumen" -PyArgs @(
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
Invoke-PythonRcCheck -Label "summarize frontier_lumen 6h" -PyArgs @(
    "scripts/summarize_frontier_lumen_6h.py",
    "--arm-a", $evalA,
    "--arm-s", $evalS,
    "--wall-clot-floor-delta", "$WallClotFloorDelta",
    "--out", $compare
)

Write-Host "[OK] FrontierLumen 6h done -> $OutDir" -ForegroundColor Green
Write-Host "[i] Gates: wall clot F1 >= A-$WallClotFloorDelta; hop_ge2 n_pred up; hop_ge2 strict F1 up" -ForegroundColor DarkGray
