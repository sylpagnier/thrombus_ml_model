# Step 3: retrain lumen growth specialist under wall-only vel-decay contract.
#
# Warm-start D_Orig10_Band (lumen capacity under legacy wipe). Retrain heads with
# precision-tilted lumen shape + hop_ge2_balanced compound val so spray is no longer
# "filtered" by full-band vel-decay.
#
# Contract (must match deploy):
#   SPECIES_CONTINUOUS_VEL_DECAY=1
#   SPECIES_CONTINUOUS_VEL_DECAY_WALL_ONLY=1
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_wc_v7_wall_only_retrain.ps1 -Fresh
#   powershell ... -Fast -Fresh
#   powershell ... -Smoke -Fresh
#   powershell ... -EvalOnly
#
# Resume later (no mid-epoch ckpt): re-launch with -Fresh to clear stale wall-rejected
# artifacts, then train again from D_Orig10 warm-start under WALL_ONLY=1.
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_wc_v7_wall_only_retrain.ps1 -Fresh
#

param(
    [int] $Epochs = 10,
    [int] $EarlyStop = 4,
    [int] $MaxWindows = 40,
    [int] $HopsK = 5,
    [int] $CompoundValEvery = 2,
    [double] $LumenShapeWeight = 4.0,
    # A_floor on 001 is ~0.90; compound with open lumen sits ~0.79-0.81 — need slack.
    [double] $WallClotFloorDelta = 0.12,
    [string] $ValAnchor = "patient001",
    [string] $TrainAnchors = "patient001,patient002,patient003,patient004,patient005,patient006,patient007,patient008,patient010,patient011",
    [string] $WallCkpt = "outputs/biochem/biochem_gnn/locked/species_gnn_best.pth",
    [string] $InitGrowth = "outputs/biochem/offwall_model/wc_v7_open001_6h/growth_D_Orig10_Band/best.pth",
    [string] $RunRoot = "outputs/biochem/offwall_model/wc_v7_wall_only_retrain",
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
    $TrainAnchors = "patient001,patient007"
    $ValAnchor = "patient001"
    $RunRoot = "outputs/biochem/offwall_model/wc_v7_wall_only_retrain_smoke"
    $EvalOnly = $false
    Write-Host "[i] SMOKE preset" -ForegroundColor Yellow
} elseif ($Fast) {
    $Epochs = 6
    $EarlyStop = 3
    $MaxWindows = 24
    $CompoundValEvery = 2
    Write-Host "[i] FAST preset" -ForegroundColor Yellow
}

$OutDir = Join-Path $RepoRoot $RunRoot
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$WallPath = Join-Path $RepoRoot $WallCkpt
$InitPath = Join-Path $RepoRoot $InitGrowth
if (-not (Test-Path $WallPath)) { throw "Wall ckpt missing: $WallPath" }
if (-not (Test-Path $InitPath)) { throw "Init growth ckpt missing: $InitPath" }

$deadline = (Get-Date).AddHours(5)
Write-Host "[NEW] go_wc_v7_wall_only_retrain epochs=$Epochs early_stop=$EarlyStop max_windows=$MaxWindows" -ForegroundColor Cyan
Write-Host "[i] train=$TrainAnchors val=$ValAnchor init=$InitGrowth" -ForegroundColor DarkGray
Write-Host "[i] soft deadline ~5h -> $deadline" -ForegroundColor DarkGray
Write-Host "[i] contract: VEL_DECAY=1 WALL_ONLY=1; ckpt=hop_ge2_balanced; freeze-backbone" -ForegroundColor DarkGray

$growthDir = Join-Path $OutDir "growth_wall_only_retrain"
$growthCkpt = Join-Path $growthDir "best.pth"
$evalA = Join-Path $OutDir "eval_A_canonical.json"
$evalS = Join-Path $OutDir "eval_S_wall_only_retrain.json"
$compare = Join-Path $OutDir "compare_wall_only_retrain.json"
New-Item -ItemType Directory -Force -Path $growthDir | Out-Null

if ($Fresh) {
    Remove-Item -Force $growthCkpt, $evalA, $evalS, $compare -ErrorAction SilentlyContinue
    Remove-Item -Force (Join-Path $growthDir "train_log.jsonl"), (Join-Path $growthDir "best.json") -ErrorAction SilentlyContinue
    Remove-Item -Force (Join-Path $growthDir "_compound_val_growth_tmp.pth") -ErrorAction SilentlyContinue
}

# Deploy-faithful washout contract (also in GLOBAL_TRAIN_RECIPE / DEPLOY_INFERENCE_ENV)
$env:SPECIES_CONTINUOUS_VEL_DECAY = "1"
$env:SPECIES_CONTINUOUS_VEL_DECAY_WALL_ONLY = "1"

# Precision-tilted lumen shape (not extreme FN recall push)
$env:SPECIES_LUMEN_SHAPE_FN_W = "5"
$env:SPECIES_LUMEN_SHAPE_FP_W = "2.5"
$env:SPECIES_CONTINUOUS_UNDERPRED_WEIGHT = "3.0"

if (-not $EvalOnly) {
    if ((Get-Date) -gt $deadline) {
        throw "5h budget already exceeded before train"
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
        "--train-feat-source", "band",
        "--freeze-backbone",
        "--mat-leg", "WC_v7_clot_phi_mse",
        "--init", $InitGrowth,
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
    Write-Host "[i] Training wall-only lumen specialist..." -ForegroundColor Cyan
    Invoke-PythonRcCheck -Label "train wall_only_retrain" -PyArgs $gArgs
}

if (-not (Test-Path $growthCkpt)) {
    throw "Missing growth ckpt: $growthCkpt"
}

# Clear train-only loss env before deploy-faithful eval; keep washout contract.
Remove-Item Env:SPECIES_LUMEN_SHAPE_FN_W, Env:SPECIES_LUMEN_SHAPE_FP_W, Env:SPECIES_CONTINUOUS_UNDERPRED_WEIGHT -ErrorAction SilentlyContinue
$env:SPECIES_CONTINUOUS_VEL_DECAY = "1"
$env:SPECIES_CONTINUOUS_VEL_DECAY_WALL_ONLY = "1"
$env:SPECIES_TWO_MODEL_MODE = "0"
Remove-Item Env:SPECIES_OFFWALL_MODEL_CKPT, Env:SPECIES_TWO_MODEL_ROUTE -ErrorAction SilentlyContinue

if ($Smoke) {
    Write-Host "[OK] SMOKE train path OK -> $growthCkpt" -ForegroundColor Green
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
    Write-Host "[WARN] past soft deadline before compound eval; continuing" -ForegroundColor Yellow
}

Write-Host "[i] Eval Arm S (compound wall + wall-only retrain)..." -ForegroundColor Cyan
Invoke-PythonRcCheck -Label "eval Arm S wall_only_retrain" -PyArgs @(
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

Write-Host "[i] Summarize A vs S (+ legacy / unretrain wall-only refs)..." -ForegroundColor Cyan
Invoke-PythonRcCheck -Label "summarize wall_only_retrain" -PyArgs @(
    "scripts/summarize_wall_only_retrain.py",
    "--arm-a", $evalA,
    "--arm-s", $evalS,
    "--legacy-probe", "outputs/biochem/offwall_model/wc_v7_open001_6h/probe_D_Orig10_Band.json",
    "--wall-only-unretrain", "outputs/biochem/offwall_model/wc_v7_open001_6h/compare_vel_decay_wall_only.json",
    "--out", $compare
)

Write-Host "[OK] wall-only retrain done -> $OutDir" -ForegroundColor Green
Write-Host "[i] Gates: 001 ge2>0; mean clot F1 near A; spray down vs unretrain wall-only" -ForegroundColor DarkGray
