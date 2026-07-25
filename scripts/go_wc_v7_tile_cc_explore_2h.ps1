# WC_v7 exploratory 2h: union tile vs per-clot-region tiles.
#
# Question: does training one local subgraph per uninterrupted clot CC
# help or hurt vs one union tile per window (current default)?
#
# Arms:
#   A            - frozen WC_v7 wall alone (probe baseline)
#   UnionTile    - frontier_ge2 + --tile-mode union
#   PerComponent - frontier_ge2 + --tile-mode per_component (max 8 CCs/window)
#
# Same precision-tilt recipe as the 8h run; cheap-val; compound probe on probe anchors.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_wc_v7_tile_cc_explore_2h.ps1 -Fresh
#   powershell ... -Smoke -Fresh
#

param(
    [int] $Epochs = 5,
    [int] $EarlyStop = 4,
    [int] $MaxWindows = 12,
    [int] $HopsK = 5,
    [int] $MaxTilesPerWindow = 8,
    [double] $LumenShapeWeight = 4.0,
    [string] $TrainAnchors = "patient007,patient004,patient001",
    [string] $ProbeAnchors = "patient007,patient004",
    [string] $WallCkpt = "outputs/biochem/biochem_gnn/locked/species_gnn_best.pth",
    [string] $RunRoot = "outputs/biochem/offwall_model/wc_v7_tile_cc_explore_2h",
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
    $MaxWindows = 2
    $TrainAnchors = "patient007"
    $ProbeAnchors = "patient007"
    $RunRoot = "outputs/biochem/offwall_model/wc_v7_tile_cc_explore_smoke"
    $EvalOnly = $false
    Write-Host "[i] SMOKE preset (train-only arms; skips compound probe)" -ForegroundColor Yellow
}

$OutDir = Join-Path $RepoRoot $RunRoot
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$WallPath = Join-Path $RepoRoot $WallCkpt
if (-not (Test-Path $WallPath)) {
    throw "Wall/canonical ckpt missing: $WallPath"
}

$deadline = (Get-Date).AddHours(2)
Write-Host "[NEW] go_wc_v7_tile_cc_explore_2h epochs=$Epochs max_windows=$MaxWindows" -ForegroundColor Cyan
Write-Host "[i] train=$TrainAnchors probe=$ProbeAnchors out=$RunRoot" -ForegroundColor DarkGray
Write-Host "[i] soft deadline ~2h -> $deadline" -ForegroundColor DarkGray

function Test-BudgetOk {
    if ((Get-Date) -gt $deadline) {
        Write-Host "[WARN] 2h budget exceeded; skipping remaining work" -ForegroundColor Yellow
        return $false
    }
    return $true
}

function Clear-TrainEnv {
    Remove-Item Env:SPECIES_LUMEN_SHAPE_FN_W, Env:SPECIES_LUMEN_SHAPE_FP_W, Env:SPECIES_CONTINUOUS_UNDERPRED_WEIGHT -ErrorAction SilentlyContinue
    $env:SPECIES_TWO_MODEL_MODE = "0"
    Remove-Item Env:SPECIES_OFFWALL_MODEL_CKPT, Env:SPECIES_TWO_MODEL_ROUTE -ErrorAction SilentlyContinue
}

function Invoke-GrowthArm {
    param(
        [string] $ArmId,
        [string] $TileMode
    )
    if (-not (Test-BudgetOk)) { return $null }
    $armDir = Join-Path $OutDir "growth_$ArmId"
    $ckpt = Join-Path $armDir "best.pth"
    New-Item -ItemType Directory -Force -Path $armDir | Out-Null
    if ($Fresh) {
        Remove-Item -Force $ckpt, (Join-Path $armDir "train_log.jsonl"), (Join-Path $armDir "best.json") -ErrorAction SilentlyContinue
    }
    if ($EvalOnly -and (Test-Path $ckpt)) {
        return $ckpt
    }

    $env:SPECIES_LUMEN_SHAPE_FN_W = "5"
    $env:SPECIES_LUMEN_SHAPE_FP_W = "2.5"
    $env:SPECIES_CONTINUOUS_UNDERPRED_WEIGHT = "3.0"

    $gArgs = @(
        "-m", "src.training.train_offwall_growth",
        "--val-anchor", "patient007",
        "--anchors", $TrainAnchors,
        "--epochs", "$Epochs",
        "--early-stop", "$EarlyStop",
        "--max-windows", "$MaxWindows",
        "--hops-k", "$HopsK",
        "--tile-mode", $TileMode,
        "--max-tiles-per-window", "$MaxTilesPerWindow",
        "--supervise-mode", "frontier_ge2",
        "--frontier-hops", "2",
        "--loss-mode", "loss_lumen_shape",
        "--lumen-shape-weight", "$LumenShapeWeight",
        "--ckpt-metric", "hop_ge2_balanced",
        "--freeze-backbone",
        "--cheap-val",
        "--mat-leg", "WC_v7_clot_phi_mse",
        "--init", $WallCkpt,
        "--out", $ckpt
    )
    Write-Host "[i] Training arm $ArmId (tile-mode=$TileMode)..." -ForegroundColor Cyan
    # Discard exit code: PowerShell would otherwise append it to the function output
    # and corrupt $ckptUnion / --offwall-ckpt paths ("0 C:\...").
    $null = Invoke-PythonRcCheck -Label "train $ArmId" -PyArgs $gArgs
    Clear-TrainEnv
    if (-not (Test-Path $ckpt)) {
        throw "Missing growth ckpt for $ArmId : $ckpt"
    }
    return , $ckpt
}

function Invoke-CompoundProbe {
    param(
        [string] $ArmId,
        [string] $GrowthCkpt = ""
    )
    Clear-TrainEnv
    $probeOut = Join-Path $OutDir "probe_$ArmId.json"
    if ($Fresh) { Remove-Item -Force $probeOut -ErrorAction SilentlyContinue }
    $evalArgs = @(
        "scripts/eval_mat_growth_simple.py",
        "--ckpt", $WallCkpt,
        "--mat-leg", "WC_v7_clot_phi_mse",
        "--no-baseline",
        "--out", $probeOut,
        "--anchors", $ProbeAnchors
    )
    if ($GrowthCkpt -and $GrowthCkpt.Trim()) {
        $evalArgs += @(
            "--offwall-ckpt", $GrowthCkpt,
            "--two-model-route", "wall",
            "--two-model-frontier-hops", "2"
        )
    }
    Write-Host "[i] Probe $ArmId -> $probeOut" -ForegroundColor Cyan
    Invoke-PythonRcCheck -Label "probe $ArmId" -PyArgs $evalArgs
}

# --- Train arms ---
$ckptUnion = $null
$ckptCC = $null
if (-not $EvalOnly) {
    $ckptUnion = Invoke-GrowthArm -ArmId "UnionTile" -TileMode "union"
    $ckptCC = Invoke-GrowthArm -ArmId "PerComponent" -TileMode "per_component"
} else {
    $ckptUnion = Join-Path $OutDir "growth_UnionTile/best.pth"
    $ckptCC = Join-Path $OutDir "growth_PerComponent/best.pth"
}

if ($Smoke) {
    Write-Host "[OK] SMOKE train path OK (UnionTile + PerComponent)" -ForegroundColor Green
    exit 0
}

# --- Probes ---
if (Test-BudgetOk) { Invoke-CompoundProbe -ArmId "A" }
if (Test-BudgetOk -and $ckptUnion -and (Test-Path $ckptUnion)) {
    Invoke-CompoundProbe -ArmId "UnionTile" -GrowthCkpt $ckptUnion
}
if (Test-BudgetOk -and $ckptCC -and (Test-Path $ckptCC)) {
    Invoke-CompoundProbe -ArmId "PerComponent" -GrowthCkpt $ckptCC
}

Write-Host "[i] Summarize union vs per-component..." -ForegroundColor Cyan
Invoke-PythonRcCheck -Label "summarize tile_cc_explore_2h" -PyArgs @(
    "scripts/summarize_tile_cc_explore_2h.py",
    "--run-root", $RunRoot,
    "--out", (Join-Path $OutDir "compare_tile_cc.json")
)

Write-Host "[OK] tile CC explore 2h done -> $OutDir" -ForegroundColor Green
Write-Host "[i] Verdict in compare_tile_cc.json (helps / hurts / null vs UnionTile)" -ForegroundColor DarkGray
