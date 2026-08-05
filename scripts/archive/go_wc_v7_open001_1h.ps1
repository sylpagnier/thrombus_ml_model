# ~1h test: open patient001 lumen without killing other graphs.
#
# Recipe:
#   Train lumen teachers 001+007+010 (oversample the miss + known signal)
#   frontier_ge2 + recall-tilted loss + hop_ge2_recall ckpt
#   freeze-backbone, cheap-val (budget)
#   Probe: teachers + spray sentinels + 006 (not full orig10; ~1h soft cap)
#
# Success read:
#   001 hop_ge2 pred leaves 0; 007/010 still have signal; 004/008 spray not wild
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_wc_v7_open001_1h.ps1 -Fresh
#   powershell ... -Smoke -Fresh
#

param(
    [int] $Epochs = 4,
    [int] $EarlyStop = 3,
    [int] $MaxWindows = 16,
    [int] $HopsK = 5,
    [double] $LumenShapeWeight = 8.0,
    [string] $TrainAnchors = "patient001,patient007,patient010",
    [string] $ProbeAnchors = "patient001,patient007,patient010,patient006,patient004,patient008",
    [string] $WallCkpt = "outputs/biochem/biochem_gnn/locked/species_gnn_best.pth",
    [string] $Prec8hCkpt = "outputs/biochem/offwall_model/wc_v7_frontier_ge2_prec_8h/growth_frontier_ge2_prec/best.pth",
    [string] $RunRoot = "outputs/biochem/offwall_model/wc_v7_open001_1h",
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
    $TrainAnchors = "patient001"
    $ProbeAnchors = "patient001"
    $RunRoot = "outputs/biochem/offwall_model/wc_v7_open001_smoke"
    $EvalOnly = $false
    Write-Host "[i] SMOKE preset" -ForegroundColor Yellow
}

$OutDir = Join-Path $RepoRoot $RunRoot
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$WallPath = Join-Path $RepoRoot $WallCkpt
if (-not (Test-Path $WallPath)) {
    throw "Wall ckpt missing: $WallPath"
}

$deadlineHours = if ($EvalOnly) { 2.0 } else { 1.0 }
$deadline = (Get-Date).AddHours($deadlineHours)
Write-Host "[NEW] go_wc_v7_open001_1h epochs=$Epochs max_windows=$MaxWindows" -ForegroundColor Cyan
Write-Host "[i] train=$TrainAnchors probe=$ProbeAnchors out=$RunRoot" -ForegroundColor DarkGray
Write-Host "[i] soft deadline ~${deadlineHours}h -> $deadline" -ForegroundColor DarkGray

function Test-BudgetOk {
    if ($EvalOnly) { return $true }
    if ((Get-Date) -gt $deadline) {
        Write-Host "[WARN] 1h budget exceeded; skipping remaining work" -ForegroundColor Yellow
        return $false
    }
    return $true
}

function Clear-TrainEnv {
    Remove-Item Env:SPECIES_LUMEN_SHAPE_FN_W, Env:SPECIES_LUMEN_SHAPE_FP_W, Env:SPECIES_CONTINUOUS_UNDERPRED_WEIGHT -ErrorAction SilentlyContinue
    $env:SPECIES_TWO_MODEL_MODE = "0"
    Remove-Item Env:SPECIES_OFFWALL_MODEL_CKPT, Env:SPECIES_TWO_MODEL_ROUTE -ErrorAction SilentlyContinue
}

function Invoke-CompoundProbe {
    param(
        [string] $ArmId,
        [string] $GrowthCkpt = ""
    )
    if (-not (Test-BudgetOk)) { return }
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
        $g = $GrowthCkpt
        if (-not [System.IO.Path]::IsPathRooted($g)) { $g = Join-Path $RepoRoot $g }
        if (-not (Test-Path $g)) {
            Write-Host "[WARN] skip probe $ArmId; missing $g" -ForegroundColor Yellow
            return
        }
        $evalArgs += @(
            "--offwall-ckpt", $GrowthCkpt,
            "--two-model-route", "wall",
            "--two-model-frontier-hops", "2"
        )
    }
    Write-Host "[i] Probe $ArmId -> $probeOut" -ForegroundColor Cyan
    $null = Invoke-PythonRcCheck -Label "probe $ArmId" -PyArgs $evalArgs
}

$growthDir = Join-Path $OutDir "growth_Open001"
$growthCkpt = Join-Path $growthDir "best.pth"
New-Item -ItemType Directory -Force -Path $growthDir | Out-Null
if ($Fresh) {
    Remove-Item -Force $growthCkpt, (Join-Path $growthDir "train_log.jsonl"), (Join-Path $growthDir "best.json") -ErrorAction SilentlyContinue
}

if (-not $EvalOnly) {
    if (-not (Test-BudgetOk)) { throw "1h budget already exceeded before train" }
    # Strong recall tilt, but not the full limit-2h extreme (keep some precision for spray anchors)
    $env:SPECIES_LUMEN_SHAPE_FN_W = "15"
    $env:SPECIES_LUMEN_SHAPE_FP_W = "0.5"
    $env:SPECIES_CONTINUOUS_UNDERPRED_WEIGHT = "8.0"

    $gArgs = @(
        "-m", "src.training.train_offwall_growth",
        "--val-anchor", "patient001",
        "--anchors", $TrainAnchors,
        "--epochs", "$Epochs",
        "--early-stop", "$EarlyStop",
        "--max-windows", "$MaxWindows",
        "--hops-k", "$HopsK",
        "--tile-mode", "union",
        "--supervise-mode", "frontier_ge2",
        "--frontier-hops", "2",
        "--loss-mode", "loss_lumen_shape",
        "--lumen-shape-weight", "$LumenShapeWeight",
        "--ckpt-metric", "hop_ge2_recall",
        "--freeze-backbone",
        "--cheap-val",
        "--mat-leg", "WC_v7_clot_phi_mse",
        "--init", $WallCkpt,
        "--out", $growthCkpt
    )
    Write-Host "[i] Training Open001 specialist (val-anchor=patient001)..." -ForegroundColor Cyan
    $null = Invoke-PythonRcCheck -Label "train Open001" -PyArgs $gArgs
    Clear-TrainEnv
}

if ($Smoke) {
    if (-not (Test-Path $growthCkpt)) { throw "Smoke missing ckpt: $growthCkpt" }
    Write-Host "[OK] SMOKE Open001 train path OK -> $growthCkpt" -ForegroundColor Green
    exit 0
}

if (-not (Test-Path $growthCkpt)) {
    throw "Missing Open001 ckpt: $growthCkpt"
}

if (Test-BudgetOk) { Invoke-CompoundProbe -ArmId "A" }
if (Test-BudgetOk) {
    $precPath = if ([System.IO.Path]::IsPathRooted($Prec8hCkpt)) { $Prec8hCkpt } else { Join-Path $RepoRoot $Prec8hCkpt }
    if (Test-Path $precPath) {
        Invoke-CompoundProbe -ArmId "Prec8hRef" -GrowthCkpt $Prec8hCkpt
    } else {
        Write-Host "[WARN] Prec8hRef missing; skip" -ForegroundColor Yellow
    }
}
if (Test-BudgetOk) { Invoke-CompoundProbe -ArmId "Open001" -GrowthCkpt $growthCkpt }

Write-Host "[i] Summarize open001 1h..." -ForegroundColor Cyan
$null = Invoke-PythonRcCheck -Label "summarize open001_1h" -PyArgs @(
    "scripts/summarize_lumen_recall_limit_2h.py",
    "--run-root", $RunRoot,
    "--out", (Join-Path $OutDir "compare_open001.json")
)

Write-Host "[OK] open001 1h done -> $OutDir" -ForegroundColor Green
Write-Host "[i] Primary gate: patient001 hop_ge2 pred > 0; check 007/010 hold; watch 004/008 spray" -ForegroundColor DarkGray
