# Limit analysis (~2h): can extreme recall reward open patient001 lumen / raise recall?
#
# Question: tuning vs architecture?
#   If RecallPush gets hop_ge2 on 001 and higher recall on 007 -> capacity exists (tuning/data).
#   If 001 stays at 0 while 007 only sprays -> likely inductive-bias / transfer limit.
#
# Arms:
#   A            - frozen WC_v7 wall alone
#   Prec8hRef    - existing frontier_ge2_prec_8h ckpt (no train)
#   RecallPush   - train on lumen teachers 001+007 only; extreme FN/underpred;
#                  --ckpt-metric hop_ge2_recall; frontier_ge2; freeze-backbone
#
# Probe anchors: 001,007 (teachers) + 004,008 (spray sentinels)
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_wc_v7_lumen_recall_limit_2h.ps1 -Fresh
#   powershell ... -Smoke -Fresh
#   powershell ... -EvalOnly
#

param(
    [int] $Epochs = 6,
    [int] $EarlyStop = 4,
    [int] $MaxWindows = 24,
    [int] $HopsK = 5,
    [double] $LumenShapeWeight = 10.0,
    [string] $TrainAnchors = "patient001,patient007",
    [string] $ProbeAnchors = "patient001,patient007,patient004,patient008",
    [string] $WallCkpt = "outputs/biochem/biochem_gnn/locked/species_gnn_best.pth",
    [string] $Prec8hCkpt = "outputs/biochem/offwall_model/wc_v7_frontier_ge2_prec_8h/growth_frontier_ge2_prec/best.pth",
    [string] $RunRoot = "outputs/biochem/offwall_model/wc_v7_lumen_recall_limit_2h",
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
    $RunRoot = "outputs/biochem/offwall_model/wc_v7_lumen_recall_limit_smoke"
    $EvalOnly = $false
    Write-Host "[i] SMOKE preset (train-only RecallPush; skips probes)" -ForegroundColor Yellow
}

$OutDir = Join-Path $RepoRoot $RunRoot
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$WallPath = Join-Path $RepoRoot $WallCkpt
if (-not (Test-Path $WallPath)) {
    throw "Wall ckpt missing: $WallPath"
}

$deadline = (Get-Date).AddHours(2)
Write-Host "[NEW] go_wc_v7_lumen_recall_limit_2h epochs=$Epochs max_windows=$MaxWindows" -ForegroundColor Cyan
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
        $g = $GrowthCkpt
        if (-not [System.IO.Path]::IsPathRooted($g)) { $g = Join-Path $RepoRoot $g }
        if (-not (Test-Path $g)) {
            Write-Host "[WARN] skip probe $ArmId; missing growth ckpt $g" -ForegroundColor Yellow
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

# --- Train RecallPush ---
$growthDir = Join-Path $OutDir "growth_RecallPush"
$growthCkpt = Join-Path $growthDir "best.pth"
New-Item -ItemType Directory -Force -Path $growthDir | Out-Null

if ($Fresh) {
    Remove-Item -Force $growthCkpt, (Join-Path $growthDir "train_log.jsonl"), (Join-Path $growthDir "best.json") -ErrorAction SilentlyContinue
}

if (-not $EvalOnly) {
    if (-not (Test-BudgetOk)) { throw "budget exceeded before train" }
    # Extreme recall tilt (limit analysis — not a deploy recipe)
    $env:SPECIES_LUMEN_SHAPE_FN_W = "25"
    $env:SPECIES_LUMEN_SHAPE_FP_W = "0.25"
    $env:SPECIES_CONTINUOUS_UNDERPRED_WEIGHT = "12.0"

    $gArgs = @(
        "-m", "src.training.train_offwall_growth",
        "--val-anchor", "patient007",
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
    Write-Host "[i] Training RecallPush (teachers=$TrainAnchors, extreme FN)..." -ForegroundColor Cyan
    $null = Invoke-PythonRcCheck -Label "train RecallPush" -PyArgs $gArgs
    Clear-TrainEnv
}

if ($Smoke) {
    if (-not (Test-Path $growthCkpt)) { throw "Smoke missing ckpt: $growthCkpt" }
    Write-Host "[OK] SMOKE RecallPush train path OK -> $growthCkpt" -ForegroundColor Green
    exit 0
}

if (-not (Test-Path $growthCkpt)) {
    throw "Missing RecallPush ckpt: $growthCkpt"
}

# --- Probes ---
if (Test-BudgetOk) { Invoke-CompoundProbe -ArmId "A" }
if (Test-BudgetOk) {
    $precPath = if ([System.IO.Path]::IsPathRooted($Prec8hCkpt)) { $Prec8hCkpt } else { Join-Path $RepoRoot $Prec8hCkpt }
    if (Test-Path $precPath) {
        Invoke-CompoundProbe -ArmId "Prec8hRef" -GrowthCkpt $Prec8hCkpt
    } else {
        Write-Host "[WARN] Prec8hRef ckpt missing; skip" -ForegroundColor Yellow
    }
}
if (Test-BudgetOk) { Invoke-CompoundProbe -ArmId "RecallPush" -GrowthCkpt $growthCkpt }

Write-Host "[i] Summarize recall-limit probes..." -ForegroundColor Cyan
$null = Invoke-PythonRcCheck -Label "summarize recall_limit_2h" -PyArgs @(
    "scripts/summarize_lumen_recall_limit_2h.py",
    "--run-root", $RunRoot,
    "--out", (Join-Path $OutDir "compare_recall_limit.json")
)

Write-Host "[OK] lumen recall limit 2h done -> $OutDir" -ForegroundColor Green
Write-Host "[i] Look for 001 hop_ge2 leave-0 and 007 recall up without catastrophic 004/008 spray" -ForegroundColor DarkGray
