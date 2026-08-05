# Emergency follow-up: warm-start from Prec8h (opens 007 lumen) and finetune
# on patient001 with band feats + FN tilt. Goal: transfer lumen-writing to 001.
#
# Usage (after open001_6h or standalone):
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_wc_v7_open001_prec8h_ft.ps1
#

param(
    [int] $Epochs = 8,
    [int] $EarlyStop = 5,
    [int] $MaxWindows = 48,
    [double] $DeadlineHours = 3.0,
    [string] $RunRoot = "outputs/biochem/offwall_model/wc_v7_open001_6h",
    [string] $WallCkpt = "outputs/biochem/biochem_gnn/locked/species_gnn_best.pth",
    [string] $Prec8hCkpt = "outputs/biochem/offwall_model/wc_v7_frontier_ge2_prec_8h/growth_frontier_ge2_prec/best.pth",
    [switch] $Fresh
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

$OutDir = Join-Path $RepoRoot $RunRoot
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$deadline = (Get-Date).AddHours($DeadlineHours)
Write-Host "[NEW] open001 Prec8h finetune epochs=$Epochs deadline=$deadline" -ForegroundColor Cyan

function Test-BudgetOk {
    if ((Get-Date) -gt $deadline) {
        Write-Host "[WARN] budget exceeded" -ForegroundColor Yellow
        return $false
    }
    return $true
}

$prec = Join-Path $RepoRoot $Prec8hCkpt
if (-not (Test-Path $prec)) { throw "Prec8h ckpt missing: $prec" }

$arm = "E_Prec8h_FT001"
$growthDir = Join-Path $OutDir "growth_$arm"
$growthCkpt = Join-Path $growthDir "best.pth"
New-Item -ItemType Directory -Force -Path $growthDir | Out-Null
if ($Fresh) {
    Remove-Item -Force $growthCkpt, (Join-Path $growthDir "best.json"), (Join-Path $growthDir "train_log.jsonl") -ErrorAction SilentlyContinue
    Remove-Item -Force (Join-Path $OutDir "probe_$arm.json") -ErrorAction SilentlyContinue
}

if (-not (Test-Path $growthCkpt)) {
    if (-not (Test-BudgetOk)) { exit 1 }
    $env:SPECIES_LUMEN_SHAPE_FN_W = "25"
    $env:SPECIES_LUMEN_SHAPE_FP_W = "0.35"
    $env:SPECIES_CONTINUOUS_UNDERPRED_WEIGHT = "12.0"
    # Prefer closed-loop IC so train matches deploy resting rollout.
    $env:SPECIES_CONTINUOUS_CLOSED_LOOP_INIT = "0.85"
    $env:SPECIES_CONTINUOUS_FINAL_STATE_WEIGHT = "2.0"

    $gArgs = @(
        "-m", "src.training.train_offwall_growth",
        "--val-anchor", "patient001",
        "--anchors", "patient001,patient007",
        "--epochs", "$Epochs",
        "--early-stop", "$EarlyStop",
        "--max-windows", "$MaxWindows",
        "--hops-k", "5",
        "--tile-mode", "union",
        "--max-tiles-per-window", "8",
        "--supervise-mode", "frontier_ge2",
        "--frontier-hops", "2",
        "--loss-mode", "loss_lumen_shape",
        "--lumen-shape-weight", "12",
        "--ckpt-metric", "hop_ge2_recall",
        "--train-feat-source", "band",
        "--mat-leg", "WC_v7_clot_phi_mse",
        "--init", $Prec8hCkpt,
        "--out", $growthCkpt,
        "--compound-val",
        "--wall-ckpt", $WallCkpt,
        "--wall-clot-floor-delta", "0.08",
        "--compound-val-every", "2",
        "--freeze-backbone"
    )
    Write-Host "[i] Train $arm init=Prec8h anchors=001+007 closed_loop=0.85" -ForegroundColor Cyan
    $null = Invoke-PythonRcCheck -Label "train $arm" -PyArgs $gArgs
    Remove-Item Env:SPECIES_LUMEN_SHAPE_FN_W, Env:SPECIES_LUMEN_SHAPE_FP_W, Env:SPECIES_CONTINUOUS_UNDERPRED_WEIGHT, Env:SPECIES_CONTINUOUS_CLOSED_LOOP_INIT, Env:SPECIES_CONTINUOUS_FINAL_STATE_WEIGHT -ErrorAction SilentlyContinue
}

if (-not (Test-Path $growthCkpt)) { throw "missing $growthCkpt" }

# Gate probe 001 then orig10 if open
$probe = Join-Path $OutDir "probe_$arm.json"
$env:SPECIES_TWO_MODEL_MODE = "0"
Remove-Item Env:SPECIES_OFFWALL_MODEL_CKPT -ErrorAction SilentlyContinue
$null = Invoke-PythonRcCheck -Label "probe $arm gate" -PyArgs @(
    "scripts/eval_mat_growth_simple.py",
    "--ckpt", $WallCkpt,
    "--mat-leg", "WC_v7_clot_phi_mse",
    "--no-baseline",
    "--out", $probe,
    "--anchors", "patient001,patient007",
    "--offwall-ckpt", $growthCkpt,
    "--two-model-route", "wall",
    "--two-model-frontier-hops", "2"
)

$raw = Get-Content -Raw $probe | ConvertFrom-Json
$n001 = [double]$raw.simple.per_anchor.patient001.deploy_clot_offwall_n_pred_hop_ge2
$n007 = [double]$raw.simple.per_anchor.patient007.deploy_clot_offwall_n_pred_hop_ge2
Write-Host "[i] E gate 001 hop_ge2=$n001 007 hop_ge2=$n007" -ForegroundColor $(if ($n001 -gt 0.5) { "Green" } else { "Yellow" })

if ($n001 -gt 0.5 -and (Test-BudgetOk)) {
    $probeAll = Join-Path $OutDir "probe_${arm}_orig10.json"
    $orig10 = "patient001,patient002,patient003,patient004,patient005,patient006,patient007,patient008,patient010,patient011"
    $null = Invoke-PythonRcCheck -Label "probe $arm orig10" -PyArgs @(
        "scripts/eval_mat_growth_simple.py",
        "--ckpt", $WallCkpt,
        "--mat-leg", "WC_v7_clot_phi_mse",
        "--no-baseline",
        "--out", $probeAll,
        "--anchors", $orig10,
        "--offwall-ckpt", $growthCkpt,
        "--two-model-route", "wall",
        "--two-model-frontier-hops", "2"
    )
    Write-Host "[OK] 001 opened - orig10 probe -> $probeAll" -ForegroundColor Green
} else {
    Write-Host "[WARN] Prec8h FT still closed on 001" -ForegroundColor Yellow
}
