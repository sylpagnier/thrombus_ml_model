# WC_v7 firewall-break sequence (steps 1-4).
#
# Step 1: short WC_v7 finetune package + ablations
#   WC_v7_fw1_blind_sat (primary), optional -Ablations for blind/smooth/sat30 alone
# Step 2: hop>=2 lumen-shape growth specialist + compound eval vs frozen WC_v7
# Step 3: isolate / skiphop controlled WC_v7 fins (optional)
# Step 4: hop-stratified metrics always printed by eval (wired in metrics code)
#
# Default budget (~6-9 h on 4GB GPU, orig10 anchors):
#   Step1 primary ~2-3 h (16 ep ES 8)
#   Step2 specialist ~2-3 h + eval
#   Step3 optional extras
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_wc_v7_firewall_fix_seq.ps1 -Fresh
#   powershell ... -Step 1 -Fresh
#   powershell ... -Step 2 -Fresh
#   powershell ... -Ablations -Fresh
#   powershell ... -Smoke -Fresh
#

param(
    [ValidateSet(0, 1, 2, 3)]
    [int] $Step = 0,
    [int] $Epochs = 16,
    [int] $EarlyStop = 8,
    [int] $MaxWindows = 0,
    [string] $ValAnchor = "patient007",
    [string] $TrainAnchors = "patient001,patient002,patient003,patient004,patient005,patient006,patient007,patient008,patient010,patient011",
    [string] $WallCkpt = "outputs/biochem/biochem_gnn/locked/species_gnn_best.pth",
    [string] $RunRoot = "outputs/biochem/offwall_model/wc_v7_firewall_fix_seq",
    [switch] $Ablations,
    [switch] $IncludeStep3,
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
    $RunRoot = "outputs/biochem/offwall_model/wc_v7_firewall_fix_smoke"
    Write-Host "[i] SMOKE preset" -ForegroundColor Yellow
} elseif ($Fast) {
    $Epochs = 6
    $EarlyStop = 4
    $MaxWindows = 8
}

$OutDir = Join-Path $RepoRoot $RunRoot
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$WallPath = Join-Path $RepoRoot $WallCkpt
if (-not (Test-Path $WallPath)) {
    throw "Wall/canonical ckpt missing: $WallPath"
}

$do1 = ($Step -eq 0 -or $Step -eq 1)
$do2 = ($Step -eq 0 -or $Step -eq 2)
$do3 = ($IncludeStep3 -or $Step -eq 3)

Write-Host "[NEW] go_wc_v7_firewall_fix_seq step=$Step epochs=$Epochs early_stop=$EarlyStop" -ForegroundColor Cyan
Write-Host "[i] train_anchors=$TrainAnchors run_root=$RunRoot" -ForegroundColor DarkGray

function Invoke-MatLeg {
    param([string] $Leg)
    $legDir = Join-Path $OutDir $Leg
    $ckpt = Join-Path $legDir "species/best.pth"
    $compare = Join-Path $legDir "compare.json"
    New-Item -ItemType Directory -Force -Path (Split-Path $ckpt) | Out-Null
    if ($Fresh) {
        Remove-Item -Force $ckpt, $compare -ErrorAction SilentlyContinue
        Remove-Item -Force (Join-Path $legDir "species/train_log.jsonl") -ErrorAction SilentlyContinue
    }
    if (-not $EvalOnly) {
        $trainArgs = @(
            "-m", "src.training.train_species_pushforward_continuous",
            "--phase", "biochem_gnn",
            "--val-anchor", $ValAnchor,
            "--epochs", "$Epochs",
            "--early-stop", "$EarlyStop",
            "--max-windows", "$MaxWindows",
            "--recipe", "mat_growth_simple",
            "--leg", $Leg,
            "--init-mode", "backbone",
            "--init", $WallCkpt,
            "--out", $ckpt,
            "--anchors", $TrainAnchors
        )
        if ($Smoke) { $trainArgs += "--cheap-val" }
        Invoke-PythonRcCheck -Label "train $Leg" -PyArgs $trainArgs
    }
    if (-not (Test-Path $ckpt)) { throw "Missing ckpt after train: $ckpt" }
    $evalArgs = @(
        "scripts/eval_mat_growth_simple.py",
        "--ckpt", $ckpt,
        "--mat-leg", $Leg,
        "--no-baseline",
        "--out", $compare,
        "--anchors", $TrainAnchors
    )
    Invoke-PythonRcCheck -Label "eval $Leg" -PyArgs $evalArgs
}

# ---------------------------------------------------------------------------
# Step 1: WC_v7 firewall package (+ optional single-knob ablations)
# ---------------------------------------------------------------------------
if ($do1) {
    Write-Host "[i] Step 1: WC_v7_fw1_blind_sat (midside-blind + hop1-smooth + sat30)..." -ForegroundColor Cyan
    Invoke-MatLeg -Leg "WC_v7_fw1_blind_sat"
    if ($Ablations) {
        foreach ($leg in @("WC_v7_fw1_blind", "WC_v7_fw1_smooth", "WC_v7_fw1_sat30")) {
            Write-Host "[i] Step 1 ablation: $leg..." -ForegroundColor Cyan
            Invoke-MatLeg -Leg $leg
        }
    }
}

# ---------------------------------------------------------------------------
# Step 2: hop>=2 lumen-shape specialist + compound wall-route eval
# ---------------------------------------------------------------------------
if ($do2) {
    Write-Host "[i] Step 2: lumen-shape hop>=2 growth specialist..." -ForegroundColor Cyan
    $growthDir = Join-Path $OutDir "growth_hop_ge2_lumen_shape"
    $growthCkpt = Join-Path $growthDir "best.pth"
    $evalA = Join-Path $OutDir "eval_A_canonical.json"
    $evalS = Join-Path $OutDir "eval_S_compound_lumen_shape.json"
    New-Item -ItemType Directory -Force -Path $growthDir | Out-Null
    if ($Fresh) {
        Remove-Item -Force $growthCkpt, $evalA, $evalS -ErrorAction SilentlyContinue
        Remove-Item -Force (Join-Path $growthDir "train_log.jsonl") -ErrorAction SilentlyContinue
    }
    if (-not $EvalOnly) {
        $gArgs = @(
            "-m", "src.training.train_offwall_growth",
            "--val-anchor", $ValAnchor,
            "--epochs", "$Epochs",
            "--early-stop", "$EarlyStop",
            "--max-windows", "$MaxWindows",
            "--hops-k", "4",
            "--supervise-mode", "hop_ge2",
            "--loss-mode", "loss_lumen_shape",
            "--ckpt-metric", "hop_ge2_balanced",
            "--lumen-shape-weight", "2.5",
            "--mat-leg", "WC_v7_clot_phi_mse",
            "--init", $WallCkpt,
            "--out", $growthCkpt,
            "--anchors", $TrainAnchors
        )
        if ($Smoke) { $gArgs += "--cheap-val" }
        Invoke-PythonRcCheck -Label "train lumen-shape specialist" -PyArgs $gArgs
    }
    if (-not (Test-Path $growthCkpt)) { throw "Missing growth ckpt: $growthCkpt" }

    # Canonical A
    $env:SPECIES_TWO_MODEL_MODE = "0"
    Remove-Item Env:SPECIES_OFFWALL_MODEL_CKPT -ErrorAction SilentlyContinue
    Invoke-PythonRcCheck -Label "eval Arm A" -PyArgs @(
        "scripts/eval_mat_growth_simple.py",
        "--ckpt", $WallCkpt,
        "--mat-leg", "WC_v7_clot_phi_mse",
        "--no-baseline",
        "--out", $evalA,
        "--anchors", $TrainAnchors
    )
    # Compound wall route with lumen specialist
    Invoke-PythonRcCheck -Label "eval compound lumen specialist" -PyArgs @(
        "scripts/eval_mat_growth_simple.py",
        "--ckpt", $WallCkpt,
        "--mat-leg", "WC_v7_clot_phi_mse",
        "--no-baseline",
        "--out", $evalS,
        "--anchors", $TrainAnchors,
        "--offwall-ckpt", $growthCkpt,
        "--two-model-route", "wall"
    )
    Invoke-PythonRcCheck -Label "summarize A vs S" -PyArgs @(
        "scripts/summarize_wc_v7_compound_ab.py",
        "--arm-a", $evalA,
        "--arm-c", $evalS,
        "--out", (Join-Path $OutDir "compare_step2_lumen.json")
    )
}

# ---------------------------------------------------------------------------
# Step 3: isolate / skiphop (optional)
# ---------------------------------------------------------------------------
if ($do3) {
    Write-Host "[i] Step 3: isolate + skiphop controlled fins..." -ForegroundColor Cyan
    foreach ($leg in @("WC_v7_fw3_isolate", "WC_v7_fw3_skiphop")) {
        Invoke-MatLeg -Leg $leg
    }
}

Write-Host "[OK] firewall fix sequence done -> $OutDir" -ForegroundColor Green
Write-Host "[i] Success gates: wall clot F1 near A; hop_ge2 n_pred -> n_gt; hop_ge2 strict F1 up" -ForegroundColor DarkGray
