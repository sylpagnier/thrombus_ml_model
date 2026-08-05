# Wall-generalization probe on the tight straight-vessel family of 7 (2026-07-29).
#
# Question: can the WC_v7 wall backbone generalize to a near-identical vessel it has
# never seen, when trained only on geometrically similar vessels?
#
# Cohort (all straightish: curvature 0.03-0.06, stenosis 1.11-1.45, expansion 1.05-1.28)
#   clot-rich  : 005, 006, 010 (train) | 020 (held out -> the generalization metric)
#   clot-free  : 023, 002      (train) | 034 (held out -> false-positive gate)
#
# Clot-free vessels are scored by the graded empty-GT score (predicting nothing == 1.0,
# decaying with false positives) -- see docs/GENERALIZATION_PLAN.md s2c.
#
# Baseline conditioning ON PURPOSE: no drop-xy / geom-rich yet, so this run measures the
# current wall's in-family generalization without confounds. Those are the follow-up A/B.
#
# LEAKAGE: the locked WC_v7 wall was trained via go_mat_growth_simple.ps1, which always
# passes --all-anchors (the -AllAnchors branch is a no-op duplicate), so it has already
# seen every anchor -- including this cohort's held-outs. Warm-starting from it therefore
# leaks 020/034 into the initial weights. Default is COLD (-WarmStart off) so the
# generalization number is honest; pass -WarmStart only for a deliberate fine-tune arm.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_wall_family7_probe.ps1 -Fresh
#   powershell ... -Fresh -WarmStart    # leaky fine-tune reference arm
#   powershell ... -EvalOnly

param(
    [int]    $Epochs      = 30,
    [int]    $EarlyStop   = 10,
    [int]    $MaxWindows  = 24,
    [string] $TrainAnchors = "patient005,patient006,patient010,patient023,patient002",
    [string] $ValAnchor    = "patient020",
    [string] $HoldoutAnchors = "patient020,patient034",
    [string] $WallInit   = "outputs/biochem/biochem_gnn/locked/species_gnn_best.pth",
    [string] $RunRoot    = "outputs/biochem/eda/wall_family7",
    [switch] $Fresh,
    [switch] $WarmStart,
    [switch] $EvalOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

$OutDir = Join-Path $RepoRoot $RunRoot
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$arm  = if ($WarmStart) { "warm" } else { "cold" }
$Ckpt = Join-Path $OutDir "wall_family7_$arm.pth"

Write-Host "[NEW] wall_family7 probe arm=$arm epochs=$Epochs max_windows=$MaxWindows val=$ValAnchor" -ForegroundColor Cyan
Write-Host "[i] train   = $TrainAnchors" -ForegroundColor DarkGray
Write-Host "[i] holdout = $HoldoutAnchors (020=clot-rich metric, 034=clot-free FP gate)" -ForegroundColor DarkGray
if ($WarmStart) {
    Write-Host "[WARN] warm-start arm: locked wall saw 020/034 -> holdout score is NOT leak-free" -ForegroundColor Yellow
} else {
    Write-Host "[i] cold init (--no-init): no locked-wall weights, holdout is leak-free" -ForegroundColor DarkGray
}

if (-not $EvalOnly) {
    if ($Fresh) { Remove-Item -Force $Ckpt -ErrorAction SilentlyContinue }
    $trainArgs = @(
        "-m", "src.training.train_species_pushforward_continuous",
        "--phase", "biochem_gnn",
        "--recipe", "mat_growth_simple",
        "--leg", "WC_v7_clot_phi_mse",
        "--anchors", $TrainAnchors,
        "--val-anchor", $ValAnchor,
        "--exclude-val-from-train",
        "--epochs", "$Epochs",
        "--early-stop", "$EarlyStop",
        "--max-windows", "$MaxWindows",
        "--out", $Ckpt
    )
    # Bare --init falls back to a default all-anchor ckpt, so cold requires explicit --no-init.
    if ($WarmStart) { $trainArgs += @("--init-mode", "backbone", "--init", $WallInit) }
    else            { $trainArgs += "--no-init" }
    $null = Invoke-PythonRcCheck -Label "wall_family7 $arm train" -PyArgs $trainArgs
}

if (-not (Test-Path $Ckpt)) { throw "wall_family7 ckpt missing: $Ckpt" }

# Wall-only score on the held-outs (no growth specialist: wall is the ceiling under test).
$holdJson = Join-Path $OutDir "eval_holdout_$arm.json"
$null = Invoke-PythonRcCheck -Label "wall_family7 $arm holdout eval" -PyArgs @(
    "scripts/eval_mat_growth_simple.py",
    "--ckpt", $Ckpt,
    "--mat-leg", "WC_v7_clot_phi_mse",
    "--no-baseline",
    "--anchors", $HoldoutAnchors,
    "--out", $holdJson
)

Write-Host "[OK] wall_family7 done -> $OutDir" -ForegroundColor Green
Write-Host "[i] holdout: $holdJson" -ForegroundColor DarkGray
