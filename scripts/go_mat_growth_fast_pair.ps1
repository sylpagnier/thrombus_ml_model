# Fast apples-to-apples pair run:
#   1) baseline-like triangle6 wall+3hop species leg (dual-head fi_mat)
#   2) parity single-head Mat-only leg (baseline dynamics preserved)
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_mat_growth_fast_pair.ps1 -Fresh

param(
    [string] $ValAnchor = "patient007",
    [switch] $Fresh,
    [switch] $EvalOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

# Fixed apples-to-apples fast defaults.
$EPOCHS = 10
$EARLY_STOP = 6
$MAX_WINDOWS = 16

$RunRoot = "outputs/biochem/biochem_gnn/mat_growth_fast_pair"
$BaselineCkpt = "$RunRoot/baseline_fast/species/best.pth"
$ParityCkpt = "outputs/biochem/biochem_gnn/mat_growth_ladder/D_parity_single/species/best.pth"
$PairCompare = "$RunRoot/pair_compare.json"

if ($Fresh) {
    Remove-Item -Force $BaselineCkpt -ErrorAction SilentlyContinue
    Remove-Item -Force $ParityCkpt -ErrorAction SilentlyContinue
    Remove-Item -Force "$RunRoot/baseline_fast/species/best.json" -ErrorAction SilentlyContinue
    Remove-Item -Force "$RunRoot/baseline_fast/species/train_log.jsonl" -ErrorAction SilentlyContinue
    Remove-Item -Force "$RunRoot/D_parity_single/species/best.json" -ErrorAction SilentlyContinue
    Remove-Item -Force "$RunRoot/D_parity_single/species/train_log.jsonl" -ErrorAction SilentlyContinue
    Remove-Item -Force $PairCompare -ErrorAction SilentlyContinue
}

Write-Host "[NEW] fast pair: baseline_fast vs D_parity_single" -ForegroundColor Cyan
Write-Host "[i] fixed fast defaults: epochs=$EPOCHS early_stop=$EARLY_STOP max_windows=$MAX_WINDOWS all_anchors=1" -ForegroundColor DarkGray

if (-not $EvalOnly) {
    # 1) Fast baseline-like species leg (triangle6 wall+3hop dynamics, dual-head fi_mat).
    Invoke-PythonRcCheck -Label "baseline_fast train" -PyArgs @(
        "-m", "src.training.train_species_pushforward_continuous",
        "--phase", "biochem_gnn",
        "--all-anchors",
        "--val-anchor", $ValAnchor,
        "--epochs", "$EPOCHS",
        "--early-stop", "$EARLY_STOP",
        "--max-windows", "$MAX_WINDOWS",
        "--recipe", "default",
        "--init", "outputs/biochem/biochem_gnn/locked/species_gnn_best.pth",
        "--out", $BaselineCkpt
    )

# 2) Fast parity single-head Mat-only leg (same fast defaults).
#
# NOTE: go_mat_growth_simple.ps1 writes leg checkpoints under mat_growth_ladder/<leg>/...
# so this pair runner reads D_parity_single from that canonical ladder location.
    $legArgs = @(
        "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", (Join-Path $PSScriptRoot "go_mat_growth_simple.ps1"),
        "-Leg", "D_parity_single",
        "-Fast",
        "-Epochs", "$EPOCHS",
        "-EarlyStop", "$EARLY_STOP",
        "-MaxWindows", "$MAX_WINDOWS",
        "-ValAnchor", $ValAnchor
    )
    if ($Fresh) { $legArgs += "-Fresh" }
    & powershell @legArgs
    if ($LASTEXITCODE -ne 0) { throw "D_parity_single leg failed (exit=$LASTEXITCODE)" }
}

# Pair compare: parity simple ckpt against fast baseline ckpt.
Invoke-PythonRcCheck -Label "fast pair compare" -PyArgs @(
    "scripts/eval_mat_growth_simple.py",
    "--ckpt", $ParityCkpt,
    "--baseline-ckpt", $BaselineCkpt,
    "--out", $PairCompare
)

Write-Host "[OK] fast pair compare -> $PairCompare" -ForegroundColor Green
