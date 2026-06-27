# Physics-guided triple ablation (fast or full), compared to one fast baseline.
#
# Legs:
#   G_dual_mat_neighbor_gate - dual-head Mat-only + neighbor commit-aware spatial gate
#   H_dual_mat_crit_focus    - dual-head Mat-only + crit-focused loss weighting
#   I_dual_fimat_fi_aux      - dual-head fi_mat with FI as light auxiliary target
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_mat_physics_triple_ablation.ps1 -Fast -Fresh
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_mat_physics_triple_ablation.ps1 -EvalOnly

param(
    [switch] $Fast,
    [switch] $Fresh,
    [switch] $EvalOnly,
    [string] $ValAnchor = "patient007"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

$RunRoot = "outputs/biochem/biochem_gnn/mat_physics_triple_ablation"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

$epochs = 50
$early = 35
$maxw = 0
if ($Fast) {
    $epochs = 10
    $early = 6
    $maxw = 16
}

$BaselineDir = "$RunRoot/baseline_fast/species"
$BaselineCkpt = "$BaselineDir/best.pth"

if ($Fresh) {
    Remove-Item -Force $BaselineCkpt -ErrorAction SilentlyContinue
    Remove-Item -Force "$BaselineDir/best.json" -ErrorAction SilentlyContinue
    Remove-Item -Force "$BaselineDir/train_log.jsonl" -ErrorAction SilentlyContinue
    Remove-Item -Force "$RunRoot/compare_*.json" -ErrorAction SilentlyContinue
}

Write-Host "[NEW] mat physics triple ablation (fast=$Fast) epochs=$epochs early_stop=$early max_windows=$maxw" -ForegroundColor Cyan

if (-not $EvalOnly) {
    Invoke-PythonRcCheck -Label "baseline_fast train" -PyArgs @(
        "-m", "src.training.train_species_pushforward_continuous",
        "--phase", "biochem_gnn",
        "--all-anchors",
        "--val-anchor", $ValAnchor,
        "--epochs", "$epochs",
        "--early-stop", "$early",
        "--max-windows", "$maxw",
        "--recipe", "default",
        "--init", "outputs/biochem/biochem_gnn/locked/species_gnn_best.pth",
        "--out", $BaselineCkpt
    )
}

$legs = @("G_dual_mat_neighbor_gate", "H_dual_mat_crit_focus", "I_dual_fimat_fi_aux")
foreach ($leg in $legs) {
    $legArgs = @(
        "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", (Join-Path $PSScriptRoot "go_mat_growth_simple.ps1"),
        "-Leg", $leg,
        "-Epochs", "$epochs",
        "-EarlyStop", "$early",
        "-MaxWindows", "$maxw",
        "-ValAnchor", $ValAnchor
    )
    if ($Fast) { $legArgs += "-Fast" }
    if ($Fresh) { $legArgs += "-Fresh" }
    if ($EvalOnly) { $legArgs += "-EvalOnly" }

    & powershell @legArgs
    if ($LASTEXITCODE -ne 0) { throw "$leg failed (exit=$LASTEXITCODE)" }

    $legCkpt = "outputs/biochem/biochem_gnn/mat_growth_ladder/$leg/species/best.pth"
    $cmp = "$RunRoot/compare_${leg}_vs_baseline_fast.json"
    Invoke-PythonRcCheck -Label "$leg vs baseline_fast" -PyArgs @(
        "scripts/eval_mat_growth_simple.py",
        "--ckpt", $legCkpt,
        "--baseline-ckpt", $BaselineCkpt,
        "--out", $cmp
    )
}

Write-Host "[OK] triple ablation compares saved under $RunRoot" -ForegroundColor Green
