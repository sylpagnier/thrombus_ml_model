# Precision sweep: in-training levers vs one fast dual fi_mat baseline.
#
# Hypothesis (docs/SPECIES_LEARNING_STRATEGY.md s6.13): geometry+kine is near its deployable
# ranking ceiling (LOAO AUC ~0.90), so the remaining clot_f1 gains must come from in-training
# structure, not more signal. Each leg flips ONE lever on the fi_mat baseline so the delta is
# attributable:
#
#   K_fimat_neighbor_gate      - keep the autocatalytic neighbour coupling, but on the dual
#                                fi_mat head (the user's "right instinct, wrong execution" fix
#                                for G, which used Mat-only).
#   L_fimat_geom_rich          - enrich the static geometry context beyond leg C's 3 channels
#                                with the 2-hop expansion / curvature commit-vs-eligible
#                                discriminators (SPECIES_GEOM_FEATS_RICH).
#   M_fimat_neighbor_geom_rich - combine the two surviving levers (neighbour gate + rich geom).
#   N_mat_geom_rich            - control: rich geometry on the Mat-only scope (vs leg C).
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_precision_sweep.ps1 -Fast -Fresh
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_precision_sweep.ps1 -EvalOnly

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

$RunRoot = "outputs/biochem/biochem_gnn/precision_sweep"
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

Write-Host "[NEW] precision sweep (fast=$Fast) epochs=$epochs early_stop=$early max_windows=$maxw" -ForegroundColor Cyan

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

$legs = @(
    "K_fimat_neighbor_gate",
    "L_fimat_geom_rich",
    "M_fimat_neighbor_geom_rich",
    "N_mat_geom_rich"
)
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

# Roll the per-leg compares into one ranked table.
Invoke-PythonRcCheck -Label "precision sweep summary" -PyArgs @(
    "scripts/summarize_precision_sweep.py",
    "--run-root", $RunRoot
)

Write-Host "[OK] precision sweep compares saved under $RunRoot" -ForegroundColor Green
