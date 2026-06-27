# Quick G+H combo test (neighbor gate + crit-focused loss).
#
# Trains J_dual_mat_neighbor_crit and compares against:
#   - baseline_fast from mat_physics_triple_ablation (if present)
#   - existing G and H ladder ckpts (if present)
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_mat_gh_combo_fast.ps1 -Fresh
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_mat_gh_combo_fast.ps1 -EvalOnly

param(
    [switch] $Fresh,
    [switch] $EvalOnly,
    [string] $ValAnchor = "patient007"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

$RunRoot = "outputs/biochem/biochem_gnn/mat_gh_combo_fast"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

$Leg = "J_dual_mat_neighbor_crit"
$LegCkpt = "outputs/biochem/biochem_gnn/mat_growth_ladder/$Leg/species/best.pth"
$BaselineCkpt = "outputs/biochem/biochem_gnn/mat_physics_triple_ablation/baseline_fast/species/best.pth"
$GCkpt = "outputs/biochem/biochem_gnn/mat_growth_ladder/G_dual_mat_neighbor_gate/species/best.pth"
$HCkpt = "outputs/biochem/biochem_gnn/mat_growth_ladder/H_dual_mat_crit_focus/species/best.pth"

$epochs = 10
$early = 6
$maxw = 16

Write-Host "[NEW] G+H combo fast test leg=$Leg epochs=$epochs early_stop=$early max_windows=$maxw" -ForegroundColor Cyan

if (-not $EvalOnly) {
    $legArgs = @(
        "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", (Join-Path $PSScriptRoot "go_mat_growth_simple.ps1"),
        "-Leg", $Leg,
        "-Fast",
        "-Epochs", "$epochs",
        "-EarlyStop", "$early",
        "-MaxWindows", "$maxw",
        "-ValAnchor", $ValAnchor
    )
    if ($Fresh) { $legArgs += "-Fresh" }
    & powershell @legArgs
    if ($LASTEXITCODE -ne 0) { throw "$Leg train failed (exit=$LASTEXITCODE)" }
}

if (-not (Test-Path $LegCkpt)) {
    throw "missing leg ckpt: $LegCkpt"
}

function Compare-Leg {
    param(
        [string] $Label,
        [string] $BaselinePath,
        [string] $OutJson
    )
    if (-not (Test-Path $BaselinePath)) {
        Write-Host "[skip] $Label baseline missing: $BaselinePath" -ForegroundColor DarkYellow
        return
    }
    Invoke-PythonRcCheck -Label $Label -PyArgs @(
        "scripts/eval_mat_growth_simple.py",
        "--ckpt", $LegCkpt,
        "--baseline-ckpt", $BaselinePath,
        "--out", $OutJson
    )
}

Compare-Leg -Label "J vs baseline_fast" -BaselinePath $BaselineCkpt -OutJson "$RunRoot/compare_J_vs_baseline_fast.json"
Compare-Leg -Label "J vs G" -BaselinePath $GCkpt -OutJson "$RunRoot/compare_J_vs_G.json"
Compare-Leg -Label "J vs H" -BaselinePath $HCkpt -OutJson "$RunRoot/compare_J_vs_H.json"

Write-Host "[OK] G+H combo compares saved under $RunRoot" -ForegroundColor Green
