# Head/scope A/B test (fast or full):
#   A: dual-head + Mat-only
#   B: single-head + Mat+FI
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_mat_head_scope_ab.ps1 -Fresh
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_mat_head_scope_ab.ps1 -Fast -Fresh
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_mat_head_scope_ab.ps1 -EvalOnly

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

$RunRoot = "outputs/biochem/biochem_gnn/mat_head_scope_ab"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

$epochs = 50
$early = 35
$maxw = 0
if ($Fast) {
    $epochs = 10
    $early = 6
    $maxw = 16
}

Write-Host "[NEW] mat head/scope A/B test (fast=$Fast) epochs=$epochs early_stop=$early max_windows=$maxw" -ForegroundColor Cyan

$dualLegArgs = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $PSScriptRoot "go_mat_growth_simple.ps1"),
    "-Leg", "E_dual_mat",
    "-Epochs", "$epochs",
    "-EarlyStop", "$early",
    "-MaxWindows", "$maxw",
    "-ValAnchor", $ValAnchor
)
if ($Fast) { $dualLegArgs += "-Fast" }
if ($Fresh) { $dualLegArgs += "-Fresh" }
if ($EvalOnly) { $dualLegArgs += "-EvalOnly" }
& powershell @dualLegArgs
if ($LASTEXITCODE -ne 0) { throw "E_dual_mat failed (exit=$LASTEXITCODE)" }

$singleLegArgs = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $PSScriptRoot "go_mat_growth_simple.ps1"),
    "-Leg", "F_single_fimat",
    "-Epochs", "$epochs",
    "-EarlyStop", "$early",
    "-MaxWindows", "$maxw",
    "-ValAnchor", $ValAnchor
)
if ($Fast) { $singleLegArgs += "-Fast" }
if ($Fresh) { $singleLegArgs += "-Fresh" }
if ($EvalOnly) { $singleLegArgs += "-EvalOnly" }
& powershell @singleLegArgs
if ($LASTEXITCODE -ne 0) { throw "F_single_fimat failed (exit=$LASTEXITCODE)" }

# Compare B (simple) directly against A (baseline-ckpt argument).
$CompareJson = "$RunRoot/compare_single_fimat_vs_dual_mat.json"
Invoke-PythonRcCheck -Label "head/scope A/B compare" -PyArgs @(
    "scripts/eval_mat_growth_simple.py",
    "--ckpt", "outputs/biochem/biochem_gnn/mat_growth_ladder/F_single_fimat/species/best.pth",
    "--baseline-ckpt", "outputs/biochem/biochem_gnn/mat_growth_ladder/E_dual_mat/species/best.pth",
    "--out", $CompareJson
)

Write-Host "[OK] A/B compare -> $CompareJson" -ForegroundColor Green
