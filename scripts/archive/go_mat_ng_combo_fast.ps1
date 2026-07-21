# Quick fast run: O = N + G (dual Mat-only + neighbour gate + rich geometry).
# Compares deploy metrics vs the existing N and G fast ckpts when present.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_mat_ng_combo_fast.ps1 -Fresh
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_mat_ng_combo_fast.ps1 -EvalOnly

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

$Leg = "O_mat_neighbor_geom_rich"
$RunRoot = "outputs/biochem/biochem_gnn/mat_ng_combo_fast"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

$legArgs = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $PSScriptRoot "go_mat_growth_simple.ps1"),
    "-Leg", $Leg,
    "-Fast",
    "-ValAnchor", $ValAnchor
)
if ($Fresh) { $legArgs += "-Fresh" }
if ($EvalOnly) { $legArgs += "-EvalOnly" }

Write-Host "[NEW] N+G combo fast leg=$Leg" -ForegroundColor Cyan
& powershell @legArgs
if ($LASTEXITCODE -ne 0) { throw "$Leg train/eval failed (exit=$LASTEXITCODE)" }

$OCkpt = "outputs/biochem/biochem_gnn/mat_growth_ladder/$Leg/species/best.pth"
$refs = @(
    @{ name = "N_mat_geom_rich"; ckpt = "outputs/biochem/biochem_gnn/mat_growth_ladder/N_mat_geom_rich/species/best.pth" },
    @{ name = "G_dual_mat_neighbor_gate"; ckpt = "outputs/biochem/biochem_gnn/mat_growth_ladder/G_dual_mat_neighbor_gate/species/best.pth" }
)
foreach ($ref in $refs) {
    if (-not (Test-Path $ref.ckpt)) {
        Write-Host "[skip] compare vs $($ref.name): missing $($ref.ckpt)" -ForegroundColor DarkGray
        continue
    }
    $cmp = "$RunRoot/compare_O_vs_$($ref.name).json"
    Invoke-PythonRcCheck -Label "O vs $($ref.name)" -PyArgs @(
        "scripts/eval_mat_growth_simple.py",
        "--ckpt", $OCkpt,
        "--baseline-ckpt", $ref.ckpt,
        "--out", $cmp
    )
}

Write-Host "[OK] O ckpt=$OCkpt compares under $RunRoot" -ForegroundColor Green
