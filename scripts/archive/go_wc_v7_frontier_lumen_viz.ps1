# Hop-colored GT | pred | error viz: locked WC_v7 (A) vs WC_v7+FrontierLumen compound (S).
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_wc_v7_frontier_lumen_viz.ps1
#   powershell ... -Anchors patient007,patient004,patient001
#

param(
    [string] $Anchors = "patient007,patient004,patient001",
    [string] $WallCkpt = "outputs/biochem/biochem_gnn/locked/species_gnn_best.pth",
    [string] $MatLeg = "WC_v7_clot_phi_mse",
    [string] $RunRoot = "outputs/biochem/offwall_model/wc_v7_frontier_lumen_6h",
    [string] $GrowthCkpt = "",
    [int] $MaxFrames = 6
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

if (-not $GrowthCkpt.Trim()) {
    $GrowthCkpt = Join-Path $RunRoot "growth_frontier_lumen/best.pth"
}
$GrowthPath = if ([System.IO.Path]::IsPathRooted($GrowthCkpt)) { $GrowthCkpt } else { Join-Path $RepoRoot $GrowthCkpt }
if (-not (Test-Path $GrowthPath)) {
    throw "FrontierLumen growth ckpt missing: $GrowthPath"
}
$WallPath = Join-Path $RepoRoot $WallCkpt
if (-not (Test-Path $WallPath)) {
    throw "Wall ckpt missing: $WallPath"
}

$OutRoot = Join-Path $RunRoot "viz_hop_ladder"
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

$anchorList = @($Anchors.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$arms = @(
    @{
        Label = "Arm_A_WC_v7"
        Args  = @(
            "--ckpt", $WallCkpt,
            "--mat-leg", $MatLeg,
            "--arm-label", "Arm_A_WC_v7"
        )
    },
    @{
        Label = "Arm_S_WC_v7_FrontierLumen"
        Args  = @(
            "--ckpt", $WallCkpt,
            "--mat-leg", $MatLeg,
            "--offwall-ckpt", $GrowthCkpt,
            "--two-model-route", "wall",
            "--two-model-frontier-hops", "2",
            "--arm-label", "Arm_S_WC_v7_FrontierLumen"
        )
    }
)

Write-Host "[NEW] FrontierLumen hop-ladder viz ($($arms.Count) arms x $($anchorList.Count) anchors)" -ForegroundColor Cyan
Write-Host "[i] wall=$WallCkpt growth=$GrowthCkpt out=$OutRoot" -ForegroundColor DarkGray

foreach ($arm in $arms) {
    foreach ($anc in $anchorList) {
        $outPng = Join-Path $OutRoot "$($arm.Label)_$anc.png"
        Write-Host "[viz] $($arm.Label) $anc -> $outPng" -ForegroundColor DarkGray
        $pyArgs = @(
            "scripts/viz_mat_growth_clot_ladder.py",
            "--anchor", $anc,
            "--max-frames", "$MaxFrames",
            "--out", $outPng
        ) + $arm.Args
        Invoke-PythonRcCheck -Label "viz $($arm.Label) $anc" -PyArgs $pyArgs
    }
}

Write-Host "[OK] viz done -> $OutRoot" -ForegroundColor Green
Write-Host "[i] Compare Arm_A_WC_v7_* vs Arm_S_WC_v7_FrontierLumen_* (hop-colored GT|pred|error)" -ForegroundColor DarkGray
