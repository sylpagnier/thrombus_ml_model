# Comparative hop-colored clot viz: GT | WC_v7 | growth_C + frontier hops=1.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_wc_v7_growth_c_frontier_h1_compare_viz.ps1
#   powershell ... -Anchors patient001,patient007,patient002

param(
    [string] $Anchors = "patient001,patient007,patient002",
    [string] $WallCkpt = "outputs/biochem/biochem_gnn/locked/species_gnn_best.pth",
    [string] $MatLeg = "WC_v7_clot_phi_mse",
    [string] $RunRoot = "outputs/biochem/offwall_model/wc_v7_wall_lumen_target_9h",
    [string] $GrowthCkpt = "",
    [string] $LabelA = "WC_v7",
    [string] $LabelB = "GrowthC_front_h1",
    [int] $MaxFrames = 6,
    [int] $FrontierHops = 1
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"
$env:SPECIES_CONTINUOUS_VEL_DECAY = "1"
$env:SPECIES_CONTINUOUS_VEL_DECAY_WALL_ONLY = "1"

if (-not $GrowthCkpt.Trim()) {
    $GrowthCkpt = Join-Path $RunRoot "growth_C/best.pth"
}
$GrowthPath = if ([System.IO.Path]::IsPathRooted($GrowthCkpt)) { $GrowthCkpt } else { Join-Path $RepoRoot $GrowthCkpt }
$WallPath = Join-Path $RepoRoot $WallCkpt
if (-not (Test-Path $GrowthPath)) {
    throw "Growth ckpt missing: $GrowthPath"
}
if (-not (Test-Path $WallPath)) {
    throw "Wall ckpt missing: $WallPath"
}

$OutRoot = Join-Path $RunRoot "viz_compare_wc_v7_vs_growth_c_front_h1"
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

$anchorList = @($Anchors.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })
Write-Host "[NEW] hop-compare viz GT | $LabelA | $LabelB ($($anchorList.Count) anchors)" -ForegroundColor Cyan
Write-Host "[i] wall=$WallCkpt growth=$GrowthCkpt route=frontier hops=$FrontierHops out=$OutRoot" -ForegroundColor DarkGray

foreach ($anc in $anchorList) {
    $outPng = Join-Path $OutRoot "Compare_GT_${LabelA}_${LabelB}_$anc.png"
    Write-Host "[viz] $anc -> $outPng" -ForegroundColor DarkGray
    $null = Invoke-PythonRcCheck -Label "compare-hop $anc" -PyArgs @(
        "scripts/viz_mat_growth_clot_compare_hop.py",
        "--anchor", $anc,
        "--ckpt", $WallCkpt,
        "--mat-leg", $MatLeg,
        "--offwall-ckpt", $GrowthCkpt,
        "--two-model-route", "frontier",
        "--two-model-frontier-hops", "$FrontierHops",
        "--label-a", $LabelA,
        "--label-b", $LabelB,
        "--max-frames", "$MaxFrames",
        "--out", $outPng
    )
}

Write-Host "[OK] compare viz done -> $OutRoot" -ForegroundColor Green
Write-Host "[i] Rows: GT | $LabelA | $LabelB ; color = clot hop from wall" -ForegroundColor DarkGray
