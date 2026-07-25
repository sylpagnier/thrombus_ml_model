# Hop-colored GT | pred | error viz for tile-CC explore 2h.
# Arms: locked WC_v7 (A) | UnionTile compound | PerComponent compound.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_wc_v7_tile_cc_explore_viz.ps1
#   powershell ... -Anchors patient007,patient004,patient001
#

param(
    [string] $Anchors = "patient007,patient004",
    [string] $WallCkpt = "outputs/biochem/biochem_gnn/locked/species_gnn_best.pth",
    [string] $MatLeg = "WC_v7_clot_phi_mse",
    [string] $RunRoot = "outputs/biochem/offwall_model/wc_v7_tile_cc_explore_2h",
    [string] $UnionCkpt = "",
    [string] $PerComponentCkpt = "",
    [int] $MaxFrames = 6
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

if (-not $UnionCkpt.Trim()) {
    $UnionCkpt = Join-Path $RunRoot "growth_UnionTile/best.pth"
}
if (-not $PerComponentCkpt.Trim()) {
    $PerComponentCkpt = Join-Path $RunRoot "growth_PerComponent/best.pth"
}

function Resolve-RepoPath([string] $p) {
    if ([System.IO.Path]::IsPathRooted($p)) { return $p }
    return (Join-Path $RepoRoot $p)
}

$UnionPath = Resolve-RepoPath $UnionCkpt
$CcPath = Resolve-RepoPath $PerComponentCkpt
$WallPath = Resolve-RepoPath $WallCkpt
foreach ($pair in @(
    @{ N = "UnionTile growth"; P = $UnionPath },
    @{ N = "PerComponent growth"; P = $CcPath },
    @{ N = "Wall"; P = $WallPath }
)) {
    if (-not (Test-Path $pair.P)) {
        throw "$($pair.N) ckpt missing: $($pair.P)"
    }
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
        Label = "Arm_UnionTile"
        Args  = @(
            "--ckpt", $WallCkpt,
            "--mat-leg", $MatLeg,
            "--offwall-ckpt", $UnionCkpt,
            "--two-model-route", "wall",
            "--two-model-frontier-hops", "2",
            "--arm-label", "Arm_UnionTile"
        )
    },
    @{
        Label = "Arm_PerComponent"
        Args  = @(
            "--ckpt", $WallCkpt,
            "--mat-leg", $MatLeg,
            "--offwall-ckpt", $PerComponentCkpt,
            "--two-model-route", "wall",
            "--two-model-frontier-hops", "2",
            "--arm-label", "Arm_PerComponent"
        )
    }
)

Write-Host "[NEW] tile-CC explore hop-ladder viz ($($arms.Count) arms x $($anchorList.Count) anchors)" -ForegroundColor Cyan
Write-Host "[i] wall=$WallCkpt union=$UnionCkpt cc=$PerComponentCkpt out=$OutRoot" -ForegroundColor DarkGray

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
        $null = Invoke-PythonRcCheck -Label "viz $($arm.Label) $anc" -PyArgs $pyArgs
    }
}

Write-Host "[OK] viz done -> $OutRoot" -ForegroundColor Green
Write-Host "[i] Compare Arm_A_WC_v7_* vs Arm_UnionTile_* vs Arm_PerComponent_* (same style as FrontierLumen / ge2 viz)" -ForegroundColor DarkGray
Write-Host "[i] Past viz dirs for side-by-side: wc_v7_frontier_lumen_6h/viz_hop_ladder , wc_v7_compound_abc_orig10_9h/viz_hop_ladder" -ForegroundColor DarkGray
