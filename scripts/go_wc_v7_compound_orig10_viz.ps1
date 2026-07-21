# Hop-colored GT | pred | error ladder viz for orig10 compound A/B/C.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_wc_v7_compound_orig10_viz.ps1
#   powershell ... -Anchors patient001,patient007,patient004

param(
    [string] $Anchors = "patient001,patient007,patient004",
    [string] $WallCkpt = "outputs/biochem/biochem_gnn/locked/species_gnn_best.pth",
    [string] $MatLeg = "WC_v7_clot_phi_mse",
    [string] $RunRoot = "outputs/biochem/offwall_model/wc_v7_compound_abc_orig10_9h",
    [int] $MaxFrames = 6
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

$OutRoot = Join-Path $RunRoot "viz_hop_ladder"
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

$GrowthB = Join-Path $RunRoot "growth_B_frontier_blurring_prec/best.pth"
$GrowthC = Join-Path $RunRoot "growth_C_offwall_blurring_prec/best.pth"

$anchorList = @($Anchors.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$arms = @(
    @{
        Label = "Arm_A_canonical"
        Args  = @(
            "--ckpt", $WallCkpt,
            "--mat-leg", $MatLeg,
            "--arm-label", "Arm_A_canonical"
        )
    },
    @{
        Label = "Arm_B_compound_frontier"
        Args  = @(
            "--ckpt", $WallCkpt,
            "--mat-leg", $MatLeg,
            "--offwall-ckpt", $GrowthB,
            "--two-model-route", "frontier",
            "--two-model-frontier-hops", "2",
            "--arm-label", "Arm_B_compound_frontier"
        )
    },
    @{
        Label = "Arm_C_compound_wall"
        Args  = @(
            "--ckpt", $WallCkpt,
            "--mat-leg", $MatLeg,
            "--offwall-ckpt", $GrowthC,
            "--two-model-route", "wall",
            "--two-model-frontier-hops", "2",
            "--arm-label", "Arm_C_compound_wall"
        )
    }
)

Write-Host "[NEW] orig10 compound hop-ladder viz ($($arms.Count) arms x $($anchorList.Count) anchors)" -ForegroundColor Cyan
Write-Host "[i] out=$OutRoot max_frames=$MaxFrames" -ForegroundColor DarkGray

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
