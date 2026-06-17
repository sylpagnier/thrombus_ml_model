# Clot ladder viz for species GNN deploy stack (manifest + LOAO picks).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_species_gnn_deploy_viz.ps1"
#   powershell ... -Anchors patient007 -Flow kinematics

param(
    [string] $Manifest = "data/reference/species_gnn_deploy_baseline.json",
    [string] $Anchors = "patient007,patient001,patient004",
    [ValidateSet("gt", "kinematics", "both")]
    [string] $Flow = "gt",
    [int] $MaxFrames = 10
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

$flows = if ($Flow -eq "both") { @("gt", "kinematics") } else { @($Flow) }
foreach ($anc in ($Anchors -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })) {
    foreach ($fl in $flows) {
        Write-Host "[NEW] viz $anc flow=$fl" -ForegroundColor Cyan
        Invoke-PythonRcCheck -Label "viz $anc $fl" -PyArgs @(
            "scripts/viz_species_gnn_deploy.py",
            "--anchor", $anc,
            "--manifest", $Manifest,
            "--flow", $fl,
            "--max-frames", "$MaxFrames"
        )
    }
}
Write-Host "[OK] outputs/biochem/viz/species_gnn_deploy/" -ForegroundColor Green
