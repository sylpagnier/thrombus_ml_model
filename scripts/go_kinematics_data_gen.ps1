param(
  [int]$NumVessels = 100,
  [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$env:PYTHONPATH = "."
$env:PYTHONIOENCODING = "utf-8"

# 1. Vessels
Write-Host "Generating straight_max vessels..." -ForegroundColor Cyan
$vesselArgs = @("--phase", "1", "--level", "0", "-n", "$NumVessels", "--pathology-mode", "straight_max")
if ($Overwrite) {
    $vesselArgs += "--overwrite"
}
& python -m src.data_gen.lib.vessel_generator @vesselArgs

# 2. Anchors (COMSOL)
Write-Host "Generating COMSOL anchors..." -ForegroundColor Cyan
$anchorArgs = @("--phase", "1", "--rheology", "newtonian")
# anchor_generator automatically processes new meshes
& python -m src.data_gen.lib.anchor_generator @anchorArgs

# 3. Mesh to Graph (PyG)
Write-Host "Converting meshes to PyG graphs..." -ForegroundColor Cyan
$meshArgs = @("--phase", "1", "--rheology", "newtonian")
& python -m src.data_gen.lib.mesh_to_graph @meshArgs

Write-Host "Done!" -ForegroundColor Green
