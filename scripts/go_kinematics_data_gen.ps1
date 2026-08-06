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
& python src/data_gen/lib/vessel_generator.py @vesselArgs

# 2. Anchors (COMSOL)
Write-Host "Generating COMSOL anchors..." -ForegroundColor Cyan
$ow_str = if ($Overwrite) { "True" } else { "False" }
$pyScript = @"
from src.data_gen.lib.anchor_generator import AnchorGenerator
gen = AnchorGenerator(phase='phase1')
allow_overwrite = $ow_str
gen.run_batch(max_new=$NumVessels, allow_overwrite=allow_overwrite)
"@
& python -c $pyScript

# 3. Mesh to Graph (PyG)
Write-Host "Converting meshes to PyG graphs..." -ForegroundColor Cyan
$meshArgs = @("--phase", "1", "--rheology", "newtonian")
& python src/data_gen/lib/mesh_to_graph.py @meshArgs

Write-Host "Done!" -ForegroundColor Green
