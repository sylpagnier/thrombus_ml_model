# Re-extract biochem anchor graphs with full COMSOL time horizon.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_extract_biochem_anchors_full_time.ps1

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "[NEW] Extracting biochem anchors (full COMSOL time horizon)..." -ForegroundColor Cyan
python -m src.data_gen.lib.extract_biochem_comsol_data
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[OK]  Graphs -> data/processed/graphs_biochem_anchors/" -ForegroundColor Green
