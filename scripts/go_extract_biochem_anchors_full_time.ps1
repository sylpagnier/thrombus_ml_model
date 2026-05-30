# Re-extract biochem anchor graphs with full COMSOL time horizon (no BIOCHEM_T_MAX cap).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_extract_biochem_anchors_full_time.ps1

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$env:BIOCHEM_EXTRACT_FULL_TIME = "1"
Remove-Item Env:BIOCHEM_EXTRACT_T_MAX_S -ErrorAction SilentlyContinue

Write-Host "[NEW] Extracting biochem anchors (full COMSOL time horizon)..." -ForegroundColor Cyan
python -m src.data_gen.lib.extract_biochem_comsol_data
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[OK]  Graphs -> data/processed/graphs_biochem_anchors/" -ForegroundColor Green
