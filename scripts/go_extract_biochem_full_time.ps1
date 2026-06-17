# Re-extract biochem anchor graphs keeping the full COMSOL time horizon.
#
#   powershell -File .\scripts\go_extract_biochem_full_time.ps1

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "[NEW] Extracting biochem anchors (full COMSOL time horizon)..." -ForegroundColor Cyan
python -m src.data_gen.lib.extract_biochem_comsol_data
exit $LASTEXITCODE
