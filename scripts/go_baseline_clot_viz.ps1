# Viz clot baseline (promoted manifest + dump anchors).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_baseline_clot_viz.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_baseline_clot_viz.ps1 -Anchor patient003 -TimeIndex 4

param(
    [string] $Anchor = "patient007",
    [int] $TimeIndex = -1,
    [string] $Manifest = "outputs/biochem/clot_baseline/manifest.json"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")
. (Join-Path $PSScriptRoot "_gnode_viz_helpers.ps1")

$manifestPath = Join-Path $RepoRoot ($Manifest -replace '/', '\')
if (-not (Test-Path $manifestPath)) {
    Write-Host "[ERR] Missing manifest: $Manifest (run go_baseline_clot.ps1 first)" -ForegroundColor Red
    exit 1
}

$json = Get-Content $manifestPath -Raw | ConvertFrom-Json
$dumpDir = $json.recipe.dump_anchor_dir
$ckpt = $json.recipe.clot_phi_ckpt

$env:CLOT_PHI_ANCHOR_DIR = $dumpDir
Invoke-ClotPhiScatterViz -Checkpoint $ckpt -Anchor $Anchor -TimeIndex $TimeIndex `
    -Out "outputs/biochem/viz/clot_baseline_${Anchor}_t${TimeIndex}.png"
Remove-Item Env:CLOT_PHI_ANCHOR_DIR -ErrorAction SilentlyContinue

Write-Host "[OK]  clot baseline viz written" -ForegroundColor Green
