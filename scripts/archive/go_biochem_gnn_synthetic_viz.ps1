# biochem_gnn clot timeline on a synthetic vessel (deploy, no GT).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_biochem_gnn_synthetic_viz.ps1
#   powershell ... -Seed 7 -Regenerate
#   powershell ... -Flow coupled

param(
    [int] $Seed = 42,
    [int] $Level = 1,
    [int] $MaxFrames = 12,
    [string] $Manifest = "",
    [ValidateSet("frozen_kine", "coupled", "gt")]
    [string] $Flow = "frozen_kine",
    [switch] $Regenerate,
    [switch] $NoInc40,
    [switch] $NoS0  # legacy alias for -NoInc40
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

if (-not $Manifest.Trim()) {
    $Manifest = "data/reference/biochem_gnn_baseline.json"
    if (-not (Test-Path (Join-Path $RepoRoot $Manifest))) {
        $Manifest = "data/reference/species_gnn_deploy_baseline.json"
    }
}

$pyArgs = @(
    "scripts/viz_biochem_gnn_timeline.py",
    "--seed", "$Seed",
    "--level", "$Level",
    "--flow", $Flow,
    "--max-frames", "$MaxFrames",
    "--manifest", $Manifest
)
if ($Regenerate) { $pyArgs += "--regenerate" }
if ($NoInc40 -or $NoS0) { $pyArgs += "--no-inc40" }

Write-Host "[NEW] biochem_gnn synthetic timeline viz seed=$Seed flow=$Flow" -ForegroundColor Cyan
Invoke-PythonRcCheck -Label "biochem_gnn synthetic viz" -PyArgs $pyArgs
Write-Host "[OK] outputs/biochem/viz/biochem_gnn/" -ForegroundColor Green
