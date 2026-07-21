# Viz promoted Mat-growth winner (W) + optional anchors.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_viz_mat_w_winner.ps1
#   powershell ... -Anchor patient003
#   powershell ... -Leg WC_mat_flow_dynamic

param(
    [string] $Leg = "W_mat_flow_stagnation",
    [string] $Anchor = "patient007",
    [string] $Ckpt = "",
    [int] $MaxFrames = 10
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

$OutDir = "outputs/biochem/viz/mat_growth"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$pyArgs = @(
    "scripts/viz_mat_growth_clot_ladder.py",
    "--leg", $Leg,
    "--anchor", $Anchor,
    "--max-frames", "$MaxFrames",
    "--out", "$OutDir/clot_ladder_${Leg}_${Anchor}.png"
)
if ($Ckpt.Trim()) { $pyArgs += @("--ckpt", $Ckpt.Trim()) }

Write-Host "[NEW] mat growth clot viz: leg=$Leg anchor=$Anchor" -ForegroundColor Cyan
Invoke-PythonRcCheck -Label "mat growth clot viz" -PyArgs $pyArgs
Write-Host "[OK] see $OutDir" -ForegroundColor Green
