# S0 fullmesh viz (+ optional mask overlay). Does not train.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_deploy_s0_viz.ps1"
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_deploy_s0_viz.ps1" -MaskOverlay
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_deploy_s0_viz.ps1" -Anchor patient006 -TimeIndex -1

param(
    [string] $LegName = "s0_static_final",
    [string] $Anchor = "patient007",
    [int] $TimeIndex = -1,
    [string] $AnchorDir = "data/processed/graphs_biochem_anchors",
    [double] $ScatterSize = 6,
    [switch] $MaskOverlay
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")
. (Join-Path $PSScriptRoot "_clot_deploy_binary_base.ps1")

$env:CLOT_PHI_ANCHOR_DIR = ($AnchorDir -replace '\\', '/')

$ckpt = Join-Path $RepoRoot "outputs/biochem/clot_deploy/$LegName/clot_phi_best.pth"
if (-not (Test-Path $ckpt)) {
    Write-Host "[ERR] Missing checkpoint: $ckpt" -ForegroundColor Red
    exit 1
}

$vizDir = Join-Path $RepoRoot "outputs/biochem/viz/clot_deploy"
New-Item -ItemType Directory -Force -Path $vizDir | Out-Null

$tLabel = if ($TimeIndex -lt 0) { "tfinal" } else { "t$TimeIndex" }
$png = Join-Path $vizDir "${LegName}_${Anchor}_${tLabel}_fullmesh.png"

$vizArgs = @(
    "-m", "src.evaluation.viz_clot_phi_simple",
    "--anchor", $Anchor,
    "--checkpoint", $ckpt,
    "--time-index", "$TimeIndex",
    "--layout", "fullmesh",
    "--plot-mode", "scatter",
    "--scatter-size", "$ScatterSize",
    "--out", $png
)

if ($MaskOverlay) {
    $overlay = Join-Path $vizDir "${LegName}_${Anchor}_${tLabel}_masks.png"
    $vizArgs += @("--mask-overlay-out", $overlay)
}

Invoke-PythonRcCheck @vizArgs -Label "S0 viz"

Write-Host "[OK]  fullmesh -> $png" -ForegroundColor Green
if ($MaskOverlay) {
    Write-Host "[OK]  masks    -> $overlay" -ForegroundColor Green
}
