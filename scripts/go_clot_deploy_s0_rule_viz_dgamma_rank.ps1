# Viz refined winner with dgamma-slice rank pool (ceiling intersect -dgamma/dx band).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_deploy_s0_rule_viz_dgamma_rank.ps1"
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_deploy_s0_rule_viz_dgamma_rank.ps1" -MaskOverlay

param(
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
. (Join-Path $PSScriptRoot "_clot_prior_rule_refined_dgamma_rank_env.ps1")
$env:BIOCHEM_PRIOR_COMSOL_ALIGNED = "1"
$env:BIOCHEM_PRIOR_NORM_MASK = "adjacent"

$vizDir = Join-Path $RepoRoot "outputs/biochem/viz/clot_deploy"
New-Item -ItemType Directory -Force -Path $vizDir | Out-Null

$tLabel = if ($TimeIndex -lt 0) { "tfinal" } else { "t$TimeIndex" }
$png = Join-Path $vizDir "prior_rule_dgamma_rank_${Anchor}_${tLabel}_fullmesh.png"

$pyArgs = @(
    "scripts/viz_clot_prior_rule_baseline.py",
    "--anchor", $Anchor,
    "--anchor-dir", $AnchorDir,
    "--time-index", "$TimeIndex",
    "--scatter-size", "$ScatterSize",
    "--out", $png
)

if ($MaskOverlay) {
    $overlay = Join-Path $vizDir "prior_rule_dgamma_rank_${Anchor}_${tLabel}_masks.png"
    $pyArgs += @("--mask-overlay-out", $overlay)
}

Write-Host "[NEW] dgamma-rank prior rule viz anchor=$Anchor" -ForegroundColor Cyan
Invoke-PythonRcCheck @pyArgs -Label "dgamma rank prior rule viz"

Write-Host "[OK]  fullmesh -> $png" -ForegroundColor Green
if ($MaskOverlay) {
    Write-Host "[OK]  masks    -> $overlay" -ForegroundColor Green
}
