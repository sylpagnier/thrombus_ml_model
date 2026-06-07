# Interactive viz: oracle gt_clot vs deploy neighbor MLP commit masks (time slider).
#
# Opens visualize_pipeline with a second figure:
#   left  = Oracle commit (gt_clot, GT labels -- eval upper bound)
#   right = Active commit (gt_clot for Leg B, neighbor for Leg B_deploy)
# Scrub Time idx on the Flow window to compare masks over rollout.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_mlp_commit_mask_viz.ps1
#   powershell ... -Anchor patient007 -Leg B_deploy -Fast
#   powershell ... -Leg Both   # B then B_deploy, close each window to advance

param(
    [ValidateSet("B", "B_deploy", "Both")]
    [string] $Leg = "Both",
    [string] $TeacherCheckpoint = "outputs\biochem\clot_baseline\teacher_best_high_mu.pth",
    [string] $ClotPhiCheckpoint = "outputs\biochem\clot_baseline\clot_phi_best.pth",
    [string] $Anchor = "patient007",
    [int] $SimEndS = 30000,
    [int] $TimeStride = 5,
    [double] $MuRatioMax = 20,
    [switch] $Fast,
    [switch] $FullViz
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$legs = if ($Leg -eq "Both") { @("B", "B_deploy") } else { @($Leg) }

Write-Host "[NEW] MLP commit mask viz (oracle gt_clot vs active leg)" -ForegroundColor Cyan
Write-Host "[i]  Leg B        = both panels use gt_clot (oracle reference)" -ForegroundColor DarkGray
Write-Host "[i]  Leg B_deploy = left gt_clot oracle, right neighbor deploy mask" -ForegroundColor DarkGray
Write-Host "[i]  Scrub Time idx on Flow window; close figure to advance legs" -ForegroundColor DarkGray

foreach ($legId in $legs) {
    $vizArgs = @(
        "-File", (Join-Path $PSScriptRoot "go_mlp_abc_viz.ps1"),
        "-Leg", $legId,
        "-Anchor", $Anchor,
        "-SimEndS", "$SimEndS",
        "-TimeStride", "$TimeStride",
        "-MuRatioMax", "$MuRatioMax"
    )
    if ($Fast) { $vizArgs += "-Fast" }
    if ($FullViz) { $vizArgs += "-FullViz" }
    & powershell -NoProfile -ExecutionPolicy Bypass @vizArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "[OK]  commit mask viz complete." -ForegroundColor Green
