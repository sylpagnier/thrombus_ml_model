# Leg B oracle (gt_clot) vs Leg B deploy (neighbor: wall + 1-hop from pred clot / phi).
# No COMSOL mu in the deploy commit mask; scorecard still uses GT for clot_shape eval only.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_mlp_b_deploy_probe.ps1 -Fast
#   powershell ... -VizOnly -Anchor patient007 -Leg B_deploy

param(
    [string] $TeacherCheckpoint = "outputs\biochem\clot_baseline\teacher_best_high_mu.pth",
    [string] $ClotPhiCheckpoint = "outputs\biochem\clot_baseline\clot_phi_best.pth",
    [string] $Anchors = "patient003,patient007,patient006",
    [string] $Anchor = "patient007",
    [ValidateSet("B", "B_deploy", "Both")]
    [string] $Leg = "Both",
    [int] $TimeStride = 5,
    [double] $MuRatioMax = 20,
    [double] $PhiThresh = 0.5,
    [switch] $Fast,
    [switch] $VizOnly,
    [switch] $NoPhiGate
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")

$env:BIOCHEM_GT_KINE_VEL = "0"
Remove-Item Env:BIOCHEM_GT_KINE_SKIP_DEQ -ErrorAction SilentlyContinue
$env:BIOCHEM_TEACHER_MU_RATIO_MAX = "$MuRatioMax"
$env:BIOCHEM_MLP_CLOT_BLEND = "1.0"
$env:BIOCHEM_MLP_MU_MAP_PHI_THRESH = "$PhiThresh"
if (-not $env:CLOT_SHAPE_MU_THRESH_SI) { $env:CLOT_SHAPE_MU_THRESH_SI = "0.055" }
if ($NoPhiGate) { $env:BIOCHEM_MLP_NEIGHBOR_REQUIRE_PHI = "0" }

Write-Host "[NEW] Leg B gt_clot vs B_deploy neighbor (no GT in mask)" -ForegroundColor Cyan
Write-Host "[i]  B        = oracle gt_clot mask" -ForegroundColor DarkGray
Write-Host "[i]  B_deploy = GT supervision band @ t=0 (wall/dgamma) + neighbor commit + 1-hop grow" -ForegroundColor DarkGray

if ($VizOnly) {
    $vizLegs = if ($Leg -eq "Both") { @("B", "B_deploy") } else { @($Leg) }
    foreach ($legId in $vizLegs) {
        & (Join-Path $PSScriptRoot "go_mlp_abc_viz.ps1") `
            -Leg $legId -Anchor $Anchor -TimeStride $TimeStride `
            -TeacherCheckpoint $TeacherCheckpoint -ClotPhiCheckpoint $ClotPhiCheckpoint `
            -MuRatioMax $MuRatioMax
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    exit 0
}

$legsArg = switch ($Leg) {
    "B" { "B" }
    "B_deploy" { "B_deploy" }
    default { "B,B_deploy" }
}

$outJson = if ($Fast) {
    if ($Leg -eq "B_deploy") {
        "outputs/biochem/mlp_clot_inject_probe/b_deploy_baseline_fast.json"
    } else {
        "outputs/biochem/mlp_clot_inject_probe/b_deploy_compare_fast.json"
    }
} else {
    "outputs/biochem/mlp_clot_inject_probe/b_deploy_compare_1h.json"
}

$pyArgs = @(
    (Join-Path $RepoRoot "scripts\run_mlp_clot_inject_probe.py"),
    "--teacher-checkpoint", $TeacherCheckpoint,
    "--clot-phi-checkpoint", $ClotPhiCheckpoint,
    "--anchors", $Anchors,
    "--legs", $legsArg,
    "--time-stride", "$TimeStride",
    "--mu-ratio-max", "$MuRatioMax",
    "--out", $outJson
)
if ($Fast) { $pyArgs += "--fast" }

$rc = Invoke-PythonRc @pyArgs
if ($rc -ne 0) { exit $rc }

Write-Host "[OK]  $outJson" -ForegroundColor Green
Write-Host "[i]  viz: .\scripts\go_mlp_b_deploy_probe.ps1 -VizOnly -Anchor $Anchor" -ForegroundColor DarkGray
