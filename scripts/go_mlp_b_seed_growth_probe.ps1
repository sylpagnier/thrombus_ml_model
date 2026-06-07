# B_seed_growth: GT clot map @ COMSOL t=0 restricts commit vision; grows 1-hop after pred clot inside mask.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_mlp_b_seed_growth_probe.ps1 -Fast
#   powershell ... -VizOnly -Anchor patient007

param(
    [string] $TeacherCheckpoint = "outputs\biochem\clot_baseline\teacher_best_high_mu.pth",
    [string] $ClotPhiCheckpoint = "outputs\biochem\clot_baseline\clot_phi_best.pth",
    [string] $Anchors = "patient003,patient007,patient006",
    [string] $Anchor = "patient007",
    [int] $TimeStride = 5,
    [double] $MuRatioMax = 20,
    [switch] $Fast,
    [switch] $VizOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")

$env:BIOCHEM_GT_KINE_VEL = "0"
$env:BIOCHEM_TEACHER_MU_RATIO_MAX = "$MuRatioMax"
$env:BIOCHEM_MLP_CLOT_BLEND = "1.0"

Write-Host "[NEW] B_seed_growth probe (GT t=0 vision + pred-clot 1-hop growth)" -ForegroundColor Cyan

if ($VizOnly) {
    & (Join-Path $PSScriptRoot "go_mlp_abc_viz.ps1") `
        -Leg B_seed_growth -Anchor $Anchor -TimeStride $TimeStride `
        -TeacherCheckpoint $TeacherCheckpoint -ClotPhiCheckpoint $ClotPhiCheckpoint `
        -MuRatioMax $MuRatioMax $(if ($Fast) { "-Fast" })
    exit $LASTEXITCODE
}

$outJson = if ($Fast) {
    "outputs/biochem/mlp_clot_inject_probe/b_seed_growth_fast.json"
} else {
    "outputs/biochem/mlp_clot_inject_probe/b_seed_growth.json"
}

$pyArgs = @(
    (Join-Path $RepoRoot "scripts\run_mlp_clot_inject_probe.py"),
    "--teacher-checkpoint", $TeacherCheckpoint,
    "--clot-phi-checkpoint", $ClotPhiCheckpoint,
    "--anchors", $Anchors,
    "--legs", "B_seed_growth",
    "--time-stride", "$TimeStride",
    "--mu-ratio-max", "$MuRatioMax",
    "--out", $outJson
)
if ($Fast) { $pyArgs += "--fast" }

$rc = Invoke-PythonRc @pyArgs
if ($rc -ne 0) { exit $rc }
Write-Host "[OK]  $outJson" -ForegroundColor Green
Write-Host "[i]  viz: .\scripts\go_mlp_b_seed_growth_probe.ps1 -VizOnly -Anchor $Anchor -Fast" -ForegroundColor DarkGray
