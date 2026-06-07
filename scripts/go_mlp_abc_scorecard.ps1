# A/B/C clot-shape scorecard (~1h, 3 anchors). North-star = spatial clot overlap
# (location-weighted F1 on rollout ch3 mu) + flow/bulk sanity guards.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_mlp_abc_scorecard.ps1
#   powershell ... -Fast                              # smoke (~15 min)
#   powershell ... -TimeStride 6 -Anchors "patient007"

param(
    [string] $TeacherCheckpoint = "outputs\biochem\clot_baseline\teacher_best_high_mu.pth",
    [string] $ClotPhiCheckpoint = "outputs\biochem\clot_baseline\clot_phi_best.pth",
    [string] $Anchors = "patient003,patient007,patient006",
    [string] $Legs = "A,B,C",
    [int] $TimeStride = 5,
    [double] $MuRatioMax = 20,
    [double] $Blend = 1.0,
    [switch] $Fast
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")

$env:BIOCHEM_GT_KINE_VEL = "0"
Remove-Item Env:BIOCHEM_GT_KINE_SKIP_DEQ -ErrorAction SilentlyContinue
$env:BIOCHEM_TEACHER_MU_RATIO_MAX = "$MuRatioMax"
$env:BIOCHEM_MLP_CLOT_BLEND = "$Blend"
Remove-Item Env:BIOCHEM_MLP_CLOT_INJECT, Env:BIOCHEM_MLP_MU_MAP, Env:BIOCHEM_MU_NEIGHBOR_WALL_ONLY -ErrorAction SilentlyContinue

# Visual clot threshold aligned with dynamic-mu viz (override with CLOT_SHAPE_MU_THRESH_SI).
if (-not $env:CLOT_SHAPE_MU_THRESH_SI) { $env:CLOT_SHAPE_MU_THRESH_SI = "0.055" }

Write-Host "[NEW] A/B/C clot-shape scorecard" -ForegroundColor Cyan
Write-Host "[i]  north-star = clot_shape only; flow_score + recall listed separately" -ForegroundColor DarkGray
Write-Host "[i]  clot = mu_eff >= CLOT_SHAPE_MU_THRESH_SI on full mesh @ rollout ch3" -ForegroundColor DarkGray
Write-Host "[i]  legs=$Legs  anchors=$Anchors  time_stride=$TimeStride  fast=$($Fast.IsPresent)" -ForegroundColor DarkGray

$outJson = if ($Fast) {
    "outputs/biochem/mlp_clot_inject_probe/abc_scorecard_fast.json"
} else {
    "outputs/biochem/mlp_clot_inject_probe/abc_scorecard_1h.json"
}

$pyArgs = @(
    (Join-Path $RepoRoot "scripts\run_mlp_clot_inject_probe.py"),
    "--teacher-checkpoint", $TeacherCheckpoint,
    "--clot-phi-checkpoint", $ClotPhiCheckpoint,
    "--anchors", $Anchors,
    "--legs", $Legs,
    "--time-stride", "$TimeStride",
    "--mu-ratio-max", "$MuRatioMax",
    "--out", $outJson
)
if ($Fast) { $pyArgs += "--fast" }

$rc = Invoke-PythonRc @pyArgs
if ($rc -ne 0) { exit $rc }

Write-Host "[OK]  $outJson" -ForegroundColor Green
Write-Host "[i]  per-leg viz: .\scripts\go_mlp_abc_viz.ps1 -Leg A|B|C -Anchor patient007" -ForegroundColor DarkGray
Write-Host "[i]  headless PNGs: .\scripts\go_mlp_abc_viz.ps1 -Headless -Leg All -Anchor patient007" -ForegroundColor DarkGray
