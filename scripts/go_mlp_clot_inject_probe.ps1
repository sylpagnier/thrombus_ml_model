# A/B/C mu coupling probe (closed-loop GNODE + DEQ).
#
#   A = baseline Lane A teacher (no inject, no neighbor_wall gate)
#   B = Leg B v2: full MLP mu map in neighbor_wall + Carreau bulk (default)
#   C = neighbor_wall mask-only: cap_low_shear Carreau bulk + GNODE mu head on gt_clot mask
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_mlp_clot_inject_probe.ps1
#   powershell ... -Fast -Anchors "patient007"            # ~15-25 min smoke (3 keyframes x 3 legs)
#   powershell ... -Anchors "patient007" -TimeStride 6   # medium (~40 min/leg)

param(
    [string] $TeacherCheckpoint = "outputs\biochem\clot_baseline\teacher_best_high_mu.pth",
    [string] $ClotPhiCheckpoint = "outputs\biochem\clot_baseline\clot_phi_best.pth",
    [string] $Anchors = "patient003,patient007,patient006",
    [int] $TimeIndex = -1,
    [int] $TimeStride = 1,
    [switch] $Fast,
    [double] $MuRatioMax = 20,
    [double] $MuClotSi = 0.10,
    [double] $Blend = 1.0,
    [switch] $LegBV1
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")

$env:BIOCHEM_GT_KINE_VEL = "0"
Remove-Item Env:BIOCHEM_GT_KINE_SKIP_DEQ -ErrorAction SilentlyContinue
$env:BIOCHEM_TEACHER_MU_RATIO_MAX = "$MuRatioMax"
if ($LegBV1) {
    $env:BIOCHEM_MLP_CLOT_MU_SI = "$MuClotSi"
}
$env:BIOCHEM_MLP_CLOT_BLEND = "$Blend"
Remove-Item Env:BIOCHEM_MLP_CLOT_INJECT, Env:BIOCHEM_MLP_MU_MAP, Env:BIOCHEM_MU_NEIGHBOR_WALL_ONLY -ErrorAction SilentlyContinue

Write-Host "[NEW] A/B/C mu coupling probe" -ForegroundColor Cyan
if ($LegBV1) {
    Write-Host "[i]  A=baseline  B=MLP trigger v1  C=neighbor_wall mu only" -ForegroundColor DarkGray
} else {
    Write-Host "[i]  A=baseline  B=MLP mu map v2  C=neighbor_wall mu only" -ForegroundColor DarkGray
}
Write-Host "[i]  teacher=$TeacherCheckpoint" -ForegroundColor DarkGray
if ($Fast) {
    Write-Host "[i]  mode=FAST (3 keyframes, low DEQ/ODE cost)" -ForegroundColor DarkGray
} else {
    Write-Host "[i]  time_stride=$TimeStride" -ForegroundColor DarkGray
}
Write-Host "[i]  macro-step progress on (BIOCHEM_ROLLOUT_PROGRESS=1)" -ForegroundColor DarkGray

$pyArgs = @(
    (Join-Path $RepoRoot "scripts\run_mlp_clot_inject_probe.py"),
    "--teacher-checkpoint", $TeacherCheckpoint,
    "--clot-phi-checkpoint", $ClotPhiCheckpoint,
    "--anchors", $Anchors,
    "--time-index", "$TimeIndex",
    "--time-stride", "$TimeStride",
    "--mu-ratio-max", "$MuRatioMax"
)
if ($Fast) { $pyArgs += "--fast" }
if ($LegBV1) { $pyArgs += "--leg-b-v1" }
$rc = Invoke-PythonRc @pyArgs
if ($rc -ne 0) { exit $rc }

$outJson = if ($Fast) { "abc_compare_fast.json" } else { "abc_compare.json" }
Write-Host "[OK]  See outputs\biochem\mlp_clot_inject_probe\$outJson" -ForegroundColor Green
