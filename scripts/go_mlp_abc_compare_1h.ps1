# ~1h A/B/C mu coupling comparison (3 anchors, strided full rollout, not --fast).
#
#   A = baseline teacher (full-domain GNODE mu)
#   B = Leg B v2: cap_low_shear Carreau bulk + MLP mu map on gt_clot mask
#   C = Leg C: cap_low_shear Carreau bulk + GNODE delta_mu head on gt_clot mask only
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_mlp_abc_compare_1h.ps1
#   powershell ... -TimeStride 6 -Anchors "patient003,patient007,patient006"

param(
    [string] $TeacherCheckpoint = "outputs\biochem\clot_baseline\teacher_best_high_mu.pth",
    [string] $ClotPhiCheckpoint = "outputs\biochem\clot_baseline\clot_phi_best.pth",
    [string] $Anchors = "patient003,patient007,patient006",
    [int] $TimeStride = 5,
    [double] $MuRatioMax = 20,
    [double] $Blend = 1.0
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

# Leg B v2 + Leg C shared bulk/mask defaults (probe _configure_leg also sets per-leg).
$env:BIOCHEM_MLP_MU_MAP_PHI_GATE = "1"
$env:BIOCHEM_MLP_MU_MAP_MASK = "gt_clot"
$env:BIOCHEM_MLP_MU_MAP_BULK = "cap_low_shear"
$env:BIOCHEM_MLP_MU_MAP_GAMMA_THRESH_ND = "0.01"
$env:BIOCHEM_MLP_MU_MAP_GEO_CAP = "0"
$env:BIOCHEM_MU_NEIGHBOR_WALL_MASK = "gt_clot"
$env:BIOCHEM_MU_NEIGHBOR_WALL_BULK = "cap_low_shear"
Remove-Item Env:BIOCHEM_MLP_CLOT_REGION -ErrorAction SilentlyContinue

Write-Host "[NEW] A/B/C mu coupling ~1h compare" -ForegroundColor Cyan
Write-Host "[i]  A=baseline  B=MLP mu map v2  C=GNODE mask-only mu head" -ForegroundColor DarkGray
Write-Host "[i]  anchors=$Anchors  time_stride=$TimeStride  (~45-75 min typical)" -ForegroundColor DarkGray
Write-Host "[i]  teacher=$TeacherCheckpoint" -ForegroundColor DarkGray

$pyArgs = @(
    (Join-Path $RepoRoot "scripts\run_mlp_clot_inject_probe.py"),
    "--teacher-checkpoint", $TeacherCheckpoint,
    "--clot-phi-checkpoint", $ClotPhiCheckpoint,
    "--anchors", $Anchors,
    "--time-stride", "$TimeStride",
    "--mu-ratio-max", "$MuRatioMax",
    "--out", "outputs/biochem/mlp_clot_inject_probe/abc_compare_1h.json"
)
$rc = Invoke-PythonRc @pyArgs
if ($rc -ne 0) { exit $rc }

Write-Host "[OK]  outputs\biochem\mlp_clot_inject_probe\abc_compare_1h.json" -ForegroundColor Green
