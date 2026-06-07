# Leg B v2 fast smoke: full MLP mu map in neighbor_wall + Carreau bulk (A/B/C probe).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_mlp_mu_map_v2_fast.ps1
#   powershell ... -Anchors "patient007"

param(
    [string] $TeacherCheckpoint = "outputs\biochem\clot_baseline\teacher_best_high_mu.pth",
    [string] $ClotPhiCheckpoint = "outputs\biochem\clot_baseline\clot_phi_best.pth",
    [string] $Anchors = "patient007",
    [double] $MuRatioMax = 20,
    [double] $Blend = 1.0
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")

$env:BIOCHEM_GT_KINE_VEL = "0"
Remove-Item Env:BIOCHEM_MLP_CLOT_INJECT, Env:BIOCHEM_MLP_MU_MAP, Env:BIOCHEM_MU_NEIGHBOR_WALL_ONLY -ErrorAction SilentlyContinue
$env:BIOCHEM_TEACHER_MU_RATIO_MAX = "$MuRatioMax"
$env:BIOCHEM_MLP_CLOT_BLEND = "$Blend"
$env:BIOCHEM_MLP_MU_MAP_PHI_GATE = "1"
$env:BIOCHEM_MLP_MU_MAP_MASK = "gt_clot"
$env:BIOCHEM_MLP_MU_MAP_BULK = "cap_low_shear"
$env:BIOCHEM_MLP_MU_MAP_GAMMA_THRESH_ND = "0.01"
$env:BIOCHEM_MLP_MU_MAP_GEO_CAP = "0"
Remove-Item Env:BIOCHEM_MLP_CLOT_REGION -ErrorAction SilentlyContinue
$env:BIOCHEM_MU_NEIGHBOR_WALL_MASK = "gt_clot"
$env:BIOCHEM_MU_NEIGHBOR_WALL_BULK = "cap_low_shear"

Write-Host "[NEW] Leg B v2 fast A/B/C probe (MLP mu map)" -ForegroundColor Cyan
Write-Host "[i]  B=mu_map_v2  C=neighbor_wall  ~15-25 min for 1 anchor" -ForegroundColor DarkGray

$pyArgs = @(
    (Join-Path $RepoRoot "scripts\run_mlp_clot_inject_probe.py"),
    "--teacher-checkpoint", $TeacherCheckpoint,
    "--clot-phi-checkpoint", $ClotPhiCheckpoint,
    "--anchors", $Anchors,
    "--fast",
    "--mu-ratio-max", "$MuRatioMax"
)
$rc = Invoke-PythonRc @pyArgs
if ($rc -ne 0) { exit $rc }

Write-Host "[OK]  outputs\biochem\mlp_clot_inject_probe\abc_compare_fast.json" -ForegroundColor Green
