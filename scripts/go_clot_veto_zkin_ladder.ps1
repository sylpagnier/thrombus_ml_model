# Final flow-lever ladder on the REAL deploy clot F1.
#
# Four configs per anchor (see scripts/eval_clot_veto_zkin_ladder.py):
#   none        - standard deploy rollout (frozen kine).
#   kine_veto   - low-shear veto using the deployable kine shear (percentile-swept = ceiling of the deployable veto).
#   gt_veto     - low-shear veto using COMSOL GT shear (ceiling of the veto idea, same operator).
#   tiled_zkin  - no veto; z_kin refreshed mid-rollout by GEOMETRY OCCLUSION (clot nodes -> wall, DEQ re-solve).
#
# z_kin cannot be set from a corrector (u,v) field (it is the DEQ equilibrium; UV_PRIOR is only a warm
# start the solve washes out). Geometry occlusion is the only in-distribution clot-aware z_kin update.
#
# Run:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_clot_veto_zkin_ladder.ps1

param(
    [string] $SpeciesCkpt = "outputs/biochem/biochem_gnn/flow_aware_leashed_dynamic/sage/species/best.pth",
    [string] $Anchors = "patient001,patient002,patient003,patient004,patient006,patient007",
    [string] $ValAnchor = "patient007",
    [string] $Times = "53,200",
    [int]    $MinClot = 30,
    [double] $Growth = 2.0,
    [int]    $MaxRefresh = 3,
    [string] $Out = "outputs/biochem/corrector_coupling/veto_zkin_ladder/ladder.json"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$CkptAbs = Join-Path $RepoRoot $SpeciesCkpt
if (-not (Test-Path $CkptAbs)) { throw "missing species ckpt: $CkptAbs" }

Write-Host "[run] veto+occlusion-zkin ladder (deploy clot F1)" -ForegroundColor Cyan
$null = Invoke-PythonRcCheck -Label "veto_zkin ladder" -PyArgs @(
    "scripts/eval_clot_veto_zkin_ladder.py",
    "--species-ckpt", $SpeciesCkpt,
    "--anchors", $Anchors,
    "--val-anchor", $ValAnchor,
    "--times", $Times,
    "--min-clot", "$MinClot",
    "--growth", "$Growth",
    "--max-refresh", "$MaxRefresh",
    "--out", $Out
)
Write-Host "[OK] ladder -> $Out" -ForegroundColor Green
