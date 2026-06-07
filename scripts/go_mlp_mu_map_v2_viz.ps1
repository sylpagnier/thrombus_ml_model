# Viz with Leg B v2: MLP owns mu_eff in neighbor_wall during GNODE rollout.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_mlp_mu_map_v2_viz.ps1
#   powershell ... -Anchor patient007

param(
    [string] $Checkpoint = "",
    [string] $ClotPhiCheckpoint = "outputs\biochem\clot_baseline\clot_phi_best.pth",
    [string] $Anchor = "patient007",
    [int] $SimEndS = 30000,
    [double] $MuRatioMax = 20,
    [double] $Blend = 1.0
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")

function Resolve-TeacherDeployCkpt {
    param([string] $UserPath = "")
    if ($UserPath) {
        $p = if ([System.IO.Path]::IsPathRooted($UserPath)) { $UserPath } else { Join-Path $RepoRoot $UserPath }
        if (Test-Path $p) { return $p }
    }
    foreach ($rel in @(
            "outputs\biochem\clot_baseline\teacher_best_high_mu.pth",
            "outputs\biochem\biochem_teacher_best_high_mu.pth"
        )) {
        $c = Join-Path $RepoRoot $rel
        if (Test-Path $c) { return $c }
    }
    return $null
}

$Ckpt = Resolve-TeacherDeployCkpt -UserPath $Checkpoint
if (-not $Ckpt) {
    Write-Host "[ERR] No teacher checkpoint found." -ForegroundColor Red
    exit 1
}

$ClotPhi = Join-Path $RepoRoot ($ClotPhiCheckpoint -replace '/', '\')
if (-not (Test-Path $ClotPhi)) {
    Write-Host "[ERR] Missing clot-phi ckpt: $ClotPhiCheckpoint" -ForegroundColor Red
    exit 1
}

$env:BIOCHEM_GT_KINE_VEL = "0"
Remove-Item Env:BIOCHEM_MLP_CLOT_INJECT, Env:BIOCHEM_MU_NEIGHBOR_WALL_ONLY -ErrorAction SilentlyContinue
$env:BIOCHEM_MLP_MU_MAP = "1"
$env:BIOCHEM_MLP_MU_MAP_PHI_GATE = "1"
$env:BIOCHEM_MLP_MU_MAP_MASK = "gt_clot"
$env:BIOCHEM_MLP_MU_MAP_BULK = "cap_low_shear"
$env:BIOCHEM_MLP_MU_MAP_GAMMA_THRESH_ND = "0.01"
$env:BIOCHEM_MLP_MU_MAP_GEO_CAP = "0"
Remove-Item Env:BIOCHEM_MLP_CLOT_REGION -ErrorAction SilentlyContinue
$env:BIOCHEM_MLP_CLOT_CKPT = $ClotPhi
$env:BIOCHEM_MLP_CLOT_BLEND = "$Blend"
$env:BIOCHEM_TEACHER_MU_RATIO_MAX = "$MuRatioMax"

Write-Host "[NEW] Teacher viz + MLP mu map v2" -ForegroundColor Cyan
Write-Host "[i]  cap_low_shear Carreau bulk + GT-clot MLP mu | blend=$Blend" -ForegroundColor DarkGray

$pyArgs = @(
    "-m", "src.evaluation.visualize_pipeline",
    "--teacher-only",
    "--biochem-checkpoint", $Ckpt,
    "--anchor", $Anchor,
    "--sim-end-s", "$SimEndS",
    "--no-sim-end-prompt",
    "--clot-phi-checkpoint", $ClotPhi
)

$rc = Invoke-PythonRc @pyArgs
if ($rc -ne 0) { exit $rc }
Write-Host "[OK]  Done." -ForegroundColor Green
