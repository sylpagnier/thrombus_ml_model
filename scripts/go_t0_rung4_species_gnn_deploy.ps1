# Promote species GNN (s34+s35) into Rung 4 deploy stack + LOAO eval (6 anchors).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_t0_rung4_species_gnn_deploy.ps1"
#   powershell ... -Flow kinematics -SkipPromote
#   powershell ... -ValAnchor patient007 -StepEval patient004

param(
    [string] $SpeciesCkpt = "outputs/biochem/species_snapshot_s34/best.pth",
    [string] $Beta = "outputs/biochem/species_snapshot_s35/beta.pth",
    [string] $KineCkpt = "outputs/kinematics/kinematics_best.pth",
    [string] $ValAnchor = "patient007",
    [string] $Flow = "both",
    [string] $StepEval = "",
    [switch] $SkipPromote,
    [switch] $SkipLoao,
    [switch] $Viz
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

$manifest = "data/reference/species_gnn_deploy_r4.json"

if (-not $SkipPromote) {
    Write-Host "[NEW] Promote species GNN deploy manifest" -ForegroundColor Cyan
    Invoke-PythonRcCheck -Label "promote species gnn" -PyArgs @(
        "scripts/promote_species_gnn_deploy.py",
        "--species-ckpt", $SpeciesCkpt,
        "--beta", $Beta,
        "--kine-ckpt", $KineCkpt,
        "--val-anchor", $ValAnchor,
        "--out", $manifest
    )
}

if (-not $SkipLoao) {
    Write-Host "[NEW] LOAO eval (species_gnn vs s0)" -ForegroundColor Cyan
    Invoke-PythonRcCheck -Label "species gnn LOAO" -PyArgs @(
        "scripts/eval_t0_rung4_species_gnn_loao.py",
        "--manifest", $manifest,
        "--flow", $Flow
    )
}

if ($StepEval.Trim()) {
    Write-Host "[NEW] Rung4 step eval species_gnn ($StepEval)" -ForegroundColor Cyan
    $env:SPECIES_GNN_DEPLOY_MANIFEST = $manifest
    Invoke-PythonRcCheck -Label "rung4 species_gnn step" -PyArgs @(
        "scripts/eval_t0_rung4_step.py",
        "--anchor", $StepEval,
        "--step", "species_gnn",
        "--times", "0,27,53"
    )
}

if ($Viz) {
    $vizAnchor = if ($StepEval.Trim()) { $StepEval } else { $ValAnchor }
    Write-Host "[NEW] Clot ladder viz ($vizAnchor)" -ForegroundColor Cyan
    $env:SPECIES_GNN_CLOUT_CKPT = $SpeciesCkpt
    $env:SPECIES_VISCOSITY_CALIB = "1"
    $env:SPECIES_VISCOSITY_CALIB_PATH = (Join-Path $RepoRoot $Beta)
    Invoke-PythonRcCheck -Label "species gnn clot ladder" -PyArgs @(
        "scripts/viz_species_gnn_clot_ladder.py",
        "--anchor", $vizAnchor,
        "--ckpt", $SpeciesCkpt
    )
    Invoke-PythonRcCheck -Label "rung4 species_gnn viz" -PyArgs @(
        "scripts/viz_t0_rung4_step.py",
        "--anchor", $vizAnchor,
        "--step", "species_gnn",
        "--max-frames", "10"
    )
}

Write-Host "[OK] manifest=$manifest species=$SpeciesCkpt beta=$Beta" -ForegroundColor Green
