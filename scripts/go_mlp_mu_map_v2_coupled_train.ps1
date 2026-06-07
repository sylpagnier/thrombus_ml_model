# Quick GNODE finetune with Leg B v2 frozen MLP mu map in forward (closed-loop DEQ).
# MLP stays frozen; teacher adapts flow/species to coupled viscosity.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_mlp_mu_map_v2_coupled_train.ps1
#   powershell ... -Epochs 4 -InitCkpt outputs\biochem\clot_baseline\teacher_best_high_mu.pth

param(
    [int] $Epochs = 4,
    [string] $RunNote = "mlp_mu_map_v2_coupled",
    [string] $InitCkpt = "outputs\biochem\clot_baseline\teacher_best_high_mu.pth",
    [string] $ClotPhiCheckpoint = "outputs\biochem\clot_baseline\clot_phi_best.pth",
    [double] $MuRatioMax = 20,
    [double] $Blend = 1.0,
    [switch] $DeployNeighbor
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")

$initPath = Join-Path $RepoRoot ($InitCkpt -replace '/', '\')
if (-not (Test-Path $initPath)) {
    Write-Host "[ERR] Missing init ckpt: $initPath" -ForegroundColor Red
    exit 1
}
$clotPath = Join-Path $RepoRoot ($ClotPhiCheckpoint -replace '/', '\')
if (-not (Test-Path $clotPath)) {
    Write-Host "[ERR] Missing clot-phi ckpt: $clotPath" -ForegroundColor Red
    exit 1
}

$env:BIOCHEM_GT_KINE_VEL = "0"
Remove-Item Env:BIOCHEM_MLP_CLOT_INJECT, Env:BIOCHEM_MU_NEIGHBOR_WALL_ONLY -ErrorAction SilentlyContinue
if ($DeployNeighbor) {
    . (Join-Path $PSScriptRoot "_python_rc.ps1")
    python -c "from src.inference.deploy_mu_map_env import apply_deploy_mu_map_env; apply_deploy_mu_map_env()"
} else {
    $env:BIOCHEM_MLP_MU_MAP = "1"
    $env:BIOCHEM_MLP_MU_MAP_PHI_GATE = "1"
    $env:BIOCHEM_MLP_MU_MAP_MASK = "gt_clot"
    $env:BIOCHEM_MLP_MU_MAP_BULK = "cap_low_shear"
    $env:BIOCHEM_MLP_MU_MAP_GAMMA_THRESH_ND = "0.01"
    $env:BIOCHEM_MLP_MU_MAP_GEO_CAP = "0"
    Remove-Item Env:BIOCHEM_MLP_CLOT_REGION -ErrorAction SilentlyContinue
}
$env:BIOCHEM_MLP_CLOT_CKPT = $clotPath
$env:BIOCHEM_MLP_CLOT_BLEND = "$Blend"
$env:BIOCHEM_MLP_CLOT_USE_PRED_SPECIES = "1"
$env:BIOCHEM_TEACHER_MU_RATIO_MAX = "$MuRatioMax"
$env:BIOCHEM_COMPLEXITY_STEP = "2"
$env:BIOCHEM_LOSS_DATA_ONLY = "1"
$env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "0.35"
$env:BIOCHEM_MU_LOG_WALL_WEIGHT = "0.5"
$env:BIOCHEM_MU_LOG_HIGH_WEIGHT = "0.75"
$env:BIOCHEM_LORA_RANK = "4"

Write-Host "[NEW] Leg B v2 coupled finetune ($Epochs ep)" -ForegroundColor Cyan
Write-Host "[i]  MLP mu map ON (frozen) | init=$InitCkpt" -ForegroundColor DarkGray

Copy-Item $initPath (Join-Path $RepoRoot "outputs\biochem\biochem_teacher_best_high_mu.pth") -Force

$rc = Invoke-PythonRc -m src.training.train_biochem_corrector --new --skip-pretrain --init-from-best `
    --epochs $Epochs --save-best --run-name $RunNote
if ($rc -ne 0) {
    Write-Host "[ERR] Training failed (exit $rc)" -ForegroundColor Red
    exit $rc
}

$outCkpt = Join-Path $RepoRoot "outputs\biochem\biochem_teacher_best_high_mu.pth"
Write-Host "[OK]  coupled teacher -> $outCkpt" -ForegroundColor Green
Write-Host "[i]  smoke: .\scripts\go_mlp_mu_map_v2_fast.ps1 -TeacherCheckpoint outputs\biochem\biochem_teacher_best_high_mu.pth" -ForegroundColor DarkGray
