# Step 2 v3: diagnose-aligned deploy coupled finetune (B_wired).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_mlp_deploy_coupled_finetune.ps1 -Fresh
#   powershell ... -Smoke          # 2 ep FAST (~30 min)
#   powershell ... -Dev -Fresh     # p007-only final-frame iterate (~15 min/ep)

param(
    [switch] $Fresh,
    [switch] $Smoke,
    [switch] $Dev,
    [int] $Epochs = 12,
    [double] $Lr = 5e-4,
    [int] $TimeStride = 5,
    [string] $TeacherCheckpoint = "outputs/biochem/gnode10_sweep/gnode12_lane_a_promoted/biochem_teacher_best_high_mu.pth",
    [string] $InitClotPhi = "outputs/biochem/passive_species_focus_compare/gnode12_lane_a_clotphi/clot_phi_best.pth",
    [string] $OutDir = "outputs/biochem/mlp_deploy_coupled_v3",
    [switch] $TrainPhiToo
)

if ($Smoke) {
    $Epochs = 2
    $TimeStride = 5
    $OutDir = "outputs/biochem/mlp_deploy_coupled_v3_smoke"
}

if ($Dev) {
    $Epochs = 4
    $TimeStride = 5
    $OutDir = "outputs/biochem/mlp_deploy_coupled_v3_dev2"
    $Lr = 1e-3
}

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
. (Join-Path $PSScriptRoot "_clot_phi_shared_env.ps1")

$env:CLOT_PHI_DGAMMA_FEATURE_TIME = "current"
$env:CLOT_PHI_JOINT_USE_PRED_SPECIES = "1"
$env:CLOT_PHI_PHYSICS_BLEND = "0"
$env:CLOT_PHI_MU_LOG_LAMBDA = "2.0"
$env:CLOT_PHI_DEPLOY_SOFT_COMMIT_LAMBDA = "4.0"
$env:CLOT_PHI_DEPLOY_MU_HINGE_LAMBDA = "5.0"
$env:CLOT_PHI_DEPLOY_ALLOWED_HINGE_LAMBDA = $(if ($Dev) { "30.0" } else { "12.0" })
$env:CLOT_PHI_DEPLOY_PHI_LAMBDA = $(if ($TrainPhiToo) { "0.5" } else { "0" })
$env:CLOT_PHI_DEPLOY_TRAIN_MU_ONLY = $(if ($TrainPhiToo) { "0" } else { "1" })
$env:MLP_DEPLOY_COUPLED_CLOSED_LOOP = "1"
$env:MLP_DEPLOY_COUPLED_GRAPH_PASSES = "2"
$env:MLP_DEPLOY_COUPLED_BIAS_INIT = "1"
$env:MLP_DEPLOY_COUPLED_FINAL_FRAME_WEIGHT = "3.0"
$env:MLP_DEPLOY_COUPLED_EPOCHS = "$Epochs"
$env:MLP_DEPLOY_COUPLED_LR = "$Lr"
$env:MLP_DEPLOY_COUPLED_TIME_STRIDE = "$TimeStride"
$env:MLP_DEPLOY_COUPLED_TEACHER = $TeacherCheckpoint
$env:MLP_DEPLOY_COUPLED_INIT_CLOTPHI = $InitClotPhi
$env:MLP_DEPLOY_COUPLED_OUT_DIR = $OutDir
$env:BIOCHEM_GT_KINE_VEL = "0"
$env:BIOCHEM_TEACHER_MU_RATIO_MAX = "20"

if ($Smoke) {
    $env:MLP_DEPLOY_COUPLED_FAST = "1"
    $env:MLP_DEPLOY_COUPLED_MAX_TRAIN = "2"
    $env:MLP_DEPLOY_COUPLED_TRAIN_ANCHORS = "patient001,patient006"
    $env:MLP_DEPLOY_COUPLED_GRAPH_PASSES = "1"
    Remove-Item Env:MLP_DEPLOY_COUPLED_VAL_STRIDE -ErrorAction SilentlyContinue
}

if ($Dev) {
    $env:MLP_DEPLOY_COUPLED_FAST = "1"
    $env:MLP_DEPLOY_COUPLED_TRAIN_ON_VAL = "1"
    $env:MLP_DEPLOY_COUPLED_TRAIN_ANCHORS = "patient007"
    $env:MLP_DEPLOY_COUPLED_MAX_TRAIN = "1"
    $env:MLP_DEPLOY_COUPLED_FINAL_FRAME_ONLY = "1"
    $env:MLP_DEPLOY_COUPLED_GRAPH_PASSES = "1"
    $env:MLP_DEPLOY_COUPLED_BIAS_INIT = "0"
    $env:CLOT_PHI_VAL_ANCHOR = "patient007"
    Remove-Item Env:MLP_DEPLOY_COUPLED_VAL_STRIDE -ErrorAction SilentlyContinue
}

$teacherPath = Join-Path $RepoRoot ($TeacherCheckpoint -replace '/', '\')
$initPath = Join-Path $RepoRoot ($InitClotPhi -replace '/', '\')
if (-not (Test-Path $teacherPath)) {
    Write-Host "[ERR] Missing teacher ckpt: $teacherPath" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $initPath)) {
    Write-Host "[ERR] Missing clot-phi init: $initPath" -ForegroundColor Red
    exit 1
}

$LegDir = Join-Path $RepoRoot ($OutDir -replace '/', '\')
$Ckpt = Join-Path $LegDir "clot_phi_best.pth"
if ($Fresh) {
    Remove-Item -Force -ErrorAction SilentlyContinue $Ckpt, (Join-Path $LegDir "deploy_coupled_train_log.jsonl")
}

Write-Host "[NEW] Step 2 v3 deploy coupled (diagnose-aligned allowed mask)" -ForegroundColor Cyan
if ($Smoke) { Write-Host "[i]  SMOKE mode" -ForegroundColor Yellow }
if ($Dev) { Write-Host "[i]  DEV: p007 final-frame only, 4 ep" -ForegroundColor Yellow }
Write-Host "[i]  out=$OutDir epochs=$Epochs mu_only=$(-not $TrainPhiToo)" -ForegroundColor DarkGray

$rc = Invoke-PythonRc -m src.training.train_mlp_deploy_coupled
if ($rc -ne 0) {
    Write-Host "[ERR] Training failed (exit $rc)" -ForegroundColor Red
    exit $rc
}

Write-Host "[OK]  ckpt -> $Ckpt" -ForegroundColor Green
Write-Host "[i]  verify:" -ForegroundColor DarkGray
Write-Host "  powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_diagnose_deploy_gate.ps1 -Anchor patient007 -Leg B_wired -Fast -ClotPhiCheckpoint $OutDir/clot_phi_best.pth" -ForegroundColor DarkGray
