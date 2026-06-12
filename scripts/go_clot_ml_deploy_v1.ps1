# Deploy v1 ladder run: 5a -> 5b -> 5c -> 9 -> 10 -> 11
# First fully deployable ML stack (frozen step1_a35, no retrain 0-7).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_ml_deploy_v1.ps1"
#   powershell ... -SkipCoupled -SkipHorizon   # faster: skip 5c + step 11

param(
    [string] $AnchorDir = "data/processed/graphs_biochem_anchors",
    [string] $Step0Json = "outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json",
    [string] $Step1Ckpt = "outputs/biochem/sweep_clot_ml_physics_6h/step1_a35/clot_ml_step1_best.pth",
    [string] $KineCkpt = "outputs/kinematics/kinematics_best.pth",
    [string] $Recipe = "data/reference/clot_ml_deploy_v1.json",
    [string] $OutRoot = "outputs/biochem/clot_ml_ladder/deploy_v1",
    [string] $HorizonScales = "1.0,1.25,1.5,2.0",
    [string] $HoldoutAnchors = "",
    [switch] $SkipCoupled,
    [switch] $SkipAudit,
    [switch] $SkipOod,
    [switch] $SkipHorizon
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_clot_prior_rule_winner_env.ps1")
. (Join-Path $PSScriptRoot "_python_rc.ps1")

$env:CLOT_TEMPORAL_VEL_SOURCE = "kinematics"
$env:CLOT_PHI_KINE_CKPT = $KineCkpt
$env:CLOT_PHI_MINIMAL_FEATURES = "1"
$env:CLOT_ML_USE_MACRO_TAU = "1"
$env:CLOT_ML_SIM_END_SCALE = "1.0"
$env:PYTHONUNBUFFERED = "1"

New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot $OutRoot) | Out-Null

function Invoke-Deploy {
    param([string] $Label, [string[]] $PyArgs)
    Write-Host ""
    Write-Host "[NEW] $Label" -ForegroundColor Cyan
    Invoke-PythonRcCheck -Label $Label -PyArgs $PyArgs
}

Write-Host "[NEW] clot ML deploy v1 ladder (5a-5c, 9-11)" -ForegroundColor Cyan
Write-Host "[i] recipe=$Recipe step1=$Step1Ckpt" -ForegroundColor DarkGray

# Preflight recipe write (refresh paths)
Invoke-Deploy "preflight recipe" @(
    "-c",
    "from pathlib import Path; from src.inference.clot_ml_deploy_v1 import DeployV1Recipe, save_deploy_v1_recipe; r=DeployV1Recipe(step0_json=r'$Step0Json', step1_ckpt=r'$Step1Ckpt', kine_ckpt=r'$KineCkpt'); save_deploy_v1_recipe(r, Path(r'$Recipe')); print('[OK] recipe', r.step1_ckpt)"
)

# Step 5a (LOAO mu readout via deploy v1 eval path)
Invoke-Deploy "5a step1 LOAO" @(
    "scripts/eval_clot_ml_step5a_mu_readout.py",
    "--shell", "step1",
    "--step0-json", $Step0Json,
    "--step1-ckpt", $Step1Ckpt,
    "--kine-ckpt", $KineCkpt,
    "--out", "$OutRoot/step5a_step1_summary.json"
)

# Step 5b smoke all anchors
Invoke-Deploy "5b coupled smoke p007" @(
    "scripts/smoke_clot_ml_step5b_coupled_kine.py",
    "--anchor", "patient007",
    "--shell", "step1",
    "--step0-json", $Step0Json,
    "--step1-ckpt", $Step1Ckpt,
    "--kine-ckpt", $KineCkpt,
    "--out", "$OutRoot/step5b_smoke_step1.json"
)

if (-not $SkipCoupled) {
    Invoke-Deploy "5c frozen vs coupled LOAO" @(
        "scripts/eval_clot_ml_step5c_closed_loop.py",
        "--recipe", $Recipe,
        "--out", "$OutRoot/step5c_compare.json"
    )
}

if (-not $SkipAudit) {
    Invoke-Deploy "9 forward audit" @(
        "scripts/audit_clot_ml_deploy_forward.py",
        "--recipe", $Recipe,
        "--out", "$OutRoot/step9_audit.json"
    )
}

if (-not $SkipOod) {
    $oodArgs = @(
        "scripts/eval_clot_ml_step10_spatial_ood.py",
        "--recipe", $Recipe,
        "--anchor-dir", $AnchorDir,
        "--out", "$OutRoot/step10_ood.json"
    )
    if ($HoldoutAnchors) {
        $oodArgs += @("--holdout", $HoldoutAnchors)
    }
    Invoke-Deploy "10 spatial OOD" $oodArgs
}

if (-not $SkipHorizon) {
    try {
        Invoke-Deploy "11 horizon gate" @(
            "scripts/eval_clot_ml_step11_horizon_gate.py",
            "--recipe", $Recipe,
            "--anchor-dir", $AnchorDir,
            "--scales", $HorizonScales,
            "--out", "$OutRoot/step11_horizon.json"
        )
    }
    catch {
        Write-Host "[WARN] step 11 gate failed (see step11_horizon.json); ladder artifacts saved" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "[OK] deploy v1 ladder complete -> $OutRoot" -ForegroundColor Green
Write-Host "[i] manifest: $Recipe" -ForegroundColor DarkGray
