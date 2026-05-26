# Clot-phi biology ladder: physics oracle -> species features -> joint L_Data_Bio.
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_clot_phi_biology_ladder.ps1 -Fresh
#   powershell ... -Stage physics|species|joint|all

param(
    [ValidateSet("all", "physics", "species", "joint")]
    [string] $Stage = "all",
    [switch] $Fresh,
    [int] $Epochs = 50
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_clot_phi_shared_env.ps1")

$env:CLOT_PHI_MODEL = "mlp"
$env:CLOT_PHI_HIDDEN = "32"
$env:CLOT_PHI_MLP_DEPTH = "2"
$env:CLOT_PHI_DROPOUT = "0.15"
$env:CLOT_PHI_MU_LOG_LAMBDA = "1.5"
$env:CLOT_PHI_DICE_LAMBDA = "0.2"
$env:CLOT_PHI_LR = "1e-3"
$env:CLOT_PHI_WEIGHT_DECAY = "1e-4"
$env:CLOT_PHI_EPOCHS = "$Epochs"
$env:CLOT_PHI_MINIMAL_FEATURES = "1"
$env:CLOT_PHI_RULE_BASELINE = "0"
$env:CLOT_PHI_SWEEP_DIR = ""
$env:CLOT_PHI_SWEEP_LEG = ""

$LadderRoot = Join-Path $RepoRoot "outputs\biochem\clot_phi_ladder"
New-Item -ItemType Directory -Force -Path $LadderRoot | Out-Null

function Invoke-PhysicsOracle {
    Write-Host ""
    Write-Host "[NEW] Stage 1: physics oracle (GT u,v + GT Mat/FI gelation)" -ForegroundColor Cyan
    $env:CLOT_PHI_PHYSICS_ORACLE = "1"
    $env:CLOT_PHI_SPECIES_FEATURES = "0"
    $env:CLOT_PHI_JOINT_BIO = "0"
    foreach ($ratio in @("1.0", "80")) {
        $env:CLOT_PHI_PHYSICS_MU_RATIO_MAX = $ratio
        Write-Host "[i]  mu_ratio_max=$ratio" -ForegroundColor DarkCyan
        $logFile = Join-Path $LadderRoot "physics_mu_ratio_$ratio.log"
        $prevEap = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        python -m src.training.train_clot_phi_simple *> $logFile
        $ec = $LASTEXITCODE
        $ErrorActionPreference = $prevEap
        if ($ec -ne 0) { throw "physics oracle failed (mu_ratio_max=$ratio)" }
        Get-Content $logFile -Tail 4
    }
    $env:CLOT_PHI_PHYSICS_ORACLE = "0"
}

function Invoke-SpeciesFeatures {
    Write-Host ""
    Write-Host "[NEW] Stage 2: MLP + GT species features (in_dim=5)" -ForegroundColor Cyan
    $env:CLOT_PHI_PHYSICS_ORACLE = "0"
    $env:CLOT_PHI_SPECIES_FEATURES = "1"
    $env:CLOT_PHI_JOINT_BIO = "0"
    $ckpt = Join-Path $LadderRoot "clot_phi_species_feat_best.pth"
    $log = Join-Path $LadderRoot "clot_phi_species_feat_train_log.jsonl"
    if ($Fresh) {
        Remove-Item -Force -ErrorAction SilentlyContinue $ckpt, $log
        Remove-Item -Force -ErrorAction SilentlyContinue "outputs\biochem\clot_phi_best.pth"
        Remove-Item -Force -ErrorAction SilentlyContinue "outputs\biochem\clot_phi_train_log.jsonl"
    }
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    python -m src.training.train_clot_phi_simple
    $ec = $LASTEXITCODE
    $ErrorActionPreference = $prevEap
    if ($ec -ne 0) { throw "species-features train failed" }
    if (Test-Path "outputs\biochem\clot_phi_best.pth") {
        Copy-Item -Force "outputs\biochem\clot_phi_best.pth" $ckpt
        Copy-Item -Force "outputs\biochem\clot_phi_train_log.jsonl" $log -ErrorAction SilentlyContinue
    }
}

function Invoke-JointBio {
    Write-Host ""
    Write-Host "[NEW] Stage 3: joint clot head + L_Data_Bio species head (no ADR)" -ForegroundColor Cyan
    $env:CLOT_PHI_PHYSICS_ORACLE = "0"
    $env:CLOT_PHI_SPECIES_FEATURES = "1"
    $env:CLOT_PHI_JOINT_BIO = "1"
    $env:CLOT_PHI_BIO_LAMBDA = "0.25"
    $env:CLOT_PHI_JOINT_USE_PRED_SPECIES = "0"
    $env:CLOT_PHI_SPECIES_HIDDEN = "32"
    $ckpt = Join-Path $LadderRoot "clot_phi_joint_bio_best.pth"
    $log = Join-Path $LadderRoot "clot_phi_joint_bio_train_log.jsonl"
    if ($Fresh) {
        Remove-Item -Force -ErrorAction SilentlyContinue $ckpt, $log
        Remove-Item -Force -ErrorAction SilentlyContinue "outputs\biochem\clot_phi_best.pth"
        Remove-Item -Force -ErrorAction SilentlyContinue "outputs\biochem\clot_phi_train_log.jsonl"
    }
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    python -m src.training.train_clot_phi_simple
    $ec = $LASTEXITCODE
    $ErrorActionPreference = $prevEap
    if ($ec -ne 0) { throw "joint-bio train failed" }
    if (Test-Path "outputs\biochem\clot_phi_best.pth") {
        Copy-Item -Force "outputs\biochem\clot_phi_best.pth" $ckpt
        Copy-Item -Force "outputs\biochem\clot_phi_train_log.jsonl" $log -ErrorAction SilentlyContinue
        Copy-Item -Force "outputs\biochem\clot_phi_best.pth" "outputs\biochem\clot_phi_best_mlp.pth"
    }
    $ErrorActionPreference = "Continue"
    python -m src.evaluation.viz_clot_phi_simple --anchor patient007 --checkpoint outputs/biochem/clot_phi_best.pth
    $ErrorActionPreference = "Stop"
}

switch ($Stage) {
    "physics" { Invoke-PhysicsOracle }
    "species" { Invoke-SpeciesFeatures }
    "joint"   { Invoke-JointBio }
    default   {
        Invoke-PhysicsOracle
        Invoke-SpeciesFeatures
        Invoke-JointBio
    }
}

Write-Host "[OK]  Ladder artifacts under outputs/biochem/clot_phi_ladder/" -ForegroundColor Green
