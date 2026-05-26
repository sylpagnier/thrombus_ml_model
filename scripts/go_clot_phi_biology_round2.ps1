# Round-2 biology: physics mu_ratio sweep + joint blend / pred-species trains.
#   powershell -File .\scripts\go_clot_phi_biology_round2.ps1 -Fresh
#   powershell -File .\scripts\go_clot_phi_biology_round2.ps1 -Stage train -Fresh

param(
    [ValidateSet("all", "physics", "train", "promote")]
    [string] $Stage = "all",
    [switch] $Fresh,
    [int] $Epochs = 60
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_clot_phi_shared_env.ps1")

$LadderRoot = Join-Path $RepoRoot "outputs\biochem\clot_phi_ladder"
New-Item -ItemType Directory -Force -Path $LadderRoot | Out-Null

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
$env:CLOT_PHI_SPECIES_FEATURES = "1"
$env:CLOT_PHI_RULE_BASELINE = "0"
$env:CLOT_PHI_SWEEP_DIR = ""
$env:CLOT_PHI_SWEEP_LEG = ""

function Invoke-PythonTrain {
    param([string]$Tag)
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    python -m src.training.train_clot_phi_simple
    $ec = $LASTEXITCODE
    $ErrorActionPreference = $prevEap
    if ($ec -ne 0) { throw "train failed: $Tag" }
}

function Invoke-PhysicsSweep {
    Write-Host "[NEW] Physics mu_ratio sweep (GT species)" -ForegroundColor Cyan
    $env:CLOT_PHI_PHYSICS_ORACLE = "1"
    $env:CLOT_PHI_JOINT_BIO = "0"
    $env:CLOT_PHI_SPECIES_FEATURES = "0"
    foreach ($ratio in @("2", "3", "4", "5", "8")) {
        foreach ($gate in @("0", "1")) {
            $cap = ""
            if ($ratio -eq "8") { $cap = "_cap2" ; $env:CLOT_PHI_PHYSICS_MU2_CAP = "2" }
            else { Remove-Item Env:CLOT_PHI_PHYSICS_MU2_CAP -ErrorAction SilentlyContinue }
            $env:CLOT_PHI_PHYSICS_MU_RATIO_MAX = $ratio
            $env:CLOT_PHI_PHYSICS_GELATION_GATE = $gate
            $tag = "mu${ratio}_gate${gate}${cap}"
            $logFile = Join-Path $LadderRoot "physics_${tag}.log"
            Write-Host "[i]  $tag" -ForegroundColor DarkCyan
            $prevEap = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            python -m src.training.train_clot_phi_simple *> $logFile
            $ErrorActionPreference = $prevEap
        }
    }
    $env:CLOT_PHI_PHYSICS_ORACLE = "0"
    python (Join-Path $PSScriptRoot "summarize_clot_phi_physics_sweep.py")
}

function Invoke-TrainLeg {
    param([string]$LegName, [hashtable]$Cfg)
    $env:CLOT_PHI_PHYSICS_ORACLE = "0"
    $env:CLOT_PHI_JOINT_BIO = "1"
    $env:CLOT_PHI_BIO_LAMBDA = "0.25"
    $env:CLOT_PHI_JOINT_USE_PRED_SPECIES = "0"
    $env:CLOT_PHI_PHYSICS_BLEND = "0"
    Remove-Item Env:CLOT_PHI_PHYSICS_MU2_CAP -ErrorAction SilentlyContinue
    foreach ($k in $Cfg.Keys) {
        Set-Item -Path "env:$k" -Value "$($Cfg[$k])"
    }
    $legDir = Join-Path $LadderRoot $LegName
    $env:CLOT_PHI_SWEEP_DIR = "outputs/biochem/clot_phi_ladder"
    $env:CLOT_PHI_SWEEP_LEG = $LegName
    if ($Fresh) {
        Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $legDir "clot_phi_best.pth")
        Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $legDir "clot_phi_train_log.jsonl")
    }
    Write-Host "[NEW] train leg=$LegName" -ForegroundColor Cyan
    Invoke-PythonTrain -Tag $LegName
    $env:CLOT_PHI_SWEEP_DIR = ""
    $env:CLOT_PHI_SWEEP_LEG = ""
}

function Invoke-TrainSweep {
    # Best physics ratio from round-1 (override via env CLOT_PHI_PHYSICS_MU_RATIO_MAX).
    $muRatio = if ($env:CLOT_PHI_PHYSICS_MU_RATIO_MAX) { $env:CLOT_PHI_PHYSICS_MU_RATIO_MAX } else { "4" }
    $legs = [ordered]@{
        joint_bio = @{
            CLOT_PHI_PHYSICS_GELATION_GATE = "0"
            CLOT_PHI_PHYSICS_BLEND = "0"
            CLOT_PHI_JOINT_USE_PRED_SPECIES = "0"
        }
        joint_pred = @{
            CLOT_PHI_PHYSICS_GELATION_GATE = "0"
            CLOT_PHI_PHYSICS_BLEND = "0"
            CLOT_PHI_JOINT_USE_PRED_SPECIES = "1"
        }
        joint_blend = @{
            CLOT_PHI_PHYSICS_GELATION_GATE = "1"
            CLOT_PHI_PHYSICS_BLEND = "1"
            CLOT_PHI_PHYSICS_BLEND_ALPHA = "0.55"
            CLOT_PHI_PHYSICS_MU_RATIO_MAX = $muRatio
            CLOT_PHI_JOINT_USE_PRED_SPECIES = "1"
        }
        joint_blend_gtsp = @{
            CLOT_PHI_PHYSICS_GELATION_GATE = "1"
            CLOT_PHI_PHYSICS_BLEND = "1"
            CLOT_PHI_PHYSICS_BLEND_ALPHA = "0.55"
            CLOT_PHI_PHYSICS_MU_RATIO_MAX = $muRatio
            CLOT_PHI_JOINT_USE_PRED_SPECIES = "0"
        }
    }
    foreach ($name in $legs.Keys) {
        Invoke-TrainLeg -LegName $name -Cfg $legs[$name]
    }
    python (Join-Path $PSScriptRoot "summarize_clot_phi_mlp_sweep.py") --sweep-dir "outputs/biochem/clot_phi_ladder"
}

function Invoke-PromoteBest {
    $summary = Join-Path $LadderRoot "summary.jsonl"
    if (-not (Test-Path $summary)) { throw "Missing $summary" }
    $best = (Get-Content $summary | Select-Object -First 1 | ConvertFrom-Json)
    $leg = $best.leg
    $src = Join-Path $LadderRoot "$leg\clot_phi_best.pth"
    if (-not (Test-Path $src)) { throw "Missing $src" }
    Copy-Item -Force $src "outputs\biochem\clot_phi_best.pth"
    Copy-Item -Force $src "outputs\biochem\clot_phi_best_mlp.pth"
    Write-Host "[OK]  promoted leg=$leg score=$($best.val_score) f1=$($best.val_f1)" -ForegroundColor Green
    $ErrorActionPreference = "Continue"
    python -m src.evaluation.viz_clot_phi_simple --anchor patient007 --checkpoint outputs/biochem/clot_phi_best.pth
    $ErrorActionPreference = "Stop"
}

switch ($Stage) {
    "physics" { Invoke-PhysicsSweep }
    "train"   { Invoke-TrainSweep }
    "promote" { Invoke-PromoteBest }
    default   {
        Invoke-PhysicsSweep
        Invoke-TrainSweep
        Invoke-PromoteBest
    }
}

Write-Host "[OK]  round2 done -> outputs/biochem/clot_phi_ladder/" -ForegroundColor Green
