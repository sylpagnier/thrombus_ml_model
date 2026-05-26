# Final clot-phi ablations: species crutch + blend alpha; promote winner.
#   powershell -File .\scripts\go_clot_phi_finalize.ps1 -Fresh
#   powershell -File .\scripts\go_clot_phi_finalize.ps1 -Stage eval

param(
    [ValidateSet("all", "train", "eval", "promote")]
    [string] $Stage = "all",
    [switch] $Fresh,
    [int] $Epochs = 50
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_clot_phi_shared_env.ps1")

$SweepRoot = "outputs/biochem/clot_phi_finalize"
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
$env:CLOT_PHI_PHYSICS_ORACLE = "0"
$env:CLOT_PHI_JOINT_BIO = "1"
$env:CLOT_PHI_BIO_LAMBDA = "0.25"
$env:CLOT_PHI_PHYSICS_MU_RATIO_MAX = "4"
$env:CLOT_PHI_PHYSICS_GELATION_GATE = "1"
$env:CLOT_PHI_SPECIES_HIDDEN = "32"
$env:CLOT_PHI_VAL_ANCHOR = "patient007"

$Legs = [ordered]@{
    gtsp_ctrl = @{
        CLOT_PHI_SPECIES_FEATURES = "1"
        CLOT_PHI_JOINT_USE_PRED_SPECIES = "0"
        CLOT_PHI_PHYSICS_BLEND = "1"
        CLOT_PHI_PHYSICS_BLEND_ALPHA = "0.55"
    }
    no_gt_pred_blend = @{
        CLOT_PHI_SPECIES_FEATURES = "0"
        CLOT_PHI_JOINT_USE_PRED_SPECIES = "1"
        CLOT_PHI_PHYSICS_BLEND = "1"
        CLOT_PHI_PHYSICS_BLEND_ALPHA = "0.55"
    }
    no_gt_pred_blend_a75 = @{
        CLOT_PHI_SPECIES_FEATURES = "0"
        CLOT_PHI_JOINT_USE_PRED_SPECIES = "1"
        CLOT_PHI_PHYSICS_BLEND = "1"
        CLOT_PHI_PHYSICS_BLEND_ALPHA = "0.75"
    }
    no_gt_pred_blend_a85 = @{
        CLOT_PHI_SPECIES_FEATURES = "0"
        CLOT_PHI_JOINT_USE_PRED_SPECIES = "1"
        CLOT_PHI_PHYSICS_BLEND = "1"
        CLOT_PHI_PHYSICS_BLEND_ALPHA = "0.85"
    }
    gtsp_a75 = @{
        CLOT_PHI_SPECIES_FEATURES = "1"
        CLOT_PHI_JOINT_USE_PRED_SPECIES = "0"
        CLOT_PHI_PHYSICS_BLEND = "1"
        CLOT_PHI_PHYSICS_BLEND_ALPHA = "0.75"
    }
    pred_phys_no_blend = @{
        CLOT_PHI_SPECIES_FEATURES = "1"
        CLOT_PHI_JOINT_USE_PRED_SPECIES = "1"
        CLOT_PHI_PHYSICS_BLEND = "0"
    }
}

function Invoke-Leg {
    param([string]$Name, [hashtable]$Cfg)
    foreach ($k in $Cfg.Keys) { Set-Item -Path "env:$k" -Value "$($Cfg[$k])" }
    $env:CLOT_PHI_SWEEP_DIR = $SweepRoot
    $env:CLOT_PHI_SWEEP_LEG = $Name
    $legDir = Join-Path $RepoRoot "$SweepRoot\$Name"
    if ($Fresh) {
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $legDir
    }
    Write-Host "[NEW] leg=$Name feat=$env:CLOT_PHI_SPECIES_FEATURES pred_sp=$env:CLOT_PHI_JOINT_USE_PRED_SPECIES blend=$env:CLOT_PHI_PHYSICS_BLEND alpha=$env:CLOT_PHI_PHYSICS_BLEND_ALPHA" -ForegroundColor Cyan
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    python -m src.training.train_clot_phi_simple
    $ec = $LASTEXITCODE
    $ErrorActionPreference = $prevEap
    if ($ec -ne 0) { throw "leg $Name failed" }
    $env:CLOT_PHI_SWEEP_DIR = ""
    $env:CLOT_PHI_SWEEP_LEG = ""
}

function Invoke-EvalMultiAnchor {
    $summary = Join-Path $RepoRoot "$SweepRoot\summary.jsonl"
    if (-not (Test-Path $summary)) {
        python (Join-Path $PSScriptRoot "summarize_clot_phi_mlp_sweep.py") --sweep-dir $SweepRoot
    }
    $best = Get-Content $summary | Select-Object -First 1 | ConvertFrom-Json
    $ckpt = Join-Path $RepoRoot "$SweepRoot\$($best.leg)\clot_phi_best.pth"
    if (-not (Test-Path $ckpt)) { $ckpt = "outputs/biochem/clot_phi_best.pth" }
    Write-Host "[i]  multi-anchor eval ckpt=$ckpt leg=$($best.leg)" -ForegroundColor Cyan
    $ErrorActionPreference = "Continue"
    python (Join-Path $PSScriptRoot "eval_clot_phi_multi_anchor.py") --checkpoint $ckpt --out (Join-Path $RepoRoot "$SweepRoot\multi_anchor_eval.jsonl")
    $ErrorActionPreference = "Stop"
}

function Invoke-Promote {
    python (Join-Path $PSScriptRoot "summarize_clot_phi_mlp_sweep.py") --sweep-dir $SweepRoot
    $summary = Join-Path $RepoRoot "$SweepRoot\summary.jsonl"
    $best = Get-Content $summary | Select-Object -First 1 | ConvertFrom-Json
    $src = Join-Path $RepoRoot "$SweepRoot\$($best.leg)\clot_phi_best.pth"
    Copy-Item -Force $src "outputs/biochem/clot_phi_best.pth"
    Copy-Item -Force $src "outputs/biochem/clot_phi_best_mlp.pth"
    Write-Host "[OK]  promoted $($best.leg) score=$($best.val_score) f1=$($best.val_f1)" -ForegroundColor Green
    $ErrorActionPreference = "Continue"
    python -m src.evaluation.viz_clot_phi_simple --anchor patient007 --checkpoint outputs/biochem/clot_phi_best.pth
    $ErrorActionPreference = "Stop"
}

switch ($Stage) {
    "train" { foreach ($n in $Legs.Keys) { Invoke-Leg -Name $n -Cfg $Legs[$n] }; python (Join-Path $PSScriptRoot "summarize_clot_phi_mlp_sweep.py") --sweep-dir $SweepRoot }
    "eval"  { Invoke-EvalMultiAnchor }
    "promote" { Invoke-Promote }
    default {
        foreach ($n in $Legs.Keys) { Invoke-Leg -Name $n -Cfg $Legs[$n] }
        python (Join-Path $PSScriptRoot "summarize_clot_phi_mlp_sweep.py") --sweep-dir $SweepRoot
        Invoke-EvalMultiAnchor
        Invoke-Promote
    }
}
