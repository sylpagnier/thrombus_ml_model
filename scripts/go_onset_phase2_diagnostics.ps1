# Phase-2 onset diagnostics:
# 1) train/deploy flow alignment probe
# 2) onset-focused loss ablation small grid

param(
    [string] $Anchors = "patient001,patient002,patient003,patient004,patient006,patient007",
    [string] $ValAnchor = "patient007",
    [int] $Epochs = 3,
    [int] $EarlyStop = 2,
    [int] $MaxWindows = 24,
    [string] $Times = "0,10,20,27,35,44,53,62,80,100,120",
    [double] $OnsetThreshold = 0.2
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$RunRoot = Join-Path $RepoRoot "outputs/biochem/biochem_gnn/onset_phase2_diagnostics"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

$InitWarm = Join-Path $RepoRoot "outputs/biochem/biochem_gnn/locked/species_gnn_best.pth"
$BetaCkpt = Join-Path $RepoRoot "outputs/biochem/biochem_gnn/locked/viscosity_beta.pth"
if (-not (Test-Path $InitWarm)) { throw "missing init ckpt: $InitWarm" }
if (-not (Test-Path $BetaCkpt)) { throw "missing beta ckpt: $BetaCkpt" }

function RelPath([string]$AbsPath) {
    return (Resolve-Path $AbsPath).Path.Substring($RepoRoot.Length).TrimStart('\').Replace('\', '/')
}

function Write-Manifest([string]$ManifestPath, [string]$SpeciesCkpt) {
    $payload = @{
        name = "onset_phase2_diag"
        version = 1
        baseline = @{
            species_gnn_ckpt = (RelPath $SpeciesCkpt)
            viscosity_beta = (RelPath $BetaCkpt)
            kinematics_ckpt = "outputs/kinematics/kinematics_best.pth"
            train_val_anchor = $ValAnchor
            flow_modes = "kinematics"
            gamma_mode = "max"
            deploy_horizon = "full"
            clot_score = "guiding"
            pushforward_arch = "sage"
            gate_mode = "global_sigmoid"
            species_scope = "fi_mat"
            loao_auto = "0"
        }
    }
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($ManifestPath, ($payload | ConvertTo-Json -Depth 6), $utf8NoBom)
}

function Run-Condition {
    param(
        [string]$Name,
        [string]$TrainVel = "gt",
        [string]$UnderPred = "2.0",
        [string]$FpWeight = "8.0",
        [string]$FinalStateWeight = "0.35"
    )
    $Dir = Join-Path $RunRoot $Name
    $SpeciesDir = Join-Path $Dir "species"
    $EvalDir = Join-Path $Dir "eval"
    New-Item -ItemType Directory -Force -Path $SpeciesDir, $EvalDir | Out-Null
    $SpeciesOut = Join-Path $SpeciesDir "best.pth"
    $EvalOut = Join-Path $EvalDir "deploy_ab_eval.json"
    $ManifestPath = Join-Path $Dir "manifest.json"
    $MetaPath = Join-Path $Dir "meta.json"

    # Recipe-consistent knobs with targeted overrides.
    $env:SPECIES_TRAIN_VEL_SOURCE = $TrainVel
    $env:SPECIES_CONTINUOUS_UNDERPRED_WEIGHT = $UnderPred
    $env:SPECIES_CONTINUOUS_FP_WEIGHT = $FpWeight
    $env:SPECIES_CONTINUOUS_FINAL_STATE_WEIGHT = $FinalStateWeight

    $meta = @{
        condition = $Name
        train_vel_source = $TrainVel
        underpred_weight = $UnderPred
        fp_weight = $FpWeight
        final_state_weight = $FinalStateWeight
        epochs = $Epochs
        early_stop = $EarlyStop
        max_windows = $MaxWindows
    }
    [System.IO.File]::WriteAllText($MetaPath, ($meta | ConvertTo-Json -Depth 5), (New-Object System.Text.UTF8Encoding $false))

    Write-Host "[run] $Name train (vel=$TrainVel up=$UnderPred fp=$FpWeight final=$FinalStateWeight)" -ForegroundColor Cyan
    $null = Invoke-PythonRcCheck -Label "$Name train" -PyArgs @(
        "-m", "src.training.train_species_pushforward_continuous",
        "--phase", "biochem_gnn",
        "--anchors", $Anchors,
        "--val-anchor", $ValAnchor,
        "--epochs", "$Epochs",
        "--early-stop", "$EarlyStop",
        "--max-windows", "$MaxWindows",
        "--unroll", "10",
        "--arch", "sage",
        "--init-s26", $InitWarm,
        "--out", $SpeciesOut
    )

    Write-Manifest -ManifestPath $ManifestPath -SpeciesCkpt $SpeciesOut
    Write-Host "[run] $Name eval deploy_frozen" -ForegroundColor Cyan
    $null = Invoke-PythonRcCheck -Label "$Name eval" -PyArgs @(
        "scripts/eval_biochem_gnn_deploy_ab.py",
        "--manifest", $ManifestPath,
        "--modes", "deploy_frozen",
        "--anchors", $Anchors,
        "--times", $Times,
        "--out", $EvalOut
    )
}

# Baseline + flow alignment probe + small onset-loss grid.
Run-Condition -Name "baseline" -TrainVel "gt" -UnderPred "2.0" -FpWeight "8.0" -FinalStateWeight "0.35"
Run-Condition -Name "flow_align_train_kine" -TrainVel "kinematics" -UnderPred "2.0" -FpWeight "8.0" -FinalStateWeight "0.35"
Run-Condition -Name "loss_underpred_1p0" -TrainVel "gt" -UnderPred "1.0" -FpWeight "8.0" -FinalStateWeight "0.35"
Run-Condition -Name "loss_fp_2p0" -TrainVel "gt" -UnderPred "2.0" -FpWeight "2.0" -FinalStateWeight "0.35"
Run-Condition -Name "loss_final_0p10" -TrainVel "gt" -UnderPred "2.0" -FpWeight "8.0" -FinalStateWeight "0.10"
Run-Condition -Name "loss_balanced_combo" -TrainVel "gt" -UnderPred "1.0" -FpWeight "2.0" -FinalStateWeight "0.10"

$null = Invoke-PythonRcCheck -Label "phase2 summary" -PyArgs @(
    "scripts/summarize_onset_phase2_diagnostics.py",
    "--run-root", $RunRoot,
    "--onset-threshold", "$OnsetThreshold"
)
Write-Host "[OK] report -> $RunRoot/phase2_diagnostics_report.md" -ForegroundColor Green
