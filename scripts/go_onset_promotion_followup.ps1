# Promotion follow-up: longer budget on top 2 onset recipes.
#
# Legs:
#   1) train_vel=kinematics + baseline losses
#   2) train_vel=kinematics + fp_weight=2
#
# Baseline temporal policy is handled by recipe env:
#   SPECIES_CONTINUOUS_TIME_CONTEXT=1, tau=t/3000 + Fourier, TEMPORAL_GATE=0

param(
    [string] $Anchors = "patient001,patient002,patient003,patient004,patient006,patient007",
    [string] $ValAnchor = "patient007",
    [int] $Epochs = 12,
    [int] $EarlyStop = 6,
    [int] $MaxWindows = 120,
    [string] $Times = "0,10,20,27,35,44,53,62,80,100,120,160,200",
    [double] $OnsetThreshold = 0.2
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$RunRoot = Join-Path $RepoRoot "outputs/biochem/biochem_gnn/onset_promotion_followup"
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
        name = "onset_promotion_followup"
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
    [System.IO.File]::WriteAllText(
        $ManifestPath,
        ($payload | ConvertTo-Json -Depth 6),
        (New-Object System.Text.UTF8Encoding $false)
    )
}

function Run-Leg {
    param(
        [string]$Name,
        [string]$FpWeight = "8.0"
    )
    $Dir = Join-Path $RunRoot $Name
    $SpeciesDir = Join-Path $Dir "species"
    $EvalDir = Join-Path $Dir "eval"
    New-Item -ItemType Directory -Force -Path $SpeciesDir, $EvalDir | Out-Null
    $SpeciesOut = Join-Path $SpeciesDir "best.pth"
    $ManifestPath = Join-Path $Dir "manifest.json"
    $EvalOut = Join-Path $EvalDir "deploy_ab_eval.json"
    $MetaPath = Join-Path $Dir "meta.json"

    # Force the intended temporal baseline policy + recipe deltas.
    $env:SPECIES_TRAIN_VEL_SOURCE = "kinematics"
    $env:SPECIES_CONTINUOUS_FP_WEIGHT = $FpWeight
    $env:SPECIES_CONTINUOUS_UNDERPRED_WEIGHT = "2.0"
    $env:SPECIES_CONTINUOUS_FINAL_STATE_WEIGHT = "0.35"
    $env:SPECIES_CONTINUOUS_TEMPORAL_GATE = "0"
    $env:SPECIES_CONTINUOUS_TIME_CONTEXT = "1"
    $env:SPECIES_CONTINUOUS_TIME_REF_S = "3000"
    $env:SPECIES_CONTINUOUS_TIME_FOURIER_FREQS = "8"

    $meta = @{
        condition = $Name
        train_vel_source = "kinematics"
        fp_weight = $FpWeight
        underpred_weight = "2.0"
        final_state_weight = "0.35"
        temporal_gate = "0"
        time_context = "1"
        time_ref_s = "3000"
        time_fourier_freqs = "8"
        epochs = $Epochs
        early_stop = $EarlyStop
        max_windows = $MaxWindows
    }
    [System.IO.File]::WriteAllText(
        $MetaPath,
        ($meta | ConvertTo-Json -Depth 5),
        (New-Object System.Text.UTF8Encoding $false)
    )

    Write-Host "[run] $Name train (vel=kinematics fp=$FpWeight ep=$Epochs)" -ForegroundColor Cyan
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
        "--init", $InitWarm,
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

Run-Leg -Name "kine_baseline_losses" -FpWeight "8.0"
Run-Leg -Name "kine_fp2" -FpWeight "2.0"

$null = Invoke-PythonRcCheck -Label "followup summary" -PyArgs @(
    "scripts/summarize_onset_phase2_diagnostics.py",
    "--run-root", $RunRoot,
    "--onset-threshold", "$OnsetThreshold"
)
Write-Host "[OK] report -> $RunRoot/phase2_diagnostics_report.md" -ForegroundColor Green
