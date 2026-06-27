# A/B/C/D precision ladder on the deployable clot predictor (non-flow open levers).
#
# All flow routes are exhausted (docs/SPECIES_LEARNING_STRATEGY.md 6.9). This tests the two
# remaining OPEN, NON-FLOW levers for wall false-positive precision, as a clean 2x2 factorial:
#
#   A baseline       - proven Mat+FG control (phase recipe: fp_weight=8, mature_fp_exempt, mat-based
#                      checkpoint selection). Reproduces the moves234 0.591 control.
#   B footprint_sup  - footprint-aligned supervision BEYOND baseline: stronger Mat-field FP penalty
#                      (fp_weight 8->16, the ActiveGrowthHuber term that reaches the deploy trigger)
#                      PLUS matched checkpoint selection on deploy clot F1 (score_clot_w, which the
#                      baseline lacks). NOT the saturated gelation-sigmoid physics_readout path (6.8).
#   C geom_feats     - static NON-FLOW geometry discriminators (width / expansion / wall curvature)
#                      appended to the GNN inputs (SPECIES_GEOM_FEATS).
#   D both           - B + C.
#
# All legs: fresh (C/D change input dim), identical data/epochs/lr, val=patient007, deploy_frozen eval.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_species_abcd.ps1
#   powershell ... -Epochs 40 -Legs A,B,C,D

param(
    [int] $Epochs = 40,
    [int] $EarlyStop = 18,
    [int] $MaxWindows = 40,
    [double] $Lr = 3e-4,
    [string] $Anchors = "patient001,patient002,patient003,patient004,patient006,patient007",
    [string] $ValAnchor = "patient007",
    [string] $Times = "53,200",
    [string[]] $Legs = @("A", "B", "C", "D"),
    [double] $FpWeight = 16.0,   # leg B/D: above the phase baseline of 8 (stronger footprint FP penalty)
    [double] $ScoreClotW = 0.6   # leg B/D: matched checkpoint selection on deploy clot F1 (A/C use mat)
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$RunRoot = Join-Path $RepoRoot "outputs/biochem/biochem_gnn/abcd_precision"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
$BetaCkpt = "outputs/biochem/biochem_gnn/locked/viscosity_beta.pth"

function RelPath([string]$AbsPath) {
    return (Resolve-Path $AbsPath).Path.Substring($RepoRoot.Length).TrimStart('\').Replace('\', '/')
}

# Knobs a leg may set; cleared between legs so each starts from the shared baseline state.
function Clear-LegEnv {
    foreach ($k in @(
        "SPECIES_CONTINUOUS_FP_WEIGHT",
        "SPECIES_CONTINUOUS_SCORE_CLOUT_W",
        "SPECIES_CONTINUOUS_MATURE_FP_EXEMPT",
        "SPECIES_GEOM_FEATS",
        # ensure no stale flow/move knobs leak in from a prior shell
        "SPECIES_FLOW_FEATS", "SPECIES_FLOW_FEATS_DYNAMIC", "SPECIES_STAGNATION_FEATS",
        "SPECIES_CONTINUOUS_PHYSICS_READOUT", "SPECIES_LATENT_DROPOUT"
    )) { Remove-Item "env:$k" -ErrorAction SilentlyContinue }
}

function Set-SharedRecipe {
    # Proven Mat+FG deploy recipe (matches the moves234 control = the 0.591-holdout baseline).
    $env:BIOCHEM_PUSHFORWARD_SPECIES_CHANNELS = "11,7"
    $env:SPECIES_PUSHFORWARD_ARCH = "sage"
    $env:SPECIES_TRAIN_VEL_SOURCE = "gt"
    $env:SPECIES_ROLLOUT_DEPLOY_FAITHFUL = "1"
    $env:SPECIES_ROLLOUT_VEL_SOURCE = "kinematics"
    $env:SPECIES_ROLLOUT_PIN_OTHER = "rest"
    $env:SPECIES_ROLLOUT_IC_SOURCE = "resting"
}

function Train-Leg([string]$Code, [string]$Label) {
    $legDir = Join-Path $RunRoot $Label
    $speciesDir = Join-Path $legDir "species"
    $evalDir = Join-Path $legDir "eval"
    New-Item -ItemType Directory -Force -Path $speciesDir, $evalDir | Out-Null
    $speciesOut = Join-Path $speciesDir "best.pth"
    $evalOut = Join-Path $evalDir "deploy_ab_eval.json"
    $manifest = Join-Path $RepoRoot "data/reference/biochem_gnn_abcd_$Label.json"
    if (Test-Path $speciesOut) { Remove-Item $speciesOut -Force }
    if (Test-Path $evalOut) { Remove-Item $evalOut -Force }

    Clear-LegEnv
    Set-SharedRecipe

    switch ($Code) {
        "A" { }  # baseline control
        "B" {
            $env:SPECIES_CONTINUOUS_FP_WEIGHT = "$FpWeight"
            $env:SPECIES_CONTINUOUS_SCORE_CLOUT_W = "$ScoreClotW"
            $env:SPECIES_CONTINUOUS_MATURE_FP_EXEMPT = "1"
        }
        "C" {
            $env:SPECIES_GEOM_FEATS = "1"
        }
        "D" {
            $env:SPECIES_CONTINUOUS_FP_WEIGHT = "$FpWeight"
            $env:SPECIES_CONTINUOUS_SCORE_CLOUT_W = "$ScoreClotW"
            $env:SPECIES_CONTINUOUS_MATURE_FP_EXEMPT = "1"
            $env:SPECIES_GEOM_FEATS = "1"
        }
        default { throw "unknown leg code: $Code" }
    }

    Write-Host "[run] [$Label] train (fp_w=$($env:SPECIES_CONTINUOUS_FP_WEIGHT) score_clot_w=$($env:SPECIES_CONTINUOUS_SCORE_CLOUT_W) geom=$($env:SPECIES_GEOM_FEATS)) $Epochs ep" -ForegroundColor Cyan
    $null = Invoke-PythonRcCheck -Label "[$Label] train" -PyArgs @(
        "-m", "src.training.train_species_pushforward_continuous",
        "--phase", "biochem_gnn",
        "--anchors", $Anchors,
        "--val-anchor", $ValAnchor,
        "--epochs", "$Epochs",
        "--early-stop", "$EarlyStop",
        "--max-windows", "$MaxWindows",
        "--unroll", "10",
        "--lr", "$Lr",
        "--arch", "sage",
        "--init", "__fresh_no_init__",
        "--init-s1", "__fresh_no_init__",
        "--out", $speciesOut
    )

    $payload = @{
        name = "biochem_gnn_abcd_$Label"
        version = 1
        baseline = @{
            species_gnn_ckpt = (RelPath $speciesOut)
            viscosity_beta = $BetaCkpt
            kinematics_ckpt = "outputs/kinematics/kinematics_best.pth"
            train_val_anchor = $ValAnchor
            flow_modes = "kinematics"
            gamma_mode = "max"
            deploy_horizon = "full"
            clot_score = "guiding"
            pushforward_arch = "sage"
            species_channels = "11,7"
            loao_auto = "0"
        }
    }
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($manifest, ($payload | ConvertTo-Json -Depth 6), $utf8NoBom)

    # Leg env stays set: the deploy eval must rebuild the SAME features (geom) / dims.
    Write-Host "[run] [$Label] eval deploy_frozen" -ForegroundColor Cyan
    $null = Invoke-PythonRcCheck -Label "[$Label] eval" -PyArgs @(
        "scripts/eval_biochem_gnn_deploy_ab.py",
        "--manifest", $manifest,
        "--modes", "deploy_frozen",
        "--times", $Times,
        "--anchors", $Anchors,
        "--out", $evalOut
    )
}

$LegNames = @{ A = "A_baseline"; B = "B_footprint_sup"; C = "C_geom_feats"; D = "D_both" }
Write-Host "[i] abcd root: $RunRoot  legs: $($Legs -join ',')" -ForegroundColor DarkGray
foreach ($code in $Legs) {
    $c = $code.Trim().ToUpper()
    if (-not $LegNames.ContainsKey($c)) { throw "unknown leg: $code (use A,B,C,D)" }
    Train-Leg -Code $c -Label $LegNames[$c]
}

Clear-LegEnv
$null = Invoke-PythonRcCheck -Label "abcd summary" -PyArgs @(
    "scripts/summarize_species_abcd.py",
    "--run-root", (RelPath $RunRoot),
    "--val-anchor", $ValAnchor
)
Write-Host "[OK] A/B/C/D precision ladder done -> $RunRoot" -ForegroundColor Green
