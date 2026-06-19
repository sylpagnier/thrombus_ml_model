# Moves 2+3+4 on the Mat+FG deploy GraphSAGE (A/B vs the fi_ablation winner).
#
#   Move 2 (precision): weighted Tversky footprint loss, extra FP penalty on wall nodes.
#   Move 3 (recall):    Tversky FN boost on lumen-band nodes (deep-lumen growth).
#   Move 4 (features):  deployable stagnation proxies (kine speed/shear/position) appended to inputs.
#
# Two legs, identical recipe except the moves (fair A/B). Both FRESH (move 4 changes input dim, so
# warm-start is invalid; fresh is also symmetric with the fi_ablation control).
#   baseline_matfg = Mat+FG, moves OFF   (reproduces the fi_ablation winner as a same-epoch control)
#   moves234_matfg = Mat+FG, moves ON
#
# Budget: ~4h GPU. Default 40 ep / leg, early-stop 15; physics-readout leg is heavier but skips the
# per-epoch deploy clot eval (val_anchor not excluded, score_clot_weight=0) so epochs stay fast.
#
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_species_moves234.ps1 [-Epochs 40] [-SkipBaseline]

param(
    [int] $Epochs = 40,
    [int] $EarlyStop = 15,
    [int] $MaxWindows = 40,
    [double] $Lr = 3e-4,
    [string] $Anchors = "patient001,patient002,patient003,patient004,patient006,patient007",
    [switch] $SkipBaseline
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$RunRoot = Join-Path $RepoRoot "outputs/biochem/biochem_gnn/moves234"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

function RelPath([string]$AbsPath) {
    return (Resolve-Path $AbsPath).Path.Substring($RepoRoot.Length).TrimStart('\').Replace('\', '/')
}

# Clear all move/physics knobs so a leg starts from a known state.
function Clear-MoveEnv {
    foreach ($k in @(
        "SPECIES_STAGNATION_FEATS",
        "SPECIES_CONTINUOUS_PHYSICS_READOUT",
        "SPECIES_FOOTPRINT_TVERSKY",
        "SPECIES_FOOTPRINT_TVERSKY_ALPHA",
        "SPECIES_FOOTPRINT_TVERSKY_BETA",
        "SPECIES_FOOTPRINT_WALL_FP_W",
        "SPECIES_FOOTPRINT_LUMEN_FN_W",
        "SPECIES_FOOTPRINT_BCE_BLEND",
        "SPECIES_CONTINUOUS_PHI_LOSS_WEIGHT",
        "SPECIES_CONTINUOUS_MU_LOSS_WEIGHT"
    )) { Remove-Item "env:$k" -ErrorAction SilentlyContinue }
}

function Train-Leg([string]$Label, [bool]$MovesOn) {
    $legDir = Join-Path $RunRoot $Label
    $speciesDir = Join-Path $legDir "species"
    $evalDir = Join-Path $legDir "eval"
    New-Item -ItemType Directory -Force -Path $speciesDir, $evalDir | Out-Null
    $speciesOut = Join-Path $speciesDir "best.pth"
    $evalOut = Join-Path $evalDir "deploy_ab_eval.json"
    $manifest = Join-Path $RepoRoot "data/reference/biochem_gnn_moves234_$Label.json"
    if (Test-Path $speciesOut) { Remove-Item $speciesOut -Force }
    if (Test-Path $evalOut) { Remove-Item $evalOut -Force }

    # Shared deploy recipe (matches fi_ablation: Mat+FG, sage, kine-flow deploy).
    Clear-MoveEnv
    $env:BIOCHEM_PUSHFORWARD_SPECIES_CHANNELS = "11,7"
    $env:SPECIES_PUSHFORWARD_ARCH = "sage"
    $env:SPECIES_TRAIN_VEL_SOURCE = "gt"
    $env:SPECIES_ROLLOUT_DEPLOY_FAITHFUL = "1"
    $env:SPECIES_ROLLOUT_VEL_SOURCE = "kinematics"
    $env:SPECIES_ROLLOUT_PIN_OTHER = "rest"
    $env:SPECIES_ROLLOUT_IC_SOURCE = "resting"

    if ($MovesOn) {
        $env:SPECIES_STAGNATION_FEATS = "1"            # move 4
        $env:SPECIES_CONTINUOUS_PHYSICS_READOUT = "1"  # activate the differentiable footprint loss hook
        $env:SPECIES_FOOTPRINT_TVERSKY = "1"           # moves 2+3
        $env:SPECIES_FOOTPRINT_TVERSKY_ALPHA = "0.7"   # FP weight (precision)
        $env:SPECIES_FOOTPRINT_TVERSKY_BETA = "0.3"    # FN weight (recall)
        $env:SPECIES_FOOTPRINT_WALL_FP_W = "2.0"       # extra wall-FP penalty (move 2)
        $env:SPECIES_FOOTPRINT_LUMEN_FN_W = "2.0"      # extra lumen-FN penalty (move 3)
        $env:SPECIES_FOOTPRINT_BCE_BLEND = "0.25"
        $env:SPECIES_CONTINUOUS_PHI_LOSS_WEIGHT = "1.0"
        $env:SPECIES_CONTINUOUS_MU_LOSS_WEIGHT = "0.25"
    }

    Write-Host "[run] [$Label] train Mat+FG moves_on=$MovesOn ($Epochs ep)" -ForegroundColor Cyan
    $null = Invoke-PythonRcCheck -Label "[$Label] train" -PyArgs @(
        "-m", "src.training.train_species_pushforward_continuous",
        "--phase", "biochem_gnn",
        "--anchors", $Anchors,
        "--val-anchor", "patient007",
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
        name = "biochem_gnn_moves234_$Label"
        version = 1
        baseline = @{
            species_gnn_ckpt = (RelPath $speciesOut)
            viscosity_beta = "outputs/biochem/biochem_gnn/locked/viscosity_beta.pth"
            kinematics_ckpt = "outputs/kinematics/kinematics_best.pth"
            train_val_anchor = "patient007"
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

    # NOTE: move env stays set here so the eval rebuilds the SAME features (move 4) / dims.
    Write-Host "[run] [$Label] eval deploy_frozen" -ForegroundColor Cyan
    $null = Invoke-PythonRcCheck -Label "[$Label] eval" -PyArgs @(
        "scripts/eval_biochem_gnn_deploy_ab.py",
        "--manifest", $manifest,
        "--modes", "deploy_frozen",
        "--times", "53,200",
        "--anchors", $Anchors,
        "--out", $evalOut
    )
    return $evalOut
}

Write-Host "[i] moves234 root: $RunRoot" -ForegroundColor DarkGray
if (-not $SkipBaseline) { Train-Leg -Label "baseline_matfg" -MovesOn $false | Out-Null }
Train-Leg -Label "moves234_matfg" -MovesOn $true | Out-Null

Clear-MoveEnv
$null = Invoke-PythonRcCheck -Label "moves234 summary" -PyArgs @(
    "scripts/summarize_species_moves234.py",
    "--run-root", (RelPath $RunRoot)
)
Write-Host "[OK] moves234 A/B done -> $RunRoot" -ForegroundColor Green
