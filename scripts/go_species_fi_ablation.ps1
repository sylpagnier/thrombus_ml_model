# Species channel ablation: is FI useful? Mat-only vs Mat+FG vs fi_mat vs fi_mat+FG.
#
# All legs FRESH-init (no warm start) for a FAIR comparison -- the locked warm-start ckpt is
# FI+Mat (2-ch output) and the partial loader copies output rows by POSITION, which would
# scramble FI weights into Mat/FG for the FI-dropped legs and bias the result. Fresh = symmetric.
#
# Legs (BIOCHEM_PUSHFORWARD_SPECIES_CHANNELS, canonical order):
#   mat        = 11           (Mat only)
#   mat_fg     = 11,7         (Mat + fibrinogen)
#   fi_mat     = 8,11         (FI + Mat = current baseline scope)
#   fi_mat_fg  = 8,11,7       (FI + Mat + fibrinogen)
#
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_species_fi_ablation.ps1 [-Epochs 12] [-Fresh]

param(
    [int] $Epochs = 12,
    [int] $EarlyStop = 6,
    [int] $MaxWindows = 40,
    [double] $Lr = 3e-4,
    [string] $Anchors = "patient001,patient002,patient003,patient004,patient006,patient007",
    [switch] $Fresh
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$RunRoot = Join-Path $RepoRoot "outputs/biochem/biochem_gnn/fi_ablation"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

function RelPath([string]$AbsPath) {
    return (Resolve-Path $AbsPath).Path.Substring($RepoRoot.Length).TrimStart('\').Replace('\', '/')
}

function Train-Leg([string]$Label, [string]$Channels) {
    $legDir = Join-Path $RunRoot $Label
    $speciesDir = Join-Path $legDir "species"
    $evalDir = Join-Path $legDir "eval"
    New-Item -ItemType Directory -Force -Path $speciesDir, $evalDir | Out-Null
    $speciesOut = Join-Path $speciesDir "best.pth"
    $evalOut = Join-Path $evalDir "deploy_ab_eval.json"
    $manifest = Join-Path $RepoRoot "data/reference/biochem_gnn_fi_ablation_$Label.json"

    if ($Fresh) {
        if (Test-Path $speciesOut) { Remove-Item $speciesOut -Force }
        if (Test-Path $evalOut) { Remove-Item $evalOut -Force }
    }

    $env:BIOCHEM_PUSHFORWARD_SPECIES_CHANNELS = $Channels
    $env:SPECIES_PUSHFORWARD_ARCH = "sage"
    $env:SPECIES_TRAIN_VEL_SOURCE = "gt"
    $env:SPECIES_ROLLOUT_DEPLOY_FAITHFUL = "1"
    $env:SPECIES_ROLLOUT_VEL_SOURCE = "kinematics"
    $env:SPECIES_ROLLOUT_PIN_OTHER = "rest"
    $env:SPECIES_ROLLOUT_IC_SOURCE = "resting"

    Write-Host "[run] [$Label] train channels=$Channels FRESH ($Epochs ep)" -ForegroundColor Cyan
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
        name = "biochem_gnn_fi_ablation_$Label"
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
            species_channels = $Channels
            loao_auto = "0"
        }
    }
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($manifest, ($payload | ConvertTo-Json -Depth 6), $utf8NoBom)

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

Write-Host "[i] FI ablation root: $RunRoot" -ForegroundColor DarkGray
Train-Leg -Label "mat"       -Channels "11"     | Out-Null
Train-Leg -Label "mat_fg"    -Channels "11,7"   | Out-Null
Train-Leg -Label "fi_mat"    -Channels "8,11"   | Out-Null
Train-Leg -Label "fi_mat_fg" -Channels "8,11,7" | Out-Null

$null = Invoke-PythonRcCheck -Label "fi ablation summary" -PyArgs @(
    "scripts/summarize_species_fi_ablation.py",
    "--run-root", (RelPath $RunRoot)
)
Write-Host "[OK] FI ablation done -> $RunRoot" -ForegroundColor Green
