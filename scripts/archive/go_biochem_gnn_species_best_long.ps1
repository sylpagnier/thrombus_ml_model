# Dedicated longer run for the best species scopes from the corrected rank screen.
#
# Scopes (FI/Mat base = [8,11], canonical FI->Mat->extras order):
#   fi_mat            [8,11]        reference baseline
#   fi_mat_FG         [8,11,7]      screen rank 1 (best holdout)
#   fi_mat_FG_APR     [8,11,7,2]    ladder peak
#   fi_mat_thrombin   [8,11,5]      prior A/B winner
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_biochem_gnn_species_best_long.ps1
#   powershell ... -Epochs 30 -MaxWindows 90 -Fresh

param(
    [int] $Epochs = 26,
    [int] $EarlyStop = 12,
    [int] $MaxWindows = 80,
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

$RunRoot = Join-Path $RepoRoot "outputs/biochem/biochem_gnn/species_best_long"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

$InitWarm = Join-Path $RepoRoot "outputs/biochem/biochem_gnn/locked/species_gnn_best.pth"
$BetaCkpt = Join-Path $RepoRoot "outputs/biochem/biochem_gnn/locked/viscosity_beta.pth"
if (-not (Test-Path $InitWarm)) { throw "missing init ckpt: $InitWarm" }
if (-not (Test-Path $BetaCkpt)) { throw "missing beta ckpt: $BetaCkpt" }

# label -> canonical channel list (FI=8, Mat=11 first)
$Scopes = [ordered]@{
    "fi_mat"               = @(8, 11)
    "fi_mat_FG"            = @(8, 11, 7)
    "fi_mat_FG_APR"        = @(8, 11, 7, 2)
    "fi_mat_thrombin"      = @(8, 11, 5)
    "fi_mat_FG_APR_thrombin" = @(8, 11, 7, 2, 5)
}

function RelPath([string]$AbsPath) {
    return (Resolve-Path $AbsPath).Path.Substring($RepoRoot.Length).TrimStart('\').Replace('\', '/')
}

function ChannelListStr([int[]]$Channels) {
    return ($Channels | ForEach-Object { "$_" }) -join ","
}

function Train-Leg([string]$Label, [int[]]$Channels) {
    $legDir = Join-Path $RunRoot $Label
    $speciesDir = Join-Path $legDir "species"
    $evalDir = Join-Path $legDir "eval"
    New-Item -ItemType Directory -Force -Path $speciesDir, $evalDir | Out-Null
    $speciesOut = Join-Path $speciesDir "best.pth"
    $evalOut = Join-Path $evalDir "deploy_ab_eval.json"
    $manifest = Join-Path $legDir "manifest.json"
    $chStr = ChannelListStr $Channels

    if ($Fresh) {
        if (Test-Path $speciesOut) { Remove-Item $speciesOut -Force }
        if (Test-Path $evalOut) { Remove-Item $evalOut -Force }
    }

    Remove-Item Env:BIOCHEM_PUSHFORWARD_SPECIES_SCOPE -ErrorAction SilentlyContinue
    $env:BIOCHEM_PUSHFORWARD_SPECIES_CHANNELS = $chStr
    $env:SPECIES_PUSHFORWARD_ARCH = "sage"
    python -c "from src.biochem_gnn.config import apply_train_recipe_env; apply_train_recipe_env()" | Out-Null
    $env:SPECIES_TRAIN_VEL_SOURCE = "gt"
    $env:SPECIES_ROLLOUT_DEPLOY_FAITHFUL = "1"
    $env:SPECIES_ROLLOUT_VEL_SOURCE = "kinematics"
    $env:SPECIES_ROLLOUT_PIN_OTHER = "rest"
    $env:SPECIES_ROLLOUT_IC_SOURCE = "resting"

    Write-Host "[run] [$Label] channels=$chStr ($Epochs ep, max_windows=$MaxWindows)" -ForegroundColor Cyan
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
        "--init", $InitWarm,
        "--out", $speciesOut
    )

    $payload = @{
        name = "biochem_gnn_species_best_long_$Label"
        version = 1
        baseline = @{
            species_gnn_ckpt = (RelPath $speciesOut)
            viscosity_beta = (RelPath $BetaCkpt)
            kinematics_ckpt = "outputs/kinematics/kinematics_best.pth"
            train_val_anchor = "patient007"
            flow_modes = "kinematics"
            gamma_mode = "max"
            deploy_horizon = "full"
            clot_score = "guiding"
            pushforward_arch = "sage"
            species_channels = @($Channels)
            loao_auto = "0"
        }
    }
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($manifest, ($payload | ConvertTo-Json -Depth 6), $utf8NoBom)

    $env:BIOCHEM_PUSHFORWARD_SPECIES_CHANNELS = $chStr
    Write-Host "[run] [$Label] eval deploy_frozen" -ForegroundColor Cyan
    $null = Invoke-PythonRcCheck -Label "[$Label] eval" -PyArgs @(
        "scripts/eval_biochem_gnn_deploy_ab.py",
        "--manifest", $manifest,
        "--modes", "deploy_frozen",
        "--times", "53,200",
        "--anchors", $Anchors,
        "--out", $evalOut
    )
    return @{ label = $Label; channels = @($Channels); eval = $evalOut }
}

Write-Host "[i] species best long root: $RunRoot" -ForegroundColor DarkGray
$legs = @()
foreach ($label in $Scopes.Keys) {
    $leg = Train-Leg -Label $label -Channels $Scopes[$label]
    $legs += @{ label = $label; channels = $Scopes[$label]; addon_channel = $null; eval = $leg.eval }
}

# Reuse screen manifest format so the rank summarizer can read these legs.
$screenManifest = @{ legs = $legs; epochs = $Epochs; max_windows = $MaxWindows }
$manifestPath = Join-Path $RunRoot "screen_manifest.json"
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText($manifestPath, ($screenManifest | ConvertTo-Json -Depth 6), $utf8NoBom)

$SummaryJson = Join-Path $RunRoot "species_best_long_summary.json"
$SummaryMd = Join-Path $RunRoot "species_best_long_report.md"
$null = Invoke-PythonRcCheck -Label "species best long summary" -PyArgs @(
    "scripts/summarize_species_rank_ladder.py",
    "--screen-root", $RunRoot,
    "--baseline-eval", (Join-Path $RunRoot "fi_mat/eval/deploy_ab_eval.json"),
    "--out-json", $SummaryJson,
    "--out-md", $SummaryMd,
    "--top-n", "4"
)
Write-Host "[OK] summary -> $SummaryJson" -ForegroundColor Green
Write-Host "[OK] report  -> $SummaryMd" -ForegroundColor Green
