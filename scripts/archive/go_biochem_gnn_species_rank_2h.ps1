# Species rank screen + cumulative ladder (~2h budget).
#
# Phase 1: fi_mat baseline + fi_mat + each addon species (quick screen).
# Phase 2: rank addons by deploy guiding score.
# Phase 3: cumulative ladder fi_mat + top1, +top2, ...
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_biochem_gnn_species_rank_2h.ps1
#   powershell ... -ScreenOnly
#   powershell ... -LadderOnly -TopN 3

param(
    [int] $ScreenEpochs = 6,
    [int] $ScreenEarlyStop = 4,
    [int] $ScreenMaxWindows = 35,
    [int] $LadderEpochs = 10,
    [int] $LadderEarlyStop = 6,
    [int] $LadderMaxWindows = 45,
    [int] $TopN = 4,
    [double] $Lr = 3e-4,
    [string] $Anchors = "patient001,patient002,patient003,patient004,patient006,patient007",
    [switch] $Fresh,
    [switch] $ScreenOnly,
    [switch] $LadderOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$RunRoot = Join-Path $RepoRoot "outputs/biochem/biochem_gnn/species_rank_2h"
$ScreenRoot = Join-Path $RunRoot "screen"
$LadderRoot = Join-Path $RunRoot "ladder"
New-Item -ItemType Directory -Force -Path $ScreenRoot, $LadderRoot | Out-Null

$InitWarm = Join-Path $RepoRoot "outputs/biochem/biochem_gnn/locked/species_gnn_best.pth"
$BetaCkpt = Join-Path $RepoRoot "outputs/biochem/biochem_gnn/locked/viscosity_beta.pth"
if (-not (Test-Path $InitWarm)) { throw "missing init ckpt: $InitWarm" }
if (-not (Test-Path $BetaCkpt)) { throw "missing beta ckpt: $BetaCkpt" }

# fi_mat base + single-addon candidates (bulk indices 0-11 except FI=8, Mat=11).
$AddonChannels = @(0, 1, 2, 3, 4, 5, 6, 7, 9, 10)
$AddonNames = @{
    0 = "RP"; 1 = "AP"; 2 = "APR"; 3 = "APS"; 4 = "PT"
    5 = "T"; 6 = "AT"; 7 = "FG"; 9 = "M"; 10 = "Mas"
}
$FiMatBase = @(8, 11)

function RelPath([string]$AbsPath) {
    return (Resolve-Path $AbsPath).Path.Substring($RepoRoot.Length).TrimStart('\').Replace('\', '/')
}

function ChannelListStr([int[]]$Channels) {
    return ($Channels | ForEach-Object { "$_" }) -join ","
}

function Train-Leg(
    [string]$Phase,
    [string]$Label,
    [int[]]$Channels,
    [int]$Epochs,
    [int]$EarlyStop,
    [int]$MaxWindows,
    [string]$InitCkpt
) {
    $root = if ($Phase -eq "screen") { $ScreenRoot } else { $LadderRoot }
    $legDir = Join-Path $root $Label
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

    Write-Host "[run] [$Phase/$Label] channels=$chStr ($Epochs ep)" -ForegroundColor Cyan
    $null = Invoke-PythonRcCheck -Label "[$Phase/$Label] train" -PyArgs @(
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
        "--init", $InitCkpt,
        "--out", $speciesOut
    )

    $payload = @{
        name = "biochem_gnn_species_rank_$Label"
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
    Write-Host "[run] [$Phase/$Label] eval deploy_frozen" -ForegroundColor Cyan
    $null = Invoke-PythonRcCheck -Label "[$Phase/$Label] eval" -PyArgs @(
        "scripts/eval_biochem_gnn_deploy_ab.py",
        "--manifest", $manifest,
        "--modes", "deploy_frozen",
        "--times", "53,200",
        "--anchors", $Anchors,
        "--out", $evalOut
    )
    return @{
        label = $Label
        channels = @($Channels)
        eval = $evalOut
        manifest = $manifest
        species = $speciesOut
    }
}

if (-not $LadderOnly) {
    Write-Host "[i] Phase 1: species screen ($ScreenEpochs ep, max_windows=$ScreenMaxWindows)" -ForegroundColor DarkGray
    $screenLegs = @()

    $base = Train-Leg -Phase "screen" -Label "fi_mat" -Channels $FiMatBase `
        -Epochs $ScreenEpochs -EarlyStop $ScreenEarlyStop -MaxWindows $ScreenMaxWindows -InitCkpt $InitWarm
    $screenLegs += @{
        label = "fi_mat"
        channels = $FiMatBase
        addon_channel = $null
        eval = $base.eval
    }

    foreach ($ch in $AddonChannels) {
        $name = $AddonNames[$ch]
        $label = "fi_mat_$name"
        $channels = @(8, 11, $ch)
        $leg = Train-Leg -Phase "screen" -Label $label -Channels $channels `
            -Epochs $ScreenEpochs -EarlyStop $ScreenEarlyStop -MaxWindows $ScreenMaxWindows -InitCkpt $InitWarm
        $screenLegs += @{
            label = $label
            channels = $channels
            addon_channel = $ch
            eval = $leg.eval
        }
    }

    $screenManifest = @{
        legs = $screenLegs
        screen_epochs = $ScreenEpochs
        screen_max_windows = $ScreenMaxWindows
    }
    $screenManifestPath = Join-Path $ScreenRoot "screen_manifest.json"
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($screenManifestPath, ($screenManifest | ConvertTo-Json -Depth 6), $utf8NoBom)
    Write-Host "[OK] screen manifest -> $screenManifestPath" -ForegroundColor Green
}

if ($ScreenOnly) {
    $SummaryJson = Join-Path $RunRoot "species_rank_screen_summary.json"
    $SummaryMd = Join-Path $RunRoot "species_rank_screen_report.md"
    $null = Invoke-PythonRcCheck -Label "species rank screen summary" -PyArgs @(
        "scripts/summarize_species_rank_ladder.py",
        "--screen-root", $ScreenRoot,
        "--baseline-eval", (Join-Path $ScreenRoot "fi_mat/eval/deploy_ab_eval.json"),
        "--out-json", $SummaryJson,
        "--out-md", $SummaryMd,
        "--top-n", "$TopN"
    )
    Write-Host "[OK] screen summary -> $SummaryJson" -ForegroundColor Green
    exit 0
}

# Phase 2: rank screen legs
$RankJson = Join-Path $RunRoot "species_rank_screen_summary.json"
$RankMd = Join-Path $RunRoot "species_rank_screen_report.md"
$null = Invoke-PythonRcCheck -Label "species rank screen summary" -PyArgs @(
    "scripts/summarize_species_rank_ladder.py",
    "--screen-root", $ScreenRoot,
    "--baseline-eval", (Join-Path $ScreenRoot "fi_mat/eval/deploy_ab_eval.json"),
    "--out-json", $RankJson,
    "--out-md", $RankMd,
    "--top-n", "$TopN"
)
$rankData = Get-Content $RankJson -Raw | ConvertFrom-Json
$topAddons = @($rankData.top_addon_channels | ForEach-Object { [int]$_ })
if ($topAddons.Count -eq 0) {
    throw "no ranked addons; run screen first or check $RankJson"
}
$useN = [Math]::Min($TopN, $topAddons.Count)
Write-Host "[i] Phase 3: ladder top $useN addons: $($topAddons[0..($useN-1)] -join ',')" -ForegroundColor DarkGray

$ladderLegs = @()
for ($k = 1; $k -le $useN; $k++) {
    $subset = $topAddons[0..($k - 1)]
    $channels = @($FiMatBase + $subset)
    $nameParts = @("fi_mat") + ($subset | ForEach-Object { $AddonNames[$_] })
    $label = ($nameParts -join "_plus_")
    $leg = Train-Leg -Phase "ladder" -Label $label -Channels $channels `
        -Epochs $LadderEpochs -EarlyStop $LadderEarlyStop -MaxWindows $LadderMaxWindows -InitCkpt $InitWarm
    $ladderLegs += @{
        label = $label
        channels = $channels
        n_addons = $k
        eval = $leg.eval
    }
}

$ladderManifest = @{
    legs = $ladderLegs
    top_addon_channels = $topAddons[0..($useN - 1)]
    ladder_epochs = $LadderEpochs
    ladder_max_windows = $LadderMaxWindows
}
$ladderManifestPath = Join-Path $LadderRoot "ladder_manifest.json"
$utf8NoBom2 = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText($ladderManifestPath, ($ladderManifest | ConvertTo-Json -Depth 6), $utf8NoBom2)

$FinalJson = Join-Path $RunRoot "species_rank_ladder_summary.json"
$FinalMd = Join-Path $RunRoot "species_rank_ladder_report.md"
$null = Invoke-PythonRcCheck -Label "species rank ladder summary" -PyArgs @(
    "scripts/summarize_species_rank_ladder.py",
    "--screen-root", $ScreenRoot,
    "--ladder-root", $LadderRoot,
    "--baseline-eval", (Join-Path $ScreenRoot "fi_mat/eval/deploy_ab_eval.json"),
    "--out-json", $FinalJson,
    "--out-md", $FinalMd,
    "--top-n", "$TopN"
)
Write-Host "[OK] final summary -> $FinalJson" -ForegroundColor Green
Write-Host "[OK] final report  -> $FinalMd" -ForegroundColor Green
