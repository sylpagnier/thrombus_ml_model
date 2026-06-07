# Lane A deploy + mu CARRY on pred-kine dump anchors (deploy-faithful clot forecast).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_lane_a_deploy_carry.ps1" -Fresh
#   powershell ... -Phase deploy -InitCheckpoint outputs/.../warm/clot_phi_best.pth

param(
    [ValidateSet("warm", "deploy", "both")]
    [string] $Phase = "both",
    [string] $AnchorDir = "outputs\biochem\gnode10_sweep\anchors_gnode12_predkine_uvp",
    [string] $LegPrefix = "lane_a_deploy_carry",
    [int] $WarmEpochs = 16,
    [int] $DeployEpochs = 40,
    [string] $InitCheckpoint = "",
    [switch] $Fresh,
    [switch] $SkipEval,
    [switch] $SkipViz
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$anchorFull = Join-Path $RepoRoot $AnchorDir
if (-not (Test-Path $anchorFull)) {
    Write-Host "[ERR] Missing dump anchors: $AnchorDir" -ForegroundColor Red
    exit 1
}

$env:CLOT_PHI_ANCHOR_DIR = ($AnchorDir -replace '\\', '/')
$SweepRoot = "outputs/biochem/lane_a_deploy_carry"
$env:CLOT_PHI_SWEEP_DIR = $SweepRoot

. (Join-Path $PSScriptRoot "_lane_a_deploy_clot_base.ps1")
. (Join-Path $PSScriptRoot "_clot_forecast_deploy_carry_base.ps1")

function Invoke-LaneADeployCarryLeg {
    param(
        [string] $LegName,
        [string] $ForecastMask,
        [int] $Epochs,
        [int] $WarmupEpochs,
        [int] $FadeEpochs,
        [int] $WarmupSteps,
        [string] $InitCkpt = ""
    )

    $env:CLOT_PHI_SWEEP_LEG = $LegName
    $env:CLOT_FORECAST_MASK = $ForecastMask
    $env:CLOT_PHI_EPOCHS = "$Epochs"
    $env:CLOT_PHI_CARRY_GT_WARMUP_EPOCHS = "$WarmupEpochs"
    $env:CLOT_PHI_CARRY_GT_FADE_EPOCHS = "$FadeEpochs"
    $env:CLOT_PHI_CARRY_GT_WARMUP_STEPS = "$WarmupSteps"

    if ($InitCkpt) {
        $env:CLOT_PHI_INIT_CHECKPOINT = $InitCkpt
    } else {
        Remove-Item Env:CLOT_PHI_INIT_CHECKPOINT -ErrorAction SilentlyContinue
    }

    $legDir = Join-Path $RepoRoot "$SweepRoot\$LegName"
    New-Item -ItemType Directory -Force -Path $legDir | Out-Null
    if ($Fresh) {
        Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $legDir "clot_phi_best.pth")
        Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $legDir "clot_phi_train_log.jsonl")
    }

    Write-Host "[NEW] Lane A deploy_carry leg=$LegName mask=$ForecastMask ep=$Epochs" -ForegroundColor Cyan
    python -m src.training.train_clot_phi_simple
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $ckpt = Join-Path $legDir "clot_phi_best.pth"
    if (-not (Test-Path $ckpt)) {
        python scripts/recover_clot_phi_best_from_log.py --leg-dir "$SweepRoot/$LegName"
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    if (-not $SkipEval -and (Test-Path $ckpt)) {
        $evalOut = Join-Path $legDir "multi_anchor.jsonl"
        python scripts/eval_clot_phi_multi_anchor.py --checkpoint $ckpt --out $evalOut --anchor-dir $env:CLOT_PHI_ANCHOR_DIR
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    if (-not $SkipViz -and (Test-Path $ckpt)) {
        $vizDir = Join-Path $RepoRoot "outputs/biochem/viz/lane_a_deploy_carry"
        New-Item -ItemType Directory -Force -Path $vizDir | Out-Null
        $timelinePng = Join-Path $vizDir "${LegName}_patient007_timeline.png"
        $timelineJson = Join-Path $vizDir "${LegName}_patient007_timeline.jsonl"
        python -m src.evaluation.viz_clot_forecast_timeline --anchor patient007 --checkpoint $ckpt --keyframes 8 --out $timelinePng --summary-json $timelineJson
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}

$warmLeg = "${LegPrefix}_warm_target"
$deployLeg = "${LegPrefix}_deploy_pred"

if ($Phase -eq "warm" -or $Phase -eq "both") {
    Invoke-LaneADeployCarryLeg `
        -LegName $warmLeg `
        -ForecastMask "target" `
        -Epochs $WarmEpochs `
        -WarmupEpochs 999 `
        -FadeEpochs 0 `
        -WarmupSteps 2 `
        -InitCkpt $InitCheckpoint
}

$warmCkpt = Join-Path $RepoRoot "$SweepRoot\$warmLeg\clot_phi_best.pth"
$deployInit = $InitCheckpoint
if ($Phase -eq "both" -and (Test-Path $warmCkpt)) {
    $deployInit = $warmCkpt
}

if ($Phase -eq "deploy" -or $Phase -eq "both") {
    Invoke-LaneADeployCarryLeg `
        -LegName $deployLeg `
        -ForecastMask "deploy_pred" `
        -Epochs $DeployEpochs `
        -WarmupEpochs 0 `
        -FadeEpochs 12 `
        -WarmupSteps 0 `
        -InitCkpt $deployInit
}

Write-Host "[OK]  Lane A deploy_carry -> $SweepRoot" -ForegroundColor Green