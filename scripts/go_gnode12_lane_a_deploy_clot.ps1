# GNODE 12 Lane A deploy clot-phi: forecast one-step head on pred-kine dump anchors.
#
# Curriculum (default): target_mask warm-start -> deploy_pred finetune with mesh aux.
# Uses existing dump cache (SkipDump default); no GT hybrid / no joint bio.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_gnode12_lane_a_deploy_clot.ps1"
#   powershell ... -Phase deploy -InitCheckpoint outputs/.../target_warm/clot_phi_best.pth
#   powershell ... -Phase target -Epochs 20 -Fresh

param(
    [ValidateSet("target", "deploy", "both")]
    [string] $Phase = "both",
    [string] $AnchorDir = "outputs\biochem\gnode10_sweep\anchors_gnode12_predkine_uvp",
    [string] $LegPrefix = "lane_a_deploy",
    [int] $TargetEpochs = 20,
    [int] $DeployEpochs = 30,
    [string] $InitCheckpoint = "",
    [double] $MeshAux = 0.6,
    [double] $MeshBulk = 0.18,
    [int] $Hidden = 32,
    [int] $Depth = 2,
    [switch] $Fresh,
    [switch] $SkipEval,
    [switch] $SkipViz
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")
. (Join-Path $PSScriptRoot "_lane_a_deploy_clot_base.ps1")
. (Join-Path $PSScriptRoot "_gnode_viz_helpers.ps1")

$anchorFull = Join-Path $RepoRoot $AnchorDir
if (-not (Test-Path $anchorFull)) {
    Write-Host "[ERR] Missing dump anchors: $AnchorDir (run go_gnode12_lane_a.ps1 -SkipMuUnlock or full dump first)" -ForegroundColor Red
    exit 1
}

$ptCount = (Get-ChildItem $anchorFull -Filter *.pt -ErrorAction SilentlyContinue).Count
if ($ptCount -lt 1) {
    Write-Host "[ERR] No .pt graphs in $AnchorDir" -ForegroundColor Red
    exit 1
}

$env:CLOT_PHI_ANCHOR_DIR = ($AnchorDir -replace '\\', '/')
$SweepRoot = "outputs/biochem/lane_a_deploy_clot"
$env:CLOT_PHI_SWEEP_DIR = $SweepRoot
$env:CLOT_PHI_HIDDEN = "$Hidden"
$env:CLOT_PHI_MLP_DEPTH = "$Depth"

function Invoke-LaneADeployLeg {
    param(
        [string] $LegName,
        [string] $ForecastMask,
        [int] $Epochs,
        [double] $Aux,
        [double] $Bulk,
        [string] $InitCkpt = ""
    )

    $env:CLOT_PHI_SWEEP_LEG = $LegName
    $env:CLOT_FORECAST_MASK = $ForecastMask
    $env:CLOT_PHI_EPOCHS = "$Epochs"
    if ($Aux -gt 0) { $env:CLOT_PHI_MESH_AUX_LAMBDA = "$Aux" } else { Remove-Item Env:CLOT_PHI_MESH_AUX_LAMBDA -ErrorAction SilentlyContinue }
    if ($Bulk -gt 0) { $env:CLOT_PHI_MESH_BULK_LAMBDA = "$Bulk" } else { Remove-Item Env:CLOT_PHI_MESH_BULK_LAMBDA -ErrorAction SilentlyContinue }

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
        Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $legDir "multi_anchor.jsonl")
    }

    Write-Host ""
    Write-Host "[NEW] Lane A deploy leg=$LegName mask=$ForecastMask ep=$Epochs mesh_aux=$Aux mesh_bulk=$Bulk" -ForegroundColor Cyan
    Write-Host "[i]  anchors=$($env:CLOT_PHI_ANCHOR_DIR) init=$InitCkpt" -ForegroundColor DarkGray

    Invoke-PythonRcCheck -m src.training.train_clot_phi_simple -Label "lane A deploy train $LegName"
    $ckpt = Join-Path $legDir "clot_phi_best.pth"
    if (-not (Test-Path $ckpt)) {
        Write-Host "[WARN] no val ckpt; recovering best rank epoch from log" -ForegroundColor Yellow
        Invoke-PythonRcCheck scripts/recover_clot_phi_best_from_log.py `
            --leg-dir "$SweepRoot/$LegName" -Label "recover ckpt $LegName"
    }

    if (-not $SkipEval -and (Test-Path $ckpt)) {
        Invoke-PythonRcCheck scripts/eval_clot_phi_multi_anchor.py `
            --checkpoint $ckpt `
            --out (Join-Path $legDir "multi_anchor.jsonl") `
            --anchor-dir $env:CLOT_PHI_ANCHOR_DIR `
            -Label "eval $LegName"
    }

    if (-not $SkipViz -and (Test-Path $ckpt)) {
        Invoke-ClotPhiScatterViz -Checkpoint $ckpt -Anchor patient007 -TimeIndex -1 `
            -Out "outputs/biochem/viz/${LegName}_p007_tfinal.png"
    }

    return $ckpt
}

$warmCkpt = ""
if ($Phase -eq "target" -or $Phase -eq "both") {
    $warmCkpt = Invoke-LaneADeployLeg `
        -LegName "${LegPrefix}_target_warm" `
        -ForecastMask "target" `
        -Epochs $TargetEpochs `
        -Aux 0.3 `
        -Bulk 0.0 `
        -InitCkpt $InitCheckpoint
}

$deployInit = $InitCheckpoint
if ($Phase -eq "both" -and $warmCkpt -and (Test-Path $warmCkpt)) {
    $deployInit = $warmCkpt
}

if ($Phase -eq "deploy" -or $Phase -eq "both") {
    Invoke-LaneADeployLeg `
        -LegName "${LegPrefix}_deploy_pred" `
        -ForecastMask "deploy_pred" `
        -Epochs $DeployEpochs `
        -Aux $MeshAux `
        -Bulk $MeshBulk `
        -InitCkpt $deployInit | Out-Null
}

Write-Host "[OK]  Lane A deploy clot done -> $SweepRoot" -ForegroundColor Green
