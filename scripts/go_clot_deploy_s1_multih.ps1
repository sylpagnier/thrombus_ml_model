# Stage S1: multi-horizon from t=0 (anchor times, no carry). Init from S0 if available.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_deploy_s1_multih.ps1" -Fresh
#   ... -InitCheckpoint outputs/biochem/clot_deploy/s0_static_final/clot_phi_best.pth

param(
    [string] $LegName = "s1_from_t0",
    [int] $Epochs = 28,
    [string] $InitCheckpoint = "outputs/biochem/clot_deploy/s0_static_final/clot_phi_best.pth",
    [string] $AnchorDir = "data/processed/graphs_biochem_anchors",
    [switch] $Fresh,
    [switch] $SkipEval,
    [switch] $SkipViz
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")
. (Join-Path $PSScriptRoot "_clot_deploy_s1_multih_base.ps1")

$anchorFull = Join-Path $RepoRoot $AnchorDir
if (-not (Test-Path $anchorFull)) {
    Write-Host "[ERR] Missing anchors: $AnchorDir" -ForegroundColor Red
    exit 1
}

$env:CLOT_PHI_ANCHOR_DIR = ($AnchorDir -replace '\\', '/')
$env:CLOT_PHI_SWEEP_DIR = "outputs/biochem/clot_deploy"
$env:CLOT_PHI_SWEEP_LEG = $LegName
$env:CLOT_PHI_EPOCHS = "$Epochs"

if ($InitCheckpoint -and (Test-Path (Join-Path $RepoRoot $InitCheckpoint))) {
    $env:CLOT_PHI_INIT_CHECKPOINT = $InitCheckpoint
} else {
    Remove-Item Env:CLOT_PHI_INIT_CHECKPOINT -ErrorAction SilentlyContinue
    Write-Host "[WARN] No S0 init ckpt; cold start" -ForegroundColor Yellow
}

$legDir = Join-Path $RepoRoot "outputs/biochem/clot_deploy/$LegName"
New-Item -ItemType Directory -Force -Path $legDir | Out-Null
if ($Fresh) {
    Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $legDir "clot_phi_best.pth")
    Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $legDir "clot_phi_train_log.jsonl")
}

Write-Host ""
Write-Host "[NEW] CAVO S1 from_t0 leg=$LegName ep=$Epochs init=$InitCheckpoint" -ForegroundColor Cyan

Invoke-PythonRcCheck -m src.training.train_clot_phi_simple -Label "CAVO S1 train"

$ckpt = Join-Path $legDir "clot_phi_best.pth"
if (-not $SkipViz -and (Test-Path $ckpt)) {
    $vizDir = Join-Path $RepoRoot "outputs/biochem/viz/clot_deploy"
    New-Item -ItemType Directory -Force -Path $vizDir | Out-Null
    $timeline = Join-Path $vizDir "${LegName}_patient007_timeline.png"
    Invoke-PythonRcCheck -m src.evaluation.viz_clot_forecast_timeline `
        --anchor patient007 `
        --checkpoint $ckpt `
        --keyframes 8 `
        --out $timeline `
        --summary-json (Join-Path $vizDir "${LegName}_patient007_timeline.jsonl") `
        -Label "S1 timeline viz"
}

Write-Host "[OK]  S1 done -> outputs/biochem/clot_deploy/$LegName" -ForegroundColor Green
