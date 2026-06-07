# CAVO Stage G1 (one-step rolling): input physics band + hard mu projection + optional mu carry.
# Legacy name kept: go_clot_deploy_phase1.ps1  (see DEPLOY_ARCHITECTURE.md Stage G1).#   powershell -File .\scripts\go_gnode12_lane_a.ps1 -SkipMuUnlock -SkipClot -SkipGate
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_deploy_phase1.ps1" -Fresh
#   powershell ... -InitCheckpoint outputs/biochem/autonomy_clot_8h/.../a04_target_long/clot_phi_best.pth

param(
    [string] $LegName = "g1_one_step_input",
    [int] $Epochs = 24,
    [string] $InitCheckpoint = "",
    [string] $AnchorDir = "data/processed/graphs_biochem_anchors",
    [switch] $Fresh,
    [switch] $SkipEval,
    [switch] $SkipViz
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")
. (Join-Path $PSScriptRoot "_clot_deploy_phase1_base.ps1")

$anchorFull = Join-Path $RepoRoot $AnchorDir
if (-not (Test-Path $anchorFull)) {
    Write-Host "[ERR] Missing anchors: $AnchorDir" -ForegroundColor Red
    exit 1
}

$env:CLOT_PHI_ANCHOR_DIR = ($AnchorDir -replace '\\', '/')
$env:CLOT_PHI_SWEEP_DIR = "outputs/biochem/clot_deploy"
$env:CLOT_PHI_SWEEP_LEG = $LegName
$env:CLOT_PHI_EPOCHS = "$Epochs"

if ($InitCheckpoint) {
    $env:CLOT_PHI_INIT_CHECKPOINT = $InitCheckpoint
} else {
    Remove-Item Env:CLOT_PHI_INIT_CHECKPOINT -ErrorAction SilentlyContinue
}

$legDir = Join-Path $RepoRoot "outputs/biochem/clot_deploy/$LegName"
New-Item -ItemType Directory -Force -Path $legDir | Out-Null
if ($Fresh) {
    Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $legDir "clot_phi_best.pth")
    Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $legDir "clot_phi_train_log.jsonl")
    Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $legDir "multi_anchor.jsonl")
}

Write-Host ""
Write-Host "[NEW] CAVO G1 one-step leg=$LegName ep=$Epochs mask=input proj=1 dgamma=1 bulk=$($env:CLOT_PHI_MESH_BULK_LAMBDA)" -ForegroundColor Cyan
Write-Host "[i]  anchors=$($env:CLOT_PHI_ANCHOR_DIR) init=$InitCheckpoint" -ForegroundColor DarkGray

Invoke-PythonRcCheck -m src.training.train_clot_phi_simple -Label "CAVO phase1 train"

$ckpt = Join-Path $legDir "clot_phi_best.pth"
if (-not (Test-Path $ckpt)) {
    Invoke-PythonRcCheck scripts/recover_clot_phi_best_from_log.py --leg-dir "outputs/biochem/clot_deploy/$LegName" -Label "recover phase1 ckpt"
}

if (-not $SkipEval -and (Test-Path $ckpt)) {
    Invoke-PythonRcCheck scripts/eval_clot_phi_multi_anchor.py `
        --checkpoint $ckpt `
        --out (Join-Path $legDir "multi_anchor.jsonl") `
        --anchor-dir $env:CLOT_PHI_ANCHOR_DIR `
        -Label "phase1 eval"
}

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
        -Label "phase1 timeline viz"
}

Write-Host "[OK]  Phase 1 done -> outputs/biochem/clot_deploy/$LegName" -ForegroundColor Green
