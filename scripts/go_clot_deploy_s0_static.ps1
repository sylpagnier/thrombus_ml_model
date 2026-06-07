# Stage S0: static clot-at-T_final (localization gate). Run after Phase 0.2 dump.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_deploy_s0_static.ps1" -Fresh

param(
    [string] $LegName = "s0_static_final",
    [int] $Epochs = 32,
    [string] $AnchorDir = "data/processed/graphs_biochem_anchors",
    [switch] $Fresh,
    [switch] $SkipEval,
    [switch] $SkipViz
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")
. (Join-Path $PSScriptRoot "_clot_deploy_s0_static_base.ps1")

$anchorFull = Join-Path $RepoRoot $AnchorDir
if (-not (Test-Path $anchorFull)) {
    Write-Host "[ERR] Missing anchors: $AnchorDir" -ForegroundColor Red
    exit 1
}

$env:CLOT_PHI_ANCHOR_DIR = ($AnchorDir -replace '\\', '/')
$env:CLOT_PHI_SWEEP_DIR = "outputs/biochem/clot_deploy"
$env:CLOT_PHI_SWEEP_LEG = $LegName
$env:CLOT_PHI_EPOCHS = "$Epochs"
Remove-Item Env:CLOT_PHI_INIT_CHECKPOINT -ErrorAction SilentlyContinue

$legDir = Join-Path $RepoRoot "outputs/biochem/clot_deploy/$LegName"
New-Item -ItemType Directory -Force -Path $legDir | Out-Null
if ($Fresh) {
    Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $legDir "clot_phi_best.pth")
    Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $legDir "clot_phi_train_log.jsonl")
    Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $legDir "multi_anchor.jsonl")
}

Write-Host ""
Write-Host "[NEW] CAVO S0 static_final leg=$LegName ep=$Epochs schedule=static_final proj=1" -ForegroundColor Cyan
Write-Host "[i]  anchors=$($env:CLOT_PHI_ANCHOR_DIR)" -ForegroundColor DarkGray

Invoke-PythonRcCheck -m src.training.train_clot_phi_simple -Label "CAVO S0 train"

$ckpt = Join-Path $legDir "clot_phi_best.pth"
if (-not (Test-Path $ckpt)) {
    Invoke-PythonRcCheck scripts/recover_clot_phi_best_from_log.py --leg-dir "outputs/biochem/clot_deploy/$LegName" -Label "recover S0 ckpt"
}

if (-not $SkipEval -and (Test-Path $ckpt)) {
    Invoke-PythonRcCheck scripts/eval_clot_phi_multi_anchor.py `
        --checkpoint $ckpt `
        --out (Join-Path $legDir "multi_anchor.jsonl") `
        --anchor-dir $env:CLOT_PHI_ANCHOR_DIR `
        -Label "S0 eval"
}

if (-not $SkipViz -and (Test-Path $ckpt)) {
    $vizDir = Join-Path $RepoRoot "outputs/biochem/viz/clot_deploy"
    New-Item -ItemType Directory -Force -Path $vizDir | Out-Null
    $png = Join-Path $vizDir "${LegName}_patient007_tfinal_fullmesh.png"
    Invoke-PythonRcCheck -m src.evaluation.viz_clot_phi_simple `
        --anchor patient007 `
        --checkpoint $ckpt `
        --time-index -1 `
        --plot-mode scatter `
        --out $png `
        -Label "S0 fullmesh viz"
}

Write-Host "[OK]  S0 done -> outputs/biochem/clot_deploy/$LegName" -ForegroundColor Green
