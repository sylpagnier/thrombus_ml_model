# 6-fold LOAO species GNN train + deploy eval + patient004 smoke predict.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_species_gnn_loao.ps1"
#   powershell ... -Holdouts patient004 -SkipTrain   # eval only

param(
    [string] $Holdouts = "",
    [int] $Epochs = 40,
    [int] $EarlyStop = 18,
    [string] $InitCkpt = "outputs/biochem/species_snapshot_s34/best.pth",
    [string] $OutRoot = "outputs/biochem/species_gnn_loao",
    [switch] $SkipTrain,
    [switch] $SkipEval,
    [switch] $SkipPredict
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

if (-not $SkipTrain) {
    Write-Host "[NEW] LOAO train (s34 recipe, exclude holdout from train)" -ForegroundColor Cyan
    $trainArgs = @(
        "scripts/run_species_gnn_loao_train.py",
        "--epochs", "$Epochs",
        "--early-stop", "$EarlyStop",
        "--init", $InitCkpt,
        "--out-root", $OutRoot
    )
    if ($Holdouts.Trim()) { $trainArgs += @("--holdouts", $Holdouts) }
    Invoke-PythonRcCheck -Label "LOAO train" -PyArgs $trainArgs
}

$manifest = "data/reference/species_gnn_deploy_r4.json"
Write-Host "[NEW] Update deploy manifest (LOAO dir)" -ForegroundColor Cyan
Invoke-PythonRcCheck -Label "promote loao manifest" -PyArgs @(
    "scripts/promote_species_gnn_deploy.py",
    "--out", $manifest
)
# Patch loao_dir into manifest via python one-liner
python -c @"
import json
from pathlib import Path
p = Path('$manifest')
m = json.loads(p.read_text(encoding='utf-8'))
m['loao_dir'] = '$OutRoot'
m['phase'] = 'species_gnn_deploy_r4_loao'
p.write_text(json.dumps(m, indent=2), encoding='utf-8')
print('[OK] loao_dir in manifest')
"@

if (-not $SkipEval) {
    Write-Host "[NEW] LOAO deploy eval (per-fold ckpts)" -ForegroundColor Cyan
    Invoke-PythonRcCheck -Label "LOAO eval" -PyArgs @(
        "scripts/eval_t0_rung4_species_gnn_loao.py",
        "--manifest", $manifest,
        "--flow", "both",
        "--out", "outputs/biochem/species_gnn_deploy/loao_eval.json"
    )
}

if (-not $SkipPredict) {
    Write-Host "[NEW] Deploy predict smoke (patient004 LOAO fold)" -ForegroundColor Cyan
    Invoke-PythonRcCheck -Label "predict p004" -PyArgs @(
        "-m", "src.inference.predict_species_gnn_deploy",
        "--graph", "data/processed/graphs_biochem_anchors/patient004.pt",
        "--flow", "kinematics",
        "--loao",
        "--manifest", $manifest
    )
}

Write-Host "[OK] LOAO root=$OutRoot manifest=$manifest" -ForegroundColor Green
