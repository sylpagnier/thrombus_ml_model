# L2-heavy finetune after foundation checkpoint (resume + lower LR).
param(
    [string]$Resume = "latest",
    [double]$FinetuneLr = 1e-5,
    [int]$AdamEpochs = 70
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "=== Kinematics L2-heavy finetune (resume=$Resume, lr=$FinetuneLr) ===" -ForegroundColor Cyan

& python -m src.training.train_kinematics_predictor `
    --resume $Resume `
    --geometry-phase l2_heavy `
    --hard-mining-start-epoch 20 `
    --finetune-lr $FinetuneLr `
    --adam-epochs $AdamEpochs
