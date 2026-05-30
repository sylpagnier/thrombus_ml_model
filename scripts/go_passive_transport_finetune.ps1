# Light finetune of passive-transport teacher (resume from biochem_teacher_last).
#
#   powershell -File .\scripts\go_passive_transport_finetune.ps1

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$env:BIOCHEM_PRESET = "passive_transport"
$env:BIOCHEM_STOCK_DEFAULTS = "0"
$env:BIOCHEM_NO_TEACHER_DEFAULTS = "1"
$env:BIOCHEM_RUN_NOTE = "passive_transport_finetune"
$env:BIOCHEM_VAL_TIME_STRIDE = "10"
$env:BIOCHEM_TEACHER_VAL_EVERY = "2"
$env:BIOCHEM_TEACHER_EPOCHS = "6"
$env:BIOCHEM_EPOCHS = "6"
$env:BIOCHEM_CLI_TEACHER_EPOCHS = "6"
$env:BIOCHEM_DETACH_MACRO_STATE = "0"
$env:BIOCHEM_GT_KINE_VEL = "1"
$env:BIOCHEM_TEACHER_MU_RATIO_MAX = "1.0"
# Finetune from existing teacher weights (no interactive resume prompt).
$env:BIOCHEM_TRAIN_MODE = "new"
$env:BIOCHEM_RESUME = "0"
$env:BIOCHEM_SKIP_PRETRAIN = "1"
$env:BIOCHEM_INIT_FROM_BEST = "1"
$env:BIOCHEM_REUSE_LAST_PRETRAIN = "1"
$env:BIOCHEM_TBPTT_MAX_WINDOW = "5"
$env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "4"

Write-Host "[NEW] Passive transport finetune (6 ep, resume)..." -ForegroundColor Cyan
python -m src.training.train_biochem_corrector --epochs 6 --save-best --run-name passive_transport_finetune
exit $LASTEXITCODE
