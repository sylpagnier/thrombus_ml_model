# Focused passive teacher run (non-interactive), configurable bio loss weight.
#
# Example:
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_passive_transport_focus.ps1" -RunNote passive_focus_lbio_on -TeacherEpochs 10 -BioWeight 1.0

param(
    [string] $RunNote = "passive_focus",
    [int] $TeacherEpochs = 10,
    [double] $BioWeight = 1.0,
    [double] $KineWeight = 0.25
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$env:BIOCHEM_PRESET = "passive_transport"
$env:BIOCHEM_STOCK_DEFAULTS = "0"
$env:BIOCHEM_NO_TEACHER_DEFAULTS = "1"
$env:BIOCHEM_RUN_NOTE = $RunNote
$env:BIOCHEM_VAL_TIME_STRIDE = "10"
$env:BIOCHEM_TEACHER_VAL_EVERY = "1"
$env:BIOCHEM_TEACHER_EPOCHS = "$TeacherEpochs"
$env:BIOCHEM_EPOCHS = "$TeacherEpochs"
$env:BIOCHEM_CLI_TEACHER_EPOCHS = "$TeacherEpochs"
$env:BIOCHEM_DETACH_MACRO_STATE = "0"
$env:BIOCHEM_GT_KINE_VEL = "1"
$env:BIOCHEM_TEACHER_MU_RATIO_MAX = "1.0"
$env:BIOCHEM_TRAIN_MODE = "new"
$env:BIOCHEM_RESUME = "0"
$env:BIOCHEM_SKIP_PRETRAIN = "1"
$env:BIOCHEM_INIT_FROM_BEST = "1"
$env:BIOCHEM_REUSE_LAST_PRETRAIN = "1"
$env:BIOCHEM_TBPTT_MAX_WINDOW = "5"
$env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "4"
$env:BIOCHEM_PASSIVE_ADR_BACKPROP = "0"
$env:BIOCHEM_PASSIVE_DATA_BIO_WEIGHT = "$BioWeight"
$env:BIOCHEM_PASSIVE_DATA_KINE_WEIGHT = "$KineWeight"

Write-Host "[NEW] Passive transport focus run=$RunNote ep=$TeacherEpochs bio_w=$BioWeight kine_w=$KineWeight" -ForegroundColor Cyan
python -m src.training.train_biochem_corrector --new --skip-pretrain --init-from-best --epochs $TeacherEpochs --save-best --run-name $RunNote
exit $LASTEXITCODE
