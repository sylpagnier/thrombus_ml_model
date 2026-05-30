# Passive-transport teacher run with *clot decision band* focused species supervision.
#
# Steps:
#  1) train teacher (BIOCHEM_PRESET=passive_transport)
#  2) dump teacher species onto anchors (time-stride)
#  3) train clot-phi on dumped anchors + multi-anchor eval
#
# Example:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_passive_transport_clotband_focus.ps1 -Fresh

param(
    [switch] $Fresh,
    [int] $TeacherEpochs = 8,
    [int] $ClotEpochs = 20,
    [int] $DumpStride = 36,
    [int] $DumpMinSteps = 4
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_clot_phi_shared_env.ps1")

$OutRoot = "outputs\biochem\passive_species_clotband_focus"
if ($Fresh) {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $OutRoot
}
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

Write-Host "[NEW] Passive clotband focus teacher ep=$TeacherEpochs" -ForegroundColor Cyan

$env:BIOCHEM_PRESET = "passive_transport"
$env:BIOCHEM_STOCK_DEFAULTS = "0"
$env:BIOCHEM_NO_TEACHER_DEFAULTS = "1"
$env:BIOCHEM_RUN_NOTE = "passive_transport_clotband_focus"
$env:BIOCHEM_VAL_TIME_STRIDE = "10"
$env:BIOCHEM_TEACHER_VAL_EVERY = "1"
$env:BIOCHEM_TEACHER_EPOCHS = "$TeacherEpochs"
$env:BIOCHEM_EPOCHS = "$TeacherEpochs"
$env:BIOCHEM_CLI_TEACHER_EPOCHS = "$TeacherEpochs"
$env:BIOCHEM_DETACH_MACRO_STATE = "0"
$env:BIOCHEM_GT_KINE_VEL = "1"
$env:BIOCHEM_TEACHER_MU_RATIO_MAX = "1.0"

# Non-interactive trainer mode
$env:BIOCHEM_TRAIN_MODE = "new"
$env:BIOCHEM_RESUME = "0"
$env:BIOCHEM_SKIP_PRETRAIN = "1"
$env:BIOCHEM_INIT_FROM_BEST = "1"
$env:BIOCHEM_REUSE_LAST_PRETRAIN = "1"

$env:BIOCHEM_TBPTT_MAX_WINDOW = "5"
$env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "4"
$env:BIOCHEM_PASSIVE_ADR_BACKPROP = "0"

# Species weights (keep same recipe as earlier A/B probes)
$env:BIOCHEM_PASSIVE_DATA_KINE_WEIGHT = "0.25"
$env:BIOCHEM_PASSIVE_DATA_BIO_WEIGHT = "1.0"

# This is the new feature: focus species residual on clot-phi decision band.
$env:BIOCHEM_DATA_BIO_MASK_MODE = "clot_band"

python -m src.training.train_biochem_corrector --new --skip-pretrain --init-from-best --epochs $TeacherEpochs --save-best --run-name passive_transport_clotband_focus
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$Teacher = "outputs\biochem\biochem_teacher_last.pth"
if (-not (Test-Path $Teacher)) {
    Write-Host "[ERR] Missing teacher ckpt: $Teacher" -ForegroundColor Red
    exit 1
}

$AnchorDir = Join-Path $OutRoot ("anchors_clotband_" + $DumpStride)
Write-Host "[NEW] Dump teacher species -> $AnchorDir" -ForegroundColor Cyan
python scripts/dump_teacher_species_to_anchors.py `
    --teacher $Teacher `
    --out-dir $AnchorDir `
    --device cuda `
    --time-stride $DumpStride `
    --min-steps $DumpMinSteps `
    --force
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$LegName = "clotband_focus"
$LegDir = Join-Path $OutRoot $LegName
New-Item -ItemType Directory -Force -Path $LegDir | Out-Null
Write-Host "[NEW] Train clot-phi on clotband anchors ($ClotEpochs ep)" -ForegroundColor Cyan

powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_phi_from_anchor_dir.ps1" `
    -AnchorDir $AnchorDir `
    -LegName $LegName `
    -Epochs $ClotEpochs

exit $LASTEXITCODE

