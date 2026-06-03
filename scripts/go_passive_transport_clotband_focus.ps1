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
    [int] $DumpMinSteps = 4,
    [switch] $SkipViz
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_clot_phi_shared_env.ps1")
. (Join-Path $PSScriptRoot "_gnode_viz_helpers.ps1")

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
if (-not $SkipViz) {
    Invoke-BiochemTeacherSnapshot -Checkpoint $Teacher -Anchor patient007 -Label "passive_clotband_teacher"
    Invoke-BiochemTeacherClotbandViz -Checkpoint $Teacher -Anchor patient007 -TimeIndex -1 -Label "passive_clotband_teacher_raw"
}
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

if (-not $SkipViz) {
    Invoke-ClotPhiMaskViz -Anchor patient007 -TimeIndex 0 -Out (Join-Path $OutRoot "viz_mask_p007_t0.png")
    Invoke-ClotPhiMaskViz -Anchor patient007 -TimeIndex -1 -Out (Join-Path $OutRoot "viz_mask_p007_tfinal.png")
    Invoke-BiochemTeacherClotbandViz `
        -Checkpoint $Teacher `
        -Anchor patient007 `
        -AnchorDir $AnchorDir `
        -TimeIndex 4 `
        -Out (Join-Path $OutRoot "viz_teacher_clotband_p007_t4.png")
}

$LegName = "clotband_focus"
$LegDir = Join-Path $OutRoot $LegName
New-Item -ItemType Directory -Force -Path $LegDir | Out-Null

if ($ClotEpochs -le 0) {
    Write-Host "[skip] ClotEpochs=${ClotEpochs}: teacher + dump only (use -ClotEpochs 20 for clot-phi)." -ForegroundColor Yellow
} else {
    Write-Host "[NEW] Train clot-phi on clotband anchors ($ClotEpochs ep)" -ForegroundColor Cyan
    $clotArgs = @(
        "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", ".\scripts\go_clot_phi_from_anchor_dir.ps1",
        "-AnchorDir", $AnchorDir,
        "-LegName", $LegName,
        "-Epochs", $ClotEpochs
    )
    if ($SkipViz) { $clotArgs += "-SkipViz" }
    & powershell @clotArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    if (-not $SkipViz) {
        $ClotCkpt = "outputs\biochem\passive_species_focus_compare\$LegName\clot_phi_best.pth"
        if (Test-Path $ClotCkpt) {
            $env:CLOT_PHI_ANCHOR_DIR = $AnchorDir
            Invoke-ClotPhiScatterViz -Checkpoint $ClotCkpt -Anchor patient007 -TimeIndex -1 `
                -Out (Join-Path $OutRoot "viz_clot_phi_p007_tfinal.png")
            Remove-Item Env:CLOT_PHI_ANCHOR_DIR -ErrorAction SilentlyContinue
        }
    }
}

exit 0

