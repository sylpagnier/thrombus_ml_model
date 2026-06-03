# GNODE ladder rung 9.9 — best-practice clot_band teacher + dump + clot-phi.
#
# Fixes vs naive go_passive_transport_clotband_focus.ps1:
#   - Init from canonical species teacher (after_94 archive), not polluted global best_high_mu
#   - 8h-ladder species weights (FI=3, Mat=2, PASSIVE_SPECIES_VAL)
#   - Dump from val-best teacher (best_high_mu), not final-epoch last.pth
#   - Default dump stride 72 (9.5 parity)
#
# Example:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_gnode99.ps1 -Fresh
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_gnode99.ps1 -Fresh `
#       -InitCkpt outputs\biochem\gnode_8h_ladder\checkpoints\after_94_biochem_teacher_last.pth

param(
    [switch] $Fresh,
    [string] $InitCkpt = "",
    [int] $TeacherEpochs = 12,
    [int] $ClotEpochs = 35,
    [int] $DumpStride = 72,
    [int] $DumpMinSteps = 4,
    [switch] $SkipViz,
    [switch] $TeacherOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_clot_phi_shared_env.ps1")
. (Join-Path $PSScriptRoot "_gnode_viz_helpers.ps1")

function Resolve-Gnode99InitCkpt {
    param([string] $UserPath)
    if ($UserPath -and (Test-Path (Join-Path $RepoRoot $UserPath))) {
        return $UserPath
    }
    $candidates = @(
        "outputs\biochem\gnode_8h_ladder\checkpoints\after_94_biochem_teacher_last.pth",
        "outputs\biochem\biochem_teacher_passive_species_locked.pth",
        "outputs\biochem\biochem_teacher_passive_align_locked.pth"
    )
    foreach ($rel in $candidates) {
        if (Test-Path (Join-Path $RepoRoot $rel)) {
            return $rel
        }
    }
    return $null
}

$initRel = Resolve-Gnode99InitCkpt -UserPath $InitCkpt
if (-not $initRel) {
    Write-Host "[ERR] No init checkpoint. Run go_gnode_8h_ladder.ps1 (9.4) first, or pass -InitCkpt <path>." -ForegroundColor Red
    exit 1
}
$initPath = Join-Path $RepoRoot $initRel

$OutRoot = "outputs\biochem\gnode_99"
if ($Fresh) {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $OutRoot
}
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$ckptArchive = Join-Path $OutRoot "checkpoints"
New-Item -ItemType Directory -Force -Path $ckptArchive | Out-Null

Write-Host "[NEW] GNODE 9.9 | init=$initRel | teacher=${TeacherEpochs}ep dump_stride=$DumpStride clot=${ClotEpochs}ep" -ForegroundColor Cyan

Copy-Item -Force $initPath (Join-Path $RepoRoot "outputs\biochem\biochem_teacher_best_high_mu.pth")

$env:BIOCHEM_PRESET = "passive_transport"
$env:BIOCHEM_STOCK_DEFAULTS = "0"
$env:BIOCHEM_NO_TEACHER_DEFAULTS = "1"
$env:BIOCHEM_RUN_NOTE = "gnode_99_clotband_focus"
$env:BIOCHEM_VAL_TIME_STRIDE = "10"
$env:BIOCHEM_TEACHER_VAL_EVERY = "1"
$env:BIOCHEM_TEACHER_EPOCHS = "$TeacherEpochs"
$env:BIOCHEM_EPOCHS = "$TeacherEpochs"
$env:BIOCHEM_CLI_TEACHER_EPOCHS = "$TeacherEpochs"
$env:BIOCHEM_DETACH_MACRO_STATE = "0"
$env:BIOCHEM_GT_KINE_VEL = "1"
$env:BIOCHEM_TEACHER_MU_RATIO_MAX = "1.0"
$env:BIOCHEM_TEACHER_FORCE_MIN = "1.0"

$env:BIOCHEM_TRAIN_MODE = "new"
$env:BIOCHEM_RESUME = "0"
$env:BIOCHEM_SKIP_PRETRAIN = "1"
$env:BIOCHEM_INIT_FROM_BEST = "1"
$env:BIOCHEM_REUSE_LAST_PRETRAIN = "1"
$env:BIOCHEM_STOP_AFTER_TEACHER = "1"

$env:BIOCHEM_TBPTT_MAX_WINDOW = "5"
$env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "4"
$env:BIOCHEM_DATALOADER_WORKERS = "0"
$env:BIOCHEM_PIN_MEMORY = "0"
$env:BIOCHEM_KIN_GRADIENT_CHECKPOINT = "1"
$env:BIOCHEM_ODE_GRADIENT_CHECKPOINT = "1"
$env:BIOCHEM_PASSIVE_ADR_BACKPROP = "0"
$env:BIOCHEM_PASSIVE_DATA_KINE_WEIGHT = "0.25"
$env:BIOCHEM_PASSIVE_DATA_BIO_WEIGHT = "1.0"
$env:BIOCHEM_DATA_BIO_MASK_MODE = "clot_band"
$env:BIOCHEM_PASSIVE_SPECIES_VAL = "1"
$env:BIOCHEM_DATA_BIO_FI_WEIGHT = "3.0"
$env:BIOCHEM_DATA_BIO_MAT_WEIGHT = "2.0"

python -m src.training.train_biochem_corrector --new --skip-pretrain --init-from-best --epochs $TeacherEpochs --save-best --run-name gnode_99_clotband_focus
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

foreach ($name in @("biochem_teacher_last.pth", "biochem_teacher_best_high_mu.pth")) {
    $src = Join-Path $RepoRoot "outputs\biochem\$name"
    if (Test-Path $src) {
        Copy-Item -Force $src (Join-Path $ckptArchive "after_99_$name")
    }
}

$bestMu = Join-Path $RepoRoot "outputs\biochem\biochem_teacher_best_high_mu.pth"
$lastMu = Join-Path $RepoRoot "outputs\biochem\biochem_teacher_last.pth"
if (-not (Test-Path $bestMu)) {
    Write-Host "[ERR] Missing biochem_teacher_best_high_mu.pth after train" -ForegroundColor Red
    exit 1
}
Write-Host "[i]  Dump uses val-best teacher (best_high_mu), not final epoch last.pth" -ForegroundColor Cyan
Copy-Item -Force $bestMu $lastMu
$Teacher = $lastMu

if (-not $SkipViz) {
    Invoke-BiochemTeacherSnapshot -Checkpoint $Teacher -Anchor patient007 -Label "gnode99_teacher"
    Invoke-BiochemTeacherClotbandViz -Checkpoint $Teacher -Anchor patient007 -TimeIndex -1 -Label "gnode99_teacher_raw"
}
if ($TeacherOnly) {
    Write-Host "[OK]  Teacher-only done. Checkpoints under $ckptArchive" -ForegroundColor Green
    exit 0
}

$AnchorDir = Join-Path $OutRoot ("anchors_stride_$DumpStride")
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
    $ti = 4
    if ($DumpMinSteps -ge 6) { $ti = -1 }
    Invoke-BiochemTeacherClotbandViz `
        -Checkpoint $Teacher `
        -Anchor patient007 `
        -AnchorDir $AnchorDir `
        -TimeIndex $ti `
        -Out (Join-Path $OutRoot "viz_teacher_clotband_p007_t${ti}.png")
}

$LegName = "gnode99_clotphi"
if ($ClotEpochs -le 0) {
    Write-Host "[skip] ClotEpochs=${ClotEpochs}: teacher + dump only." -ForegroundColor Yellow
    exit 0
}

Write-Host "[NEW] Train clot-phi ($ClotEpochs ep)" -ForegroundColor Cyan
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

python scripts/eval_clot_phi_multi_anchor.py `
    --checkpoint "outputs\biochem\passive_species_focus_compare\$LegName\clot_phi_best.pth" `
    --anchor-dir $AnchorDir `
    --out (Join-Path $OutRoot "multi_anchor.jsonl")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not $SkipViz) {
    $ClotCkpt = "outputs\biochem\passive_species_focus_compare\$LegName\clot_phi_best.pth"
    if (Test-Path (Join-Path $RepoRoot $ClotCkpt)) {
        $env:CLOT_PHI_ANCHOR_DIR = $AnchorDir
        Invoke-ClotPhiScatterViz -Checkpoint $ClotCkpt -Anchor patient007 -TimeIndex -1 `
            -Out (Join-Path $OutRoot "viz_clot_phi_p007_tfinal.png")
        Remove-Item Env:CLOT_PHI_ANCHOR_DIR -ErrorAction SilentlyContinue
    }
}

Write-Host "[OK]  GNODE 9.9 done. Gate: min F1 >= 0.26 and beat 9.5 (p007 ~0.63, min ~0.34)." -ForegroundColor Green
exit 0
