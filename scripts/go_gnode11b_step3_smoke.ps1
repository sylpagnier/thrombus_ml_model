# Rung 11b: step-3 corrector smoke -- teacher + Phase 3 with Kendall multitask (not data-only).
#
# Plumbing gate: training completes, COMPLEXITY_STEP=3, LOSS_DATA_ONLY=0, corrector val rows.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_gnode11b_step3_smoke.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_gnode11b_step3_smoke.ps1 -TeacherEpochs 2 -CorrectorEpochs 4
#
# After run:
#   python scripts/check_gnode11_corrector_smoke_gate.py --step3 --archive-dir outputs/biochem/gnode10_sweep/gnode11_step3_smoke

param(
    [string] $InitCkpt = "",
    [int] $TeacherEpochs = 2,
    [int] $CorrectorEpochs = 4,
    [string] $RunNote = "gnode11_step3_smoke"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")
. (Join-Path $PSScriptRoot "_gnode11_env.ps1")

$initPath = Resolve-Gnode10K5Ckpt -UserPath $InitCkpt
if (-not $initPath) {
    Write-Host "[ERR] K5 teacher ckpt missing. Run go_gnode10_sweep.ps1 or pass -InitCkpt." -ForegroundColor Red
    exit 1
}

$ArchiveDir = Join-Path $RepoRoot "outputs\biochem\gnode10_sweep\gnode11_step3_smoke"
New-Item -ItemType Directory -Force -Path $ArchiveDir | Out-Null
$env:BIOCHEM_ARCHIVE_CHECKPOINT_DIR = $ArchiveDir

Set-Gnode11CorrectorSmokeEnv -ComplexityStep "3" -RunNote $RunNote `
    -TeacherEpochs $TeacherEpochs -CorrectorEpochs $CorrectorEpochs

Copy-Item -Force $initPath (Join-Path $RepoRoot "outputs\biochem\biochem_teacher_best_high_mu.pth")

Write-Host "[NEW] GNODE 11b step-3 corrector smoke" -ForegroundColor Cyan
Write-Host "[i]  init=$initPath" -ForegroundColor DarkGray
Write-Host "[i]  teacher=${TeacherEpochs}ep corrector=${CorrectorEpochs}ep | GT_KINE_VEL=0 | STOP_AFTER_TEACHER=0 | COMPLEXITY_STEP=3 | LOSS_DATA_ONLY=0" -ForegroundColor DarkGray
Write-Host "[i]  archive=$ArchiveDir" -ForegroundColor DarkGray

$rc = Invoke-PythonRc -m src.training.train_biochem_corrector --new --skip-pretrain --init-from-best `
    --save-best --run-name $RunNote
if ($rc -ne 0) { exit $rc }

Write-Host "[NEW] gate check" -ForegroundColor Cyan
$gateRc = Invoke-PythonRc scripts/check_gnode11_corrector_smoke_gate.py --step3 --archive-dir $ArchiveDir
if ($gateRc -ne 0) {
    Write-Host "[WARN] gate check failed (see run.jsonl under outputs/reports/training/biochem/)" -ForegroundColor Yellow
    exit $gateRc
}

Write-Host "[OK]  GNODE 11b step-3 smoke complete." -ForegroundColor Green
Write-Host "[i]  ckpt dir: $ArchiveDir" -ForegroundColor DarkGray
Write-Host "[i]  next: Phase II.0 pseudo bank or longer step-3 finetune" -ForegroundColor DarkGray