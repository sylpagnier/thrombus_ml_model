# Rung 11a: corrector smoke -- teacher (anchors) + Phase 3 corrector, predicted kine, step-2 bridge.
#
# Plumbing gate (not metric optimization): training completes, Phase 3 runs, losses finite.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_gnode11_corrector_smoke.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_gnode11_corrector_smoke.ps1 -TeacherEpochs 2 -CorrectorEpochs 4
#
# After run:
#   python scripts/check_gnode11_corrector_smoke_gate.py

param(
    [string] $InitCkpt = "",
    [int] $TeacherEpochs = 2,
    [int] $CorrectorEpochs = 4,
    [string] $RunNote = "gnode11_corrector_smoke"
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

$ArchiveDir = Join-Path $RepoRoot "outputs\biochem\gnode10_sweep\gnode11_corrector_smoke"
New-Item -ItemType Directory -Force -Path $ArchiveDir | Out-Null
$env:BIOCHEM_ARCHIVE_CHECKPOINT_DIR = $ArchiveDir

Set-Gnode11CorrectorSmokeEnv -ComplexityStep "2" -RunNote $RunNote -TeacherEpochs $TeacherEpochs -CorrectorEpochs $CorrectorEpochs

Copy-Item -Force $initPath (Join-Path $RepoRoot "outputs\biochem\biochem_teacher_best_high_mu.pth")

Write-Host "[NEW] GNODE 11a corrector smoke" -ForegroundColor Cyan
Write-Host "[i]  init=$initPath" -ForegroundColor DarkGray
Write-Host "[i]  teacher=${TeacherEpochs}ep corrector=${CorrectorEpochs}ep | GT_KINE_VEL=0 | STOP_AFTER_TEACHER=0 | step2_bridge" -ForegroundColor DarkGray
Write-Host "[i]  archive=$ArchiveDir" -ForegroundColor DarkGray

$rc = Invoke-PythonRc -m src.training.train_biochem_corrector --new --skip-pretrain --init-from-best `
    --save-best --run-name $RunNote
if ($rc -ne 0) { exit $rc }

Write-Host "[NEW] gate check" -ForegroundColor Cyan
$gateRc = Invoke-PythonRc scripts/check_gnode11_corrector_smoke_gate.py --step2 --archive-dir $ArchiveDir
if ($gateRc -ne 0) {
    Write-Host "[WARN] gate check failed (see run.jsonl under outputs/reports/training/biochem/)" -ForegroundColor Yellow
    exit $gateRc
}

Write-Host "[OK]  GNODE 11a corrector smoke complete." -ForegroundColor Green
Write-Host "[i]  ckpt dir: $ArchiveDir" -ForegroundColor DarkGray
Write-Host "[i]  next: rung 11b step-3 smoke (COMPLEXITY_STEP=3) or Phase II.0 pseudo bank" -ForegroundColor DarkGray
