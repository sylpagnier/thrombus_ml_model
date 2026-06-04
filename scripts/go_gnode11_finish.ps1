# Rung 11 finish (Phase II.0): step-2 bridge teacher + corrector with nonzero pseudo supervision.
#
# Completes rung 11 after 11a/11b plumbing smokes. Metrics are not optimized here; gate checks
# pseudo_w > 0, corrector val rows, and species FI sanity.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_gnode11_finish.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_gnode11_finish.ps1 -TeacherEpochs 8 -CorrectorEpochs 12
#
# After run:
#   python scripts/check_gnode11_finish_gate.py --archive-dir outputs/biochem/gnode10_sweep/gnode11_finish

param(
    [string] $InitCkpt = "",
    [int] $TeacherEpochs = 8,
    [int] $CorrectorEpochs = 12,
    [string] $RunNote = "gnode11_finish",
    [string] $PseudoMinTeacherMuScore = "-2.0",
    [switch] $SkipGate
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")
. (Join-Path $PSScriptRoot "_gnode11_env.ps1")

$initPath = Resolve-Gnode11InitCkpt -UserPath $InitCkpt
if (-not $initPath) {
    Write-Host "[ERR] Init teacher ckpt missing. Run go_gnode10_sweep.ps1 or pass -InitCkpt." -ForegroundColor Red
    exit 1
}

$ArchiveDir = Join-Path $RepoRoot "outputs\biochem\gnode10_sweep\gnode11_finish"
New-Item -ItemType Directory -Force -Path $ArchiveDir | Out-Null
$env:BIOCHEM_ARCHIVE_CHECKPOINT_DIR = $ArchiveDir

Set-Gnode11FinishEnv -RunNote $RunNote -TeacherEpochs $TeacherEpochs -CorrectorEpochs $CorrectorEpochs `
    -PseudoMinTeacherMuScore $PseudoMinTeacherMuScore

Copy-Item -Force $initPath (Join-Path $RepoRoot "outputs\biochem\biochem_teacher_best_high_mu.pth")

Write-Host "[NEW] GNODE 11 finish (Phase II.0 pseudo bank)" -ForegroundColor Cyan
Write-Host "[i]  init=$initPath" -ForegroundColor DarkGray
Write-Host "[i]  teacher=${TeacherEpochs}ep corrector=${CorrectorEpochs}ep | step2_bridge | PSEUDO_MIN=$PseudoMinTeacherMuScore" -ForegroundColor DarkGray
Write-Host "[i]  archive=$ArchiveDir" -ForegroundColor DarkGray

$rc = Invoke-PythonRc -m src.training.train_biochem_corrector --new --skip-pretrain --init-from-best `
    --save-best --run-name $RunNote
if ($rc -ne 0) { exit $rc }

foreach ($name in @(
        "biochem_best_high_mu.pth",
        "biochem_best.pth",
        "biochem_teacher_best_high_mu.pth",
        "biochem_latest_checkpoint.pth",
        "biochem_teacher_last.pth"
    )) {
    $src = Join-Path $RepoRoot "outputs\biochem\$name"
    if (Test-Path $src) {
        Copy-Item -Force $src (Join-Path $ArchiveDir $name)
        Write-Host "[OK]  archived $name -> gnode11_finish/" -ForegroundColor Green
    }
}

if ($SkipGate) {
    Write-Host "[OK]  training complete (gate skipped)." -ForegroundColor Green
    exit 0
}

Write-Host "[NEW] finish gate check" -ForegroundColor Cyan
$gateRc = Invoke-PythonRc scripts/check_gnode11_finish_gate.py --archive-dir $ArchiveDir `
    --min-corrector-val 3 --min-pseudo-w 0.01
if ($gateRc -ne 0) {
    Write-Host "[WARN] finish gate failed (see run.jsonl under outputs/reports/training/biochem/)" -ForegroundColor Yellow
    exit $gateRc
}

Write-Host "[OK]  GNODE 11 finish complete." -ForegroundColor Green
Write-Host "[i]  ckpt dir: $ArchiveDir" -ForegroundColor DarkGray
