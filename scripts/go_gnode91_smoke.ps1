# GNODE ladder rung 9.1: forward smoke (GT COMSOL flow, DEQ skipped) + headless viz.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_gnode91_smoke.ps1"
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_gnode91_smoke.ps1" -InteractiveViz

param(
    [switch] $InteractiveViz
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_gnode_viz_helpers.ps1")

$env:BIOCHEM_PRESET = "passive_transport"
Set-GnodeGtFlowVizEnv
$env:BIOCHEM_STOP_AFTER_TEACHER = "1"
$env:BIOCHEM_TEACHER_EPOCHS = "1"
$env:BIOCHEM_EPOCHS = "1"
$env:BIOCHEM_SKIP_PRETRAIN = "1"
$env:BIOCHEM_REUSE_LAST_PRETRAIN = "1"
$env:BIOCHEM_RESUME = "0"
$env:BIOCHEM_VAL_TIME_STRIDE = "10"
$env:BIOCHEM_TEACHER_VAL_EVERY = "1"
$env:BIOCHEM_PASSIVE_ADR_BACKPROP = "0"
$env:BIOCHEM_RUN_NOTE = "gnode91_smoke"

Write-Host "[NEW] GNODE 9.1 smoke (1ep teacher, GT vel, DEQ skipped)" -ForegroundColor Cyan

. (Join-Path $PSScriptRoot "_python_rc.ps1")
$rc = Invoke-PythonRc -PyArgs @("-m", "src.training.train_biochem_corrector", "--new", "--skip-pretrain", "--epochs", "1", "--run-name", "gnode91_smoke")
if ($rc -ne 0) { exit $rc }

$Teacher = "outputs\biochem\biochem_teacher_last.pth"
if (-not (Test-Path $Teacher)) {
    Write-Host "[ERR] Missing teacher ckpt: $Teacher" -ForegroundColor Red
    exit 1
}

Invoke-GnodeRungVizCheckup -RungLabel "gnode91" -TeacherCheckpoint $Teacher -InteractiveTeacher:$InteractiveViz

Write-Host "[i]  Full-field snapshot + clot-band teacher panels under outputs/biochem/viz/" -ForegroundColor DarkGray

Write-Host "[OK]  9.1 smoke done. PNG under outputs/biochem/viz/ ; run.jsonl under outputs/reports/training/biochem/" -ForegroundColor Green
