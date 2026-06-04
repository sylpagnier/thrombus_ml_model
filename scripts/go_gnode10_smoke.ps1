# Rung 10 smoke: 3ep predicted-kine teacher (fixed recipe). Full sweep: go_gnode10_sweep.ps1
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_gnode10_smoke.ps1

param(
    [string] $InitCkpt = "",
    [int] $Epochs = 3
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")
. (Join-Path $PSScriptRoot "_gnode10_env.ps1")

$initPath = Resolve-Gnode10InitCkpt -UserPath $InitCkpt
if (-not $initPath) {
    Write-Host "[ERR] No init checkpoint." -ForegroundColor Red
    exit 1
}

Clear-Gnode10BiochemEnv
Set-Gnode10PredictedKineBaseEnv -RunNote "gnode10_predicted_kine_smoke" -Epochs $Epochs -OomSafe
Apply-Gnode10LegOverrides -Leg @{
    Title = "smoke"
    TeacherForceMin = 0.5
    KineWeight = 0.25
    TrainKinLora = $true
    TbpttWindow = 5
}

Copy-Item -Force $initPath (Join-Path $RepoRoot "outputs\biochem\biochem_teacher_best_high_mu.pth")

Write-Host "[NEW] GNODE 10 smoke (${Epochs}ep, predicted kine, PASSIVE isolate)" -ForegroundColor Cyan
$rc = Invoke-PythonRc -m src.training.train_biochem_corrector --new --skip-pretrain --init-from-best `
    --epochs $Epochs --save-best --run-name gnode10_predicted_kine_smoke
if ($rc -ne 0) { exit $rc }

Write-Host "[OK]  Smoke done. Check run.jsonl for val_species_fi_log_mae and flow_trivial=0" -ForegroundColor Green
