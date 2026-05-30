# M3 alignment probe: co-train L_Data_Bio + masked transport_only ADR on union TBPTT mask.
#
# Prereq: outputs/biochem/biochem_teacher_phaseB_ramp1_last.pth
#   (powershell -File .\scripts\go_phaseB_xy_passive.ps1 -Ramp1Epochs 3 -Ramp2Epochs 0)
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_m3_align_probe.ps1"
#   powershell ... -Epochs 12 -InitCkpt outputs/biochem/biochem_teacher_phaseB_ramp1_last.pth
#
# After run:
#   python scripts/check_m3_align_gate.py --run-note m3_align_transport_union
#   python scripts/audit_passive_adr_alignment.py --anchor patient007 --all-formulations

param(
    [int] $Epochs = 12,
    [string] $RunNote = "m3_align_transport_union",
    [string] $InitCkpt = "outputs/biochem/biochem_teacher_phaseB_ramp1_last.pth",
    [string] $AdrWeight = "1e-4",
    [switch] $SkipAudit
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")
. (Join-Path $PSScriptRoot "_m3_align_env.ps1")

$initPath = Join-Path $RepoRoot $InitCkpt
if (-not (Test-Path $initPath)) {
    Write-Host "[ERR] Missing init ckpt: $initPath" -ForegroundColor Red
    Write-Host "[i]  Run: go_phaseB_xy_passive.ps1 -Ramp1Epochs 3 -Ramp2Epochs 0" -ForegroundColor Yellow
    exit 1
}

Write-Host "[NEW] M3 align probe ($Epochs ep): union mask + transport_only ADR + species val" -ForegroundColor Cyan
Write-Host "[i]  SUPERVISION_MASK_TIMES=union | ADR=match_data_bio+exclude_wall | PASSIVE_ADR_WEIGHT=$AdrWeight" -ForegroundColor Cyan

if (-not $SkipAudit) {
    Write-Host "[NEW] GT mask audit (patient007, union vs last)" -ForegroundColor Cyan
    $env:BIOCHEM_DATA_BIO_MASK_MODE = "clot_band"
    $env:BIOCHEM_ADR_MASK_MODE = "match_data_bio"
    $env:BIOCHEM_ADR_EXCLUDE_WALL = "1"
    $env:BIOCHEM_SUPERVISION_MASK_TIMES = "last"
    Invoke-PythonRc scripts/audit_passive_adr_alignment.py --anchor patient007 --compare-mask-times | Out-Null
    $env:BIOCHEM_SUPERVISION_MASK_TIMES = "union"
    Invoke-PythonRc scripts/audit_passive_adr_alignment.py --anchor patient007 --compare-mask-times
}

Set-M3AlignProbeEnv -RunNote $RunNote -Epochs $Epochs -AdrWeight $AdrWeight
Copy-Item $initPath (Join-Path $RepoRoot "outputs\biochem\biochem_teacher_best_high_mu.pth") -Force

$rc = Invoke-PythonRc -m src.training.train_biochem_corrector --new --skip-pretrain --init-from-best `
    --epochs $Epochs --save-best --run-name $RunNote
if ($rc -ne 0) {
    Write-Host "[ERR] Training failed (exit $rc)" -ForegroundColor Red
    exit $rc
}

$gateRc = Invoke-PythonRc scripts/check_m3_align_gate.py --run-note $RunNote
if ($gateRc -eq 0) {
    Write-Host "[OK] M3 align gate passed" -ForegroundColor Green
} else {
    Write-Host "[WARN] M3 align gate did not pass (see check_m3_align_gate.py output)" -ForegroundColor Yellow
}
Write-Host "[i]  Summarize: python scripts/check_m3_align_gate.py --run-note $RunNote" -ForegroundColor Cyan
exit $gateRc
