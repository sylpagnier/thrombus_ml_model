# 20ep passive-align confirm: same recipe as go_m3_align_probe.ps1 + species on train anchors.
# Prereq: go_passive_lock_align_ckpt.ps1 (or align probe last.pth).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_passive_align_20ep.ps1"
#   powershell ... -InitCkpt outputs/biochem/biochem_teacher_passive_align_locked.pth -SkipAudit

param(
    [int] $Epochs = 20,
    [string] $RunNote = "passive_align_20ep",
    [string] $InitCkpt = "outputs/biochem/biochem_teacher_passive_align_locked.pth",
    [string] $AdrWeight = "1e-4",
    [switch] $SkipAudit
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")
. (Join-Path $PSScriptRoot "_passive_align_recipe_env.ps1")

$initPath = Join-Path $RepoRoot $InitCkpt
if (-not (Test-Path $initPath)) {
    Write-Host "[ERR] Missing init ckpt: $initPath" -ForegroundColor Red
    Write-Host "[i]  Run: go_passive_lock_align_ckpt.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host "[NEW] Passive align 20ep ($Epochs ep): union mask + transport_only ADR + train+val species" -ForegroundColor Cyan

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

Set-PassiveAlignRecipeEnv -RunNote $RunNote -Epochs $Epochs -AdrWeight $AdrWeight -SpeciesTrainEval
Copy-Item $initPath (Join-Path $RepoRoot "outputs\biochem\biochem_teacher_best_high_mu.pth") -Force

$rc = Invoke-PythonRc -m src.training.train_biochem_corrector --new --skip-pretrain --init-from-best `
    --epochs $Epochs --save-best --run-name $RunNote
if ($rc -ne 0) {
    Write-Host "[ERR] Training failed (exit $rc)" -ForegroundColor Red
    exit $rc
}

Write-Host "[NEW] Post-hoc species table (all anchors, trained last.pth)" -ForegroundColor Cyan
Invoke-PythonRc scripts/eval_passive_species_anchors.py --checkpoint outputs/biochem/biochem_teacher_last.pth

$gateRc = Invoke-PythonRc scripts/check_m3_align_gate.py --run-note $RunNote
if ($gateRc -eq 0) {
    Write-Host "[OK] Passive align gate passed ($RunNote)" -ForegroundColor Green
} else {
    Write-Host "[WARN] Gate did not pass (see check_m3_align_gate.py)" -ForegroundColor Yellow
}
exit $gateRc
