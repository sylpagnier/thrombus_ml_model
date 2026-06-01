# Mu-unlock probe: MU_LOG-only backward on species-trained passive teacher (not step 3).
# Uses _passive_mu_unlock_env.ps1 (no passive_transport preset; TRAIN_MU=1, bio frozen).
#
# Prereq: passive_align_20ep or locked ckpt (species ~0.03 FI). Do NOT init from a failed probe last.pth.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_passive_mu_unlock_probe.ps1"
#   powershell ... -InitCkpt outputs/biochem/biochem_teacher_passive_align_locked.pth -Epochs 12 -SkipAudit

param(
    [int] $Epochs = 12,
    [string] $RunNote = "passive_mu_unlock_probe",
    [string] $InitCkpt = "outputs/biochem/biochem_teacher_passive_align_locked.pth",
    [string] $MuRatioMax = "20",
    [switch] $SkipAudit
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")
. (Join-Path $PSScriptRoot "_passive_mu_unlock_env.ps1")

$initPath = Join-Path $RepoRoot $InitCkpt
if (-not (Test-Path $initPath)) {
    Write-Host "[ERR] Missing init ckpt: $initPath" -ForegroundColor Red
    Write-Host "[i]  Run: go_passive_lock_align_ckpt.ps1 then go_passive_align_20ep.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host "[NEW] Passive mu-unlock probe ($Epochs ep): LOSS_ISOLATE=MU_LOG, mu_ratio_max=$MuRatioMax, bio frozen" -ForegroundColor Cyan
Write-Host "[i]  Init=$InitCkpt | no passive_transport preset | species val+train lines on" -ForegroundColor Cyan

if (-not $SkipAudit) {
    $env:BIOCHEM_SUPERVISION_MASK_TIMES = "union"
    Invoke-PythonRc scripts/audit_passive_adr_alignment.py --anchor patient007 --compare-mask-times | Out-Null
}

Set-PassiveMuUnlockEnv -RunNote $RunNote -Epochs $Epochs -MuRatioMax $MuRatioMax

Copy-Item $initPath (Join-Path $RepoRoot "outputs\biochem\biochem_teacher_best_high_mu.pth") -Force

$rc = Invoke-PythonRc -m src.training.train_biochem_corrector --new --skip-pretrain --init-from-best `
    --epochs $Epochs --save-best --run-name $RunNote
if ($rc -ne 0) {
    Write-Host "[ERR] Training failed (exit $rc)" -ForegroundColor Red
    exit $rc
}

Write-Host "[NEW] Lock mu-unlock all-truth-best (for finetune init)" -ForegroundColor Cyan
$unlockBest = Join-Path $RepoRoot "outputs\biochem\biochem_teacher_passive_mu_unlock_best.pth"
if (Test-Path $unlockBest) {
    & (Join-Path $PSScriptRoot "go_passive_lock_mu_unlock_best.ps1")
} else {
    Write-Host "[WARN] $unlockBest not written (re-run probe on updated trainer); finetune may need UseLastFallback on go_passive_mu_unlock_finetune.ps1" -ForegroundColor Yellow
}

Write-Host "[NEW] Species check (post-train)" -ForegroundColor Cyan
Invoke-PythonRc scripts/eval_passive_species_anchors.py --checkpoint outputs/biochem/biochem_teacher_last.pth

Write-Host "[NEW] Mu-unlock gate" -ForegroundColor Cyan
$gateRc = Invoke-PythonRc scripts/check_passive_mu_unlock_gate.py --run-note $RunNote
if ($gateRc -ne 0) {
    Write-Host "[WARN] Mu-unlock gate failed (see run.jsonl)" -ForegroundColor Yellow
}

Write-Host "[i]  Success: val mu_log_mae drops below ~1.2; species FI stays ~0.03-0.05" -ForegroundColor Cyan
Write-Host "[i]  Next: go_passive_mu_unlock_finetune.ps1 (wall+high-mu weights)" -ForegroundColor Cyan
exit $gateRc
