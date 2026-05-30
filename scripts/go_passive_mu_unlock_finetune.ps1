# Finetune after mu-unlock probe: wall + high-mu MU_LOG weights (fix bulk-only plateau).
# Prereq: go_passive_mu_unlock_probe.ps1 then go_passive_lock_mu_unlock_best.ps1
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_passive_mu_unlock_finetune.ps1"
#   powershell ... -InitCkpt outputs/biochem/biochem_teacher_passive_mu_unlock_best.pth -Epochs 8 -SkipAudit

param(
    [int] $Epochs = 8,
    [string] $RunNote = "passive_mu_unlock_finetune",
    [string] $InitCkpt = "outputs/biochem/biochem_teacher_passive_mu_unlock_best.pth",
    [string] $MuLogWeight = "0.5",
    [string] $MuLogWallWeight = "0.75",
    [string] $MuLogHighWeight = "1.5",
    [switch] $SkipAudit,
    [switch] $UseLastFallback
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")
. (Join-Path $PSScriptRoot "_passive_mu_unlock_finetune_env.ps1")

$initPath = Join-Path $RepoRoot $InitCkpt
if (-not (Test-Path $initPath)) {
    Write-Host "[i]  Promoting mu-unlock best ckpt..." -ForegroundColor Cyan
    $lockArgs = @()
    if ($UseLastFallback) { $lockArgs += "-UseLastFallback" }
    & (Join-Path $PSScriptRoot "go_passive_lock_mu_unlock_best.ps1") @lockArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $initPath = Join-Path $RepoRoot $InitCkpt
}
if (-not (Test-Path $initPath)) {
    Write-Host "[ERR] Missing init ckpt: $initPath" -ForegroundColor Red
    exit 1
}

Write-Host "[NEW] Passive mu-unlock finetune ($Epochs ep): W_log=$MuLogWeight W_wall=$MuLogWallWeight W_high=$MuLogHighWeight" -ForegroundColor Cyan
Write-Host "[i]  Init=$InitCkpt | bio frozen | species val+train lines on" -ForegroundColor Cyan

if (-not $SkipAudit) {
    $env:BIOCHEM_SUPERVISION_MASK_TIMES = "union"
    Invoke-PythonRc scripts/audit_passive_adr_alignment.py --anchor patient007 --compare-mask-times | Out-Null
}

Set-PassiveMuUnlockFinetuneEnv -RunNote $RunNote -Epochs $Epochs `
    -MuLogWeight $MuLogWeight -MuLogWallWeight $MuLogWallWeight -MuLogHighWeight $MuLogHighWeight

Copy-Item $initPath (Join-Path $RepoRoot "outputs\biochem\biochem_teacher_best_high_mu.pth") -Force

$rc = Invoke-PythonRc -m src.training.train_biochem_corrector --new --skip-pretrain --init-from-best `
    --epochs $Epochs --save-best --run-name $RunNote
if ($rc -ne 0) {
    Write-Host "[ERR] Training failed (exit $rc)" -ForegroundColor Red
    exit $rc
}

Write-Host "[NEW] Species check (post-train)" -ForegroundColor Cyan
Invoke-PythonRc scripts/eval_passive_species_anchors.py --checkpoint outputs/biochem/biochem_teacher_last.pth

Write-Host "[NEW] Finetune gate" -ForegroundColor Cyan
$gateRc = Invoke-PythonRc scripts/check_passive_mu_unlock_finetune_gate.py --run-note $RunNote
if ($gateRc -ne 0) {
    Write-Host "[WARN] Finetune gate failed (see run.jsonl)" -ForegroundColor Yellow
}
exit $gateRc
