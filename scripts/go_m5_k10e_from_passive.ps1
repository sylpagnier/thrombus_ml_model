# K10e/K10f/K10g from passive teacher init (skip pretrain, GT_KINE_VEL=1).
#
#   powershell ... -InitCkpt outputs/biochem/biochem_teacher_passive_mu_unlock_best.pth -Variant wide -Epochs 18
#   ... -Variant bias -InitCkpt outputs/biochem/biochem_teacher_last.pth

param(
    [int] $Epochs = 18,
    [string] $RunNote = "m5_k10f_wide_from_passive",
    [string] $InitCkpt = "outputs/biochem/biochem_teacher_passive_mu_unlock_best.pth",
    [ValidateSet("wide", "narrow", "bias")]
    [string] $Variant = "wide",
    [switch] $SkipAudit
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")
. (Join-Path $PSScriptRoot "_passive_k10e_from_init_env.ps1")

$initPath = Join-Path $RepoRoot $InitCkpt
if (-not (Test-Path $initPath)) {
    Write-Host "[ERR] Missing init ckpt: $initPath" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path (Join-Path $RepoRoot "outputs\kinematics\kinematics_best.pth"))) {
    Write-Host "[ERR] Missing outputs/kinematics/kinematics_best.pth" -ForegroundColor Red
    exit 1
}

Write-Host "[NEW] M5 K10 ($Variant) from passive init ($Epochs ep): $RunNote" -ForegroundColor Cyan
Write-Host "[i]  Init=$InitCkpt | LOSS_ISOLATE=K10E | skip-pretrain | GT_KINE_VEL=1" -ForegroundColor Cyan

if (-not $SkipAudit) {
    $env:BIOCHEM_SUPERVISION_MASK_TIMES = "union"
    Invoke-PythonRc scripts/audit_passive_adr_alignment.py --anchor patient007 --compare-mask-times | Out-Null
}

Set-PassiveK10FromInitEnv -RunNote $RunNote -Epochs $Epochs -Variant $Variant
Copy-Item $initPath (Join-Path $RepoRoot "outputs\biochem\biochem_teacher_best_high_mu.pth") -Force

$rc = Invoke-PythonRc -m src.training.train_biochem_corrector --new --skip-pretrain --init-from-best `
    --epochs $Epochs --save-best --run-name $RunNote
exit $rc
