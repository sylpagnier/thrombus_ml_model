# Step-2 bridge: data-only backward + modest mu aux + low-weight masked ADR (stay at COMPLEXITY_STEP=2).
# Prereq: locked or 20ep passive teacher.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_passive_step2_bridge.ps1"
#   powershell ... -Epochs 12 -InitCkpt outputs/biochem/biochem_teacher_passive_align_locked.pth

param(
    [int] $Epochs = 12,
    [string] $RunNote = "passive_step2_bridge",
    [string] $InitCkpt = "outputs/biochem/biochem_teacher_passive_align_locked.pth",
    [string] $AdrWeight = "1e-4",
    [string] $MuLogWeight = "0.75",
    [string] $MuSiWeight = "0.15",
    [switch] $SkipAudit
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")
. (Join-Path $PSScriptRoot "_passive_step2_bridge_env.ps1")

# M3 union mask for bridge (matches align / explore X baseline)
$env:BIOCHEM_SUPERVISION_MASK_TIMES = "union"
$env:BIOCHEM_ADR_MASK_MODE = "match_data_bio"
$env:BIOCHEM_ADR_EXCLUDE_WALL = "1"
$env:BIOCHEM_ADR_RESIDUAL_MODE = "transport_only"

$initPath = Join-Path $RepoRoot $InitCkpt
if (-not (Test-Path $initPath)) {
    Write-Host "[ERR] Missing init ckpt: $initPath" -ForegroundColor Red
    Write-Host "[i]  Run: go_passive_lock_align_ckpt.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host "[NEW] Passive step-2 bridge ($Epochs ep): LOSS_DATA_ONLY=1 COMPLEXITY_STEP=2" -ForegroundColor Cyan
Write-Host "[i]  W_MuLog=$MuLogWeight W_MuSI=$MuSiWeight | ADR backprop weight=$AdrWeight | GT_KINE_VEL=1" -ForegroundColor Cyan

if (-not $SkipAudit) {
    $env:BIOCHEM_SUPERVISION_MASK_TIMES = "union"
    Invoke-PythonRc scripts/audit_passive_adr_alignment.py --anchor patient007 --compare-mask-times
}

Set-PassiveStep2BridgeEnv -RunNote $RunNote -Epochs $Epochs -AdrWeight $AdrWeight `
    -MuLogWeight $MuLogWeight -MuSiWeight $MuSiWeight
Copy-Item $initPath (Join-Path $RepoRoot "outputs\biochem\biochem_teacher_best_high_mu.pth") -Force

$rc = Invoke-PythonRc -m src.training.train_biochem_corrector --new --skip-pretrain --init-from-best `
    --epochs $Epochs --save-best --run-name $RunNote
if ($rc -ne 0) {
    Write-Host "[ERR] Training failed (exit $rc)" -ForegroundColor Red
    exit $rc
}

$gateRc = Invoke-PythonRc scripts/check_passive_step2_bridge_gate.py --run-note $RunNote
if ($gateRc -eq 0) {
    Write-Host "[OK] Step-2 bridge gate passed" -ForegroundColor Green
} else {
    Write-Host "[WARN] Step-2 bridge gate did not pass" -ForegroundColor Yellow
}
exit $gateRc
