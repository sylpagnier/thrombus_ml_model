# I.3 XY block: step-2 bridge chunks (hold from M3 + learn from align + optional mu-unlock chain).
#
# Prereq: outputs/biochem/biochem_teacher_passive_m3_locked.pth (or biochem_teacher_last.pth after M3 12ep).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_passive_xy_block_pass.ps1" -Probe
#   powershell ... -SkipHold                    # learn leg only (align_locked)
#   powershell ... -WithMuUnlock                # mu-unlock 6ep then bridge 6ep

param(
    [string] $InitCkptHold = "outputs/biochem/biochem_teacher_passive_m3_locked.pth",
    [string] $InitCkptLearn = "outputs/biochem/biochem_teacher_passive_align_locked.pth",
    [string] $InitCkptUnlock = "outputs/biochem/biochem_teacher_passive_align_locked.pth",
    [int] $HoldEpochs = 3,
    [int] $LearnEpochs = 6,
    [int] $UnlockEpochs = 6,
    [switch] $Probe,
    [switch] $WithMuUnlock,
    [switch] $SkipHold,
    [switch] $SkipLearn,
    [switch] $SkipAudit
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")

if ($Probe) {
    $HoldEpochs = 3
    $LearnEpochs = 3
    $UnlockEpochs = 3
    $WithMuUnlock = $false
}

$bridgeCommon = @{
    SkipAudit       = $SkipAudit
    GradScaleOnCap  = $true
}

function Test-Ckpt {
    param([string] $Rel)
    $p = Join-Path $RepoRoot $Rel
    if (-not (Test-Path $p)) {
        Write-Host "[ERR] Missing ckpt: $p" -ForegroundColor Red
        exit 1
    }
}

Write-Host "[NEW] I.3 XY block pass (Probe=$Probe WithMuUnlock=$WithMuUnlock)" -ForegroundColor Cyan

if (-not $SkipHold) {
    Test-Ckpt $InitCkptHold
    Write-Host "[NEW] XY2-hold: bridge from M3 locked ($HoldEpochs ep, viability / species hold)" -ForegroundColor Cyan
    & (Join-Path $PSScriptRoot "go_passive_step2_bridge.ps1") @bridgeCommon `
        -Epochs $HoldEpochs `
        -InitCkpt $InitCkptHold `
        -RunNote "passive_step2_bridge_m3_hold"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $vRc = Invoke-PythonRc scripts/check_passive_xy_viability_pass.py `
        --run-note passive_step2_bridge_m3_hold --min-epochs $HoldEpochs --saturated
    if ($vRc -ne 0 -and -not $Probe) { exit $vRc }
}

if (-not $SkipLearn) {
    Test-Ckpt $InitCkptLearn
    Write-Host "[NEW] XY2-learn: bridge from align ($LearnEpochs ep, expect L_bio/ADR descent)" -ForegroundColor Cyan
    & (Join-Path $PSScriptRoot "go_passive_step2_bridge.ps1") @bridgeCommon `
        -Epochs $LearnEpochs `
        -InitCkpt $InitCkptLearn `
        -RunNote "passive_step2_bridge_align_learn"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $gRc = Invoke-PythonRc scripts/check_passive_step2_bridge_gate.py `
        --run-note passive_step2_bridge_align_learn --min-epochs $LearnEpochs
    if ($gRc -ne 0 -and -not $Probe) {
        Write-Host "[WARN] align learn bridge gate failed (ok for Probe trends)" -ForegroundColor Yellow
    }
}

if ($WithMuUnlock) {
    Test-Ckpt $InitCkptUnlock
    Write-Host "[NEW] XY3: mu-unlock probe ($UnlockEpochs ep)" -ForegroundColor Cyan
    & (Join-Path $PSScriptRoot "go_passive_mu_unlock_probe.ps1") `
        -Epochs $UnlockEpochs -InitCkpt $InitCkptUnlock -SkipAudit
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $unlockLast = Join-Path $RepoRoot "outputs\biochem\biochem_teacher_last.pth"
    if (-not (Test-Path $unlockLast)) {
        Write-Host "[ERR] Missing $unlockLast after mu-unlock" -ForegroundColor Red
        exit 1
    }
    Write-Host "[NEW] XY3: bridge from unlock last ($LearnEpochs ep)" -ForegroundColor Cyan
    & (Join-Path $PSScriptRoot "go_passive_step2_bridge.ps1") @bridgeCommon `
        -Epochs $LearnEpochs `
        -InitCkpt "outputs/biochem/biochem_teacher_last.pth" `
        -RunNote "passive_step2_bridge_unlock_learn"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "[OK] I.3 XY block pass complete" -ForegroundColor Green
Write-Host "[i]  Hold viability: python scripts/check_passive_xy_viability_pass.py --run-note passive_step2_bridge_m3_hold --saturated" -ForegroundColor Cyan
Write-Host "[i]  Learn gate: python scripts/check_passive_step2_bridge_gate.py --run-note passive_step2_bridge_align_learn" -ForegroundColor Cyan
exit 0
