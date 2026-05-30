# Run I.1 X block to completion: probe matrix -> promote (dump) -> gate.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_passive_x_block_pass.ps1"
#   ... -SkipProbe -SkipTrain   # promote dump from locked align only

param(
    [string] $InitCkpt = "outputs/biochem/biochem_teacher_passive_align_locked.pth",
    [int] $Epochs = 2,
    [int] $ConfirmEpochs = 4,
    [switch] $Turbo = $true,
    [switch] $SkipProbe,
    [switch] $SkipTrain
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (-not (Test-Path (Join-Path $RepoRoot $InitCkpt))) {
    Write-Host "[ERR] Missing init: $InitCkpt" -ForegroundColor Red
    exit 1
}

Write-Host "[NEW] I.1 X block pass (probe -> promote -> gate)" -ForegroundColor Cyan

if (-not $SkipProbe) {
    & (Join-Path $PSScriptRoot "go_passive_x_probe.ps1") -InitCkpt $InitCkpt -Epochs $Epochs
}

$promoteArgs = @{
    Promote       = $true
    InitCkpt      = $InitCkpt
    ConfirmEpochs = $ConfirmEpochs
}
if ($SkipTrain) { $promoteArgs.SkipTrain = $true }
& (Join-Path $PSScriptRoot "go_passive_x_block_finish.ps1") @promoteArgs
$promoteRc = $LASTEXITCODE
$finalRc = Invoke-PythonRc scripts/check_passive_x_block_pass.py --require-promote
if ($finalRc -ne 0) { exit $finalRc }
exit $promoteRc
