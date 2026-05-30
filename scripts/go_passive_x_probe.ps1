# Fast I.1 X probe session. Default -Turbo targets <30 min (3 legs, 2ep, fast val).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_passive_x_probe.ps1"
#   ... -Turbo:$false -Epochs 3   # full 5-leg matrix (~60-90 min)

param(
    [string] $InitCkpt = "outputs/biochem/biochem_teacher_passive_align_locked.pth",
    [int] $Epochs = 2,
    [string] $LegsOnly = "",
    [switch] $Turbo = $true
)

$ErrorActionPreference = "Stop"
$iterateArgs = @{
    InitCkpt   = $InitCkpt
    Epochs     = $Epochs
    SkipPytest = $true
}
if ($LegsOnly) { $iterateArgs.LegsOnly = $LegsOnly }
if ($Turbo) { $iterateArgs.Turbo = $true }
& (Join-Path $PSScriptRoot "go_passive_x_iterate.ps1") @iterateArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$summRc = Invoke-PythonRc scripts/summarize_passive_x_block.py
Write-Host "[i]  Probe matrix done. Promote only if logs separate recipes: go_passive_x_block_finish.ps1 -Promote" -ForegroundColor Cyan
exit $summRc
