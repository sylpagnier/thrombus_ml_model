# K11 launcher — delegates to K11e (localized best-practice bundle).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_k11_clot_gate.ps1" -Fresh

param(
    [switch] $Fresh,
    [switch] $OomSafe = $true
)

$here = $PSScriptRoot
& (Join-Path $here "go_k11e_clot_gate.ps1") @PSBoundParameters
exit $LASTEXITCODE
