# Fast S0 rule architecture compare (~1 min CPU, no GPU).
param(
    [int]$Top = 12
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
. "$PSScriptRoot\_python_rc.ps1"
Invoke-PythonRcCheck "scripts/explore_clot_s0_architectures.py" "--top" "$Top"
