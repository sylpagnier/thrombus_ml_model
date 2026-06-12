# T0 deploy mask audit (no GT forward leakage + nucleation sanity).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_trigger_t0_audit.ps1"

param(
    [string] $Out = "outputs/biochem/clot_trigger/t0_deploy_audit.json"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")

$env:PYTHONUNBUFFERED = "1"

Write-Host "[NEW] T0 deploy mask audit (pred-seed nucleation, no GT commits)" -ForegroundColor Cyan
Invoke-PythonRcCheck -Label "t0 deploy audit" -PyArgs @(
    "scripts/audit_clot_trigger_deploy_masks.py",
    "--out", $Out
)
