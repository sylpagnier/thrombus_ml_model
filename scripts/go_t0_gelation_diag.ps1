# Build COMSOL debug sidecar + run T0 gelation factorization diagnostics.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_t0_gelation_diag.ps1"

param(
    [string] $Anchor = "patient007",
    [string] $Out = "outputs/biochem/clot_trigger/t0_gelation_comsol_diag.json"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")

$env:PYTHONUNBUFFERED = "1"

$dbgTxt = Join-Path $RepoRoot "data/processed/cfd_results_biochem/${Anchor}_debugging.txt"
if (Test-Path $dbgTxt) {
    Write-Host "[NEW] Build COMSOL debug sidecar ($Anchor)" -ForegroundColor Cyan
    Invoke-PythonRcCheck -Label "build debug sidecar" -PyArgs @(
        "scripts/build_comsol_debug_sidecar.py",
        "--anchor", $Anchor
    )
} else {
    Write-Host "[WARN] Missing $dbgTxt" -ForegroundColor Yellow
}

Write-Host "[NEW] T0 gelation COMSOL diagnostics ($Anchor)" -ForegroundColor Cyan
Invoke-PythonRcCheck -Label "t0 gelation diag" -PyArgs @(
    "scripts/diagnose_t0_gelation_comsol.py",
    "--anchor", $Anchor,
    "--out", $Out
)

Write-Host "[OK] -> $Out" -ForegroundColor Green
