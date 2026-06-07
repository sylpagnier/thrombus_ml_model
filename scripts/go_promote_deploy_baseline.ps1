# Promote deploy Leg B as canonical clot_baseline manifest (no GT in mu commit).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_promote_deploy_baseline.ps1
#   powershell ... -ScorecardJson outputs\biochem\mlp_clot_inject_probe\b_deploy_baseline_fast.json

param(
    [string] $ScorecardJson = "outputs\biochem\mlp_clot_inject_probe\b_deploy_baseline_fast.json"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")

$rc = Invoke-PythonRc (Join-Path $RepoRoot "scripts\promote_clot_baseline_deploy.py") `
    "--scorecard-json", $ScorecardJson
if ($rc -ne 0) { exit $rc }

Write-Host "[OK]  outputs\biochem\clot_baseline\manifest.json (lane_b_deploy)" -ForegroundColor Green
Write-Host "[i]  smoke: powershell ... -File .\scripts\go_mlp_b_deploy_probe.ps1 -Fast -Leg B_deploy" -ForegroundColor DarkGray
