# Compare deploy-faithful biochem_gnn vs legacy GT-leak eval (locked baseline).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_biochem_gnn_deploy_ab.ps1
#   powershell ... -Inc40

param(
    [string] $Manifest = "",
    [string] $Modes = "legacy_oracle,deploy_frozen,deploy_coupled",
    [switch] $Inc40
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

$pyArgs = @(
    "scripts/eval_biochem_gnn_deploy_ab.py",
    "--modes", $Modes,
    "--times", "27,53"
)
if ($Manifest.Trim()) { $pyArgs += @("--manifest", $Manifest) }
if ($Inc40) { $pyArgs += "--inc40" }

Write-Host "[NEW] biochem_gnn deploy A/B eval" -ForegroundColor Cyan
Invoke-PythonRcCheck -Label "biochem_gnn deploy ab" -PyArgs $pyArgs
