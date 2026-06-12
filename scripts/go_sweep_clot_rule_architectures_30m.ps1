# ~30m curated rule sweep (hop-graded clot_shape, p007-priority composite) + promote + viz.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_sweep_clot_rule_architectures_30m.ps1"

param(
    [string] $Anchor = "patient007",
    [int] $Keyframes = 8
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_clot_prior_rule_winner_env.ps1")
. (Join-Path $PSScriptRoot "_python_rc.ps1")

Write-Host ""
Write-Host "[NEW] curated rule sweep (30m budget, hop-graded clot_shape, p007 priority)" -ForegroundColor Cyan

Invoke-PythonRcCheck -Label "curated rule sweep" -PyArgs @(
    "scripts/sweep_clot_rule_architectures.py",
    "--curated"
)

Invoke-PythonRcCheck -Label "promote curated winner" -PyArgs @(
    "scripts/promote_clot_architecture_winner.py",
    "--json", "outputs/biochem/diagnostics/clot_rule_curated_sweep.json"
)

$archEnv = Join-Path $PSScriptRoot "_clot_architecture_winner_env.ps1"
if (Test-Path $archEnv) { . $archEnv }

& (Join-Path $PSScriptRoot "go_clot_temporal_rule_timeline_viz.ps1") -Anchor $Anchor -Keyframes $Keyframes

Write-Host "[OK] curated sweep + viz done" -ForegroundColor Green
