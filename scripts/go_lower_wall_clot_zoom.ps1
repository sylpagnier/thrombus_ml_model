# Zoomed lower-wall GT vs temporal rule (patient007 default).
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_lower_wall_clot_zoom.ps1" -TOut 37

param(
    [string] $Anchor = "patient007",
    [int] $TOut = 37
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_clot_prior_rule_winner_env.ps1")
. (Join-Path $PSScriptRoot "_clot_temporal_rule_winner_env.ps1")
. (Join-Path $PSScriptRoot "_python_rc.ps1")

Invoke-PythonRcCheck -Label "lower wall zoom viz" -PyArgs @(
    "scripts/viz_lower_wall_clot_zoom.py",
    "--anchor", $Anchor,
    "--t-out", $TOut
)
