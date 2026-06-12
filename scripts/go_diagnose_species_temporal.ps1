# GT species temporal / spatial pattern diagnostic (all 6 anchors).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_diagnose_species_temporal.ps1"

param(
    [string] $Anchors = "",
    [int] $MaxTimes = 14,
    [string] $Out = "outputs/biochem/diagnostics/species_temporal_patterns.json"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")

$pyArgs = @(
    "scripts/diagnose_species_temporal_patterns.py",
    "--max-times", "$MaxTimes",
    "--out", $Out
)
if ($Anchors) { $pyArgs += @("--anchors", $Anchors) }

Invoke-PythonRcCheck -Label "species temporal diagnostic" -PyArgs $pyArgs
