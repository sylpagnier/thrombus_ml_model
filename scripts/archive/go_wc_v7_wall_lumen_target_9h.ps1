# 9h autonomous: WC_v7 wall accuracy + lumen specialist (GT fire, no spray).
#
# Contract: SPECIES_CONTINUOUS_VEL_DECAY=1 + WALL_ONLY=1
# Driver: scripts/run_wall_lumen_target_9h.py (phases A probe -> B/C/D train pivots)
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_wc_v7_wall_lumen_target_9h.ps1 -Fresh
#

param(
    [double] $DeadlineHours = 9.0,
    [string] $RunRoot = "outputs/biochem/offwall_model/wc_v7_wall_lumen_target_9h",
    [switch] $Fresh,
    [switch] $SkipPhaseA
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"
$env:SPECIES_CONTINUOUS_VEL_DECAY = "1"
$env:SPECIES_CONTINUOUS_VEL_DECAY_WALL_ONLY = "1"

Write-Host "[NEW] go_wc_v7_wall_lumen_target_9h hours=$DeadlineHours out=$RunRoot" -ForegroundColor Cyan

$pyArgs = @(
    "scripts/run_wall_lumen_target_9h.py",
    "--deadline-hours", "$DeadlineHours",
    "--run-root", $RunRoot
)
if ($Fresh) { $pyArgs += "--fresh" }
if ($SkipPhaseA) { $pyArgs += "--skip-phase-a" }

Invoke-PythonRcCheck -Label "wall_lumen_target_9h" -PyArgs $pyArgs
Write-Host "[OK] 9h ladder finished -> $RunRoot" -ForegroundColor Green
