# Fast signs-of-life for patient001 hop_ge2 hard-zero (~5-15 min).
# Mini-trains band vs global on one late tile; checks teacher vs closed-loop lumen fire.
# Optional single-anchor deploy (omit -SkipDeploy for even faster).
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_wc_v7_001_signs_of_life.ps1
#   powershell ... -SkipDeploy
#   powershell ... -Steps 60
#

param(
    [int] $Steps = 40,
    [switch] $SkipDeploy
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

$pyArgs = @(
    "scripts/diagnose_001_signs_of_life.py",
    "--steps", "$Steps",
    "--out", "outputs/biochem/offwall_model/wc_v7_crack_001_3h/signs_of_life.json"
)
if ($SkipDeploy) { $pyArgs += "--skip-deploy" }

Write-Host "[NEW] go_wc_v7_001_signs_of_life steps=$Steps skip_deploy=$SkipDeploy" -ForegroundColor Cyan
$null = Invoke-PythonRcCheck -Label "signs_of_life" -PyArgs $pyArgs
Write-Host "[OK] signs_of_life done" -ForegroundColor Green
Write-Host "[i] Look for verdict= signs_of_life_band | life_on_teacher_dead_on_closed_loop | dead_everywhere_*" -ForegroundColor DarkGray
