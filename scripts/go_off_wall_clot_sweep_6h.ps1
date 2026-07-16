# Off-wall clot growth sweep script.
# Runs training and evaluation for the new off-wall pivots:
#   WC_pivot1_skiphop
#   WC_pivot2_sheargate
#   WC_pivot3_occlusion
#   WC_pivot4_frontier
#   WC_pivots_combined
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_off_wall_clot_sweep_6h.ps1 -Fresh
#   powershell ... -Fast -Fresh
#

param(
    [string[]] $Legs = @("WC_pivot1_skiphop", "WC_pivot2_sheargate", "WC_pivot3_occlusion", "WC_pivot4_frontier", "WC_pivots_combined"),
    [int] $Epochs = 50,
    [int] $EarlyStop = 35,
    [int] $MaxWindows = 0,
    [string] $ValAnchor = "patient007",
    [switch] $Fast,
    [switch] $Fresh,
    [switch] $EvalOnly,
    [switch] $SkipSummary
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

# Fixed apples-to-apples fast preset.
$FAST_EPOCHS = 10
$FAST_EARLYSTOP = 6
$FAST_MAX_WINDOWS = 16
if ($Fast) {
    $Epochs = $FAST_EPOCHS
    $EarlyStop = $FAST_EARLYSTOP
    $MaxWindows = $FAST_MAX_WINDOWS
}

$RunRoot = Join-Path $RepoRoot "outputs/biochem/biochem_gnn/mat_growth_ladder"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

Write-Host "[NEW] go_off_wall_clot_sweep_6h legs=$($Legs -join ',') epochs=$Epochs early_stop=$EarlyStop max_windows=$MaxWindows fresh=$Fresh evalOnly=$EvalOnly fast=$Fast" -ForegroundColor Cyan

foreach ($leg in $Legs) {
    $legArgs = @(
        "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", (Join-Path $PSScriptRoot "go_mat_growth_simple.ps1"),
        "-Leg", $leg,
        "-Epochs", "$Epochs",
        "-EarlyStop", "$EarlyStop",
        "-MaxWindows", "$MaxWindows",
        "-ValAnchor", $ValAnchor
    )
    if ($Fast) { $legArgs += "-Fast" }
    if ($Fresh) { $legArgs += "-Fresh" }
    if ($EvalOnly) { $legArgs += "-EvalOnly" }
    & powershell @legArgs
    if ($LASTEXITCODE -ne 0) {
        throw "leg $leg failed (exit=$LASTEXITCODE)"
    }
}

if (-not $SkipSummary) {
    Invoke-PythonRcCheck -Label "off-wall sweep ladder summary" -PyArgs @(
        "scripts/summarize_mat_growth_ladder.py",
        "--run-root", "outputs/biochem/biochem_gnn/mat_growth_ladder",
        "--val-anchor", $ValAnchor
    )
}

Write-Host "[OK] off-wall sweep done -> outputs are under $RunRoot" -ForegroundColor Green
