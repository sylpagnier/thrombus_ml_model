# Bipartite Graph Firewall Sweep Script
# Runs training and evaluation for the 5 new firewall solutions:
#   WC_v5_skiphop
#   WC_v5_blind_loss
#   WC_v5_phys_gating
#   WC_v5_closed_loop
#   WC_v5_two_model
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_firewall_sweep.ps1 -Fast -Fresh

param(
    [string[]] $Legs = @("WC_v5_skiphop", "WC_v5_blind_loss", "WC_v5_phys_gating", "WC_v5_closed_loop", "WC_v5_two_model"),
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

Write-Host "[NEW] go_firewall_sweep legs=$($Legs -join ',') epochs=$Epochs early_stop=$EarlyStop max_windows=$MaxWindows fresh=$Fresh evalOnly=$EvalOnly fast=$Fast" -ForegroundColor Cyan

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
    Invoke-PythonRcCheck -Label "firewall sweep summary" -PyArgs @(
        "scripts/summarize_mat_growth_ladder.py",
        "--run-root", "outputs/biochem/biochem_gnn/mat_growth_ladder",
        "--val-anchor", $ValAnchor
    )
}

Write-Host "[OK] firewall sweep done -> outputs are under $RunRoot" -ForegroundColor Green
