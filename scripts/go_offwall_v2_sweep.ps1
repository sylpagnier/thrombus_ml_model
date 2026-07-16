# Off-wall clot growth sweep script v2.
# Runs training and evaluation for the new off-wall pivot v2 sweep:
#   WC_v2_baseline
#   WC_v2_convection
#   WC_v2_longrange
#   WC_v2_label_smooth
#   WC_v2_dilation
#   WC_v2_longrange_smooth
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_offwall_v2_sweep.ps1 -Fresh
#   powershell ... -Fast -Fresh
#

param(
    [string[]] $Legs = @("WC_v2_baseline", "WC_v2_convection", "WC_v2_longrange", "WC_v2_label_smooth", "WC_v2_dilation", "WC_v2_longrange_smooth"),
    [int] $Epochs = 25,
    [int] $EarlyStop = 15,
    [int] $MaxWindows = 32,
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

# Fast preset for quick smoke test (under 5 minutes total for all legs)
if ($Fast) {
    $Epochs = 1
    $EarlyStop = 1
    $MaxWindows = 4
}

# Split legs if passed as a single comma-separated string
$legList = @()
foreach ($l in $Legs) {
    if ($l.Contains(",")) {
        $legList += $l.Split(",") | ForEach-Object { $_.Trim() }
    } else {
        $legList += $l.Trim()
    }
}
$Legs = @($legList | Where-Object { $_ })

$RunRoot = Join-Path $RepoRoot "outputs/biochem/biochem_gnn/mat_growth_ladder"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

Write-Host "[NEW] go_offwall_v2_sweep legs=$($Legs -join ',') epochs=$Epochs early_stop=$EarlyStop max_windows=$MaxWindows fresh=$Fresh evalOnly=$EvalOnly fast=$Fast" -ForegroundColor Cyan

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
    # Do NOT pass -Fast to the child script so it uses our custom minimal parameter values directly
    # if ($Fast) { $legArgs += "-Fast" }
    if ($Fresh) { $legArgs += "-Fresh" }
    if ($EvalOnly) { $legArgs += "-EvalOnly" }
    & powershell @legArgs
    if ($LASTEXITCODE -ne 0) {
        throw "leg $leg failed (exit=$LASTEXITCODE)"
    }
}

if (-not $SkipSummary) {
    Invoke-PythonRcCheck -Label "off-wall sweep v2 ladder summary" -PyArgs @(
        "scripts/summarize_mat_growth_ladder.py",
        "--run-root", "outputs/biochem/biochem_gnn/mat_growth_ladder",
        "--val-anchor", $ValAnchor
    )
}

Write-Host "[OK] off-wall v2 sweep done -> outputs are under $RunRoot" -ForegroundColor Green
