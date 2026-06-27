# Mat-growth-simple 3-leg ladder: random vs backbone warm-start vs geometry feats.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_mat_growth_ladder.ps1 -Fresh
#   powershell ... -Legs A_random,B_backbone -Fresh
#   powershell ... -EvalOnly
#   powershell ... -Fast -Fresh
#
# Legs:
#   A_random         - random init (simplest Mat-only single-head)
#   B_backbone       - SAGE conv warm-start from triangle6 species/best.pth
#   C_geom           - random init + SPECIES_GEOM_FEATS (width / expansion / curvature)
#   D_parity_single  - baseline-like dynamics, but Mat-only + single-head

param(
    [string[]] $Legs = @("A_random", "B_backbone", "C_geom", "D_parity_single"),
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

Write-Host "[NEW] mat_growth_ladder legs=$($Legs -join ',') epochs=$Epochs early_stop=$EarlyStop max_windows=$MaxWindows fresh=$Fresh evalOnly=$EvalOnly fast=$Fast" -ForegroundColor Cyan

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
    Invoke-PythonRcCheck -Label "mat_growth ladder summary" -PyArgs @(
        "scripts/summarize_mat_growth_ladder.py",
        "--run-root", "outputs/biochem/biochem_gnn/mat_growth_ladder",
        "--val-anchor", $ValAnchor
    )
}

Write-Host "[OK] mat_growth_ladder done -> $RunRoot" -ForegroundColor Green
