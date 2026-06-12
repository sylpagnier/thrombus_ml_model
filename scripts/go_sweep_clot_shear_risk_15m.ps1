# Shear-gradient risk sweep (~16 rules x 6 anchors) + deploy-winner viz.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_sweep_clot_shear_risk_15m.ps1"
#   powershell ... -Resume -SkipSweep

param(
    [string] $Anchor = "patient007",
    [string] $AnchorDir = "data/processed/graphs_biochem_anchors",
    [switch] $Resume,
    [switch] $SkipSweep
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")

$OutJson = "outputs/biochem/diagnostics/clot_rule_shear_risk_sweep.json"

Write-Host ""
Write-Host "[NEW] shear-gradient risk sweep (~15m) + deploy-winner viz ($Anchor)" -ForegroundColor Cyan

if (-not $SkipSweep) {
    $sweepArgs = @(
        "scripts/sweep_clot_rule_architectures.py",
        "--shear-risk",
        "--anchor-dir", $AnchorDir
    )
    if ($Resume) { $sweepArgs += "--resume" }
    Invoke-PythonRcCheck -Label "shear risk sweep" -PyArgs $sweepArgs
}

$vizArgs = @(
    "scripts/viz_clot_sweep_dual_winners.py",
    "--json", $OutJson,
    "--anchor", $Anchor,
    "--anchor-dir", $AnchorDir
)
Invoke-PythonRcCheck -Label "shear deploy viz" -PyArgs $vizArgs

Write-Host ""
Write-Host "[OK] outputs:" -ForegroundColor Green
Write-Host "  JSON: $OutJson"
Write-Host "  PNG deploy: outputs/biochem/viz/clot_deploy/temporal_rule_${Anchor}_balanced_best.png"
