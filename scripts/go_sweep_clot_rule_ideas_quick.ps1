# Offset-ramp + threshold-accumulation sweep + dual-winner timeline PNGs.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_sweep_clot_rule_ideas_quick.ps1"
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

$OutJson = "outputs/biochem/diagnostics/clot_rule_ideas_sweep.json"

Write-Host ""
Write-Host "[NEW] clot rule ideas sweep (offset ramp + threshold accum) + dual-winner viz ($Anchor)" -ForegroundColor Cyan

if (-not $SkipSweep) {
    $sweepArgs = @(
        "scripts/sweep_clot_rule_architectures.py",
        "--ideas",
        "--anchor-dir", $AnchorDir
    )
    if ($Resume) { $sweepArgs += "--resume" }
    Invoke-PythonRcCheck -Label "ideas sweep" -PyArgs $sweepArgs
}

$vizArgs = @(
    "scripts/viz_clot_sweep_dual_winners.py",
    "--json", $OutJson,
    "--anchor", $Anchor,
    "--anchor-dir", $AnchorDir
)
Invoke-PythonRcCheck -Label "dual-winner viz" -PyArgs $vizArgs

Write-Host ""
Write-Host "[OK] outputs:" -ForegroundColor Green
Write-Host "  JSON: $OutJson"
Write-Host "  PNG shape:    outputs/biochem/viz/clot_deploy/temporal_rule_${Anchor}_shape_best.png"
Write-Host "  PNG balanced: outputs/biochem/viz/clot_deploy/temporal_rule_${Anchor}_balanced_best.png"
