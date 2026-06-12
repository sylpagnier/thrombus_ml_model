# Rung 1 (COMSOL sr) + Rung 2 (proxy gamma) eval and viz.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_t0_rung12.ps1"

param(
    [string] $Anchor = "patient007",
    [string] $Times = "0,27,53",
    [string] $EvalOut = "outputs/biochem/clot_trigger/t0_rung12_eval.json",
    [string] $VizOut = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

$srTxt = Join-Path $RepoRoot "data/processed/cfd_results_biochem/${Anchor}_sr.txt"
if (Test-Path $srTxt) {
    Write-Host "[NEW] Build COMSOL spf.sr sidecar ($Anchor)" -ForegroundColor Cyan
    Invoke-PythonRcCheck -Label "build sr sidecar" -PyArgs @(
        "scripts/build_comsol_sr_sidecar.py", "--anchor", $Anchor
    )
}

Write-Host "[NEW] Rung 1+2 eval ($Anchor)" -ForegroundColor Cyan
Invoke-PythonRcCheck -Label "rung12 eval" -PyArgs @(
    "scripts/eval_t0_rung12.py",
    "--anchor", $Anchor,
    "--times", $Times,
    "--out", $EvalOut
)

Write-Host "[NEW] Rung 1+2 viz ($Anchor)" -ForegroundColor Cyan
$vizArgs = @("scripts/viz_t0_rung12.py", "--anchor", $Anchor, "--max-frames", "10")
if ($VizOut) { $vizArgs += @("--out", $VizOut) }
Invoke-PythonRcCheck -Label "rung12 viz" -PyArgs $vizArgs

Write-Host "[OK] eval=$EvalOut viz=outputs/biochem/viz/clot_trigger/t0_rung12_$Anchor.png" -ForegroundColor Green
