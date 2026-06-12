# Rung 1-3 T0 eval + clot comparison viz (GT vs R2 vs R3).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_t0_rung123.ps1"

param(
    [string] $Anchor = "patient007",
    [string] $Times = "0,27,53",
    [string] $KineCkpt = "outputs/kinematics/kinematics_best.pth",
    [string] $EvalOut = "outputs/biochem/clot_trigger/t0_rung123_eval.json",
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

if (-not (Test-Path (Join-Path $RepoRoot $KineCkpt))) {
    Write-Host "[ERR] missing kinematics ckpt: $KineCkpt" -ForegroundColor Red
    exit 1
}

Write-Host "[NEW] Rung 1+2+3 eval ($Anchor)" -ForegroundColor Cyan
Invoke-PythonRcCheck -Label "rung123 eval" -PyArgs @(
    "scripts/eval_t0_rung123.py",
    "--anchor", $Anchor,
    "--times", $Times,
    "--kine-ckpt", $KineCkpt,
    "--out", $EvalOut
)

Write-Host "[NEW] Rung 2+3 clot viz ($Anchor)" -ForegroundColor Cyan
$vizArgs = @(
    "scripts/viz_t0_rung123.py",
    "--anchor", $Anchor,
    "--max-frames", "10",
    "--kine-ckpt", $KineCkpt
)
if ($VizOut) { $vizArgs += @("--out", $VizOut) }
Invoke-PythonRcCheck -Label "rung123 viz" -PyArgs $vizArgs

Write-Host "[OK] eval=$EvalOut viz=outputs/biochem/viz/clot_trigger/t0_rung123_$Anchor.png" -ForegroundColor Green
