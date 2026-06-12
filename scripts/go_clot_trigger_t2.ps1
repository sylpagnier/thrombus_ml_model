# Star 2 (T2): eval frozen T1 trigger with pred GINO-DEQ flow + GT species, then viz.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_trigger_t2.ps1"
#   powershell ... -VizOnly
#   powershell ... -SkipEval -Anchor patient002

param(
    [string] $Checkpoint = "outputs/biochem/clot_trigger/t1/clot_trigger_t1_best.pth",
    [string] $KineCkpt = "outputs/kinematics/kinematics_best.pth",
    [string] $Out = "outputs/biochem/clot_trigger/t2_pred_flow.json",
    [string] $Val = "patient007",
    [string] $Anchor = "patient007",
    [string] $Anchor2 = "patient002",
    [string] $VizDir = "outputs/biochem/viz/clot_trigger",
    [switch] $SkipEval,
    [switch] $VizOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")

$env:PYTHONUNBUFFERED = "1"

if (-not (Test-Path (Join-Path $RepoRoot $KineCkpt))) {
    Write-Host "[ERR] missing kinematics ckpt: $KineCkpt" -ForegroundColor Red
    Write-Host "[i]  run go_kinematics_production_allfix.ps1 or point -KineCkpt" -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path (Join-Path $RepoRoot $Checkpoint))) {
    Write-Host "[ERR] missing T1 trigger ckpt: $Checkpoint" -ForegroundColor Red
    Write-Host "[i]  run go_clot_trigger_t1.ps1 first" -ForegroundColor Yellow
    exit 1
}

New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot $VizDir) | Out-Null

if (-not $SkipEval -and -not $VizOnly) {
    Write-Host "[NEW] T2 eval (pred kine + GT species, frozen T1 trigger)" -ForegroundColor Cyan
    Invoke-PythonRcCheck -Label "t2 eval" -PyArgs @(
        "scripts/eval_clot_trigger_t2_pred_flow.py",
        "--checkpoint", $Checkpoint,
        "--kine-ckpt", $KineCkpt,
        "--out", $Out,
        "--val", $Val
    )
    Write-Host "[OK] results -> $Out" -ForegroundColor Green
}

foreach ($anc in @($Anchor, $Anchor2)) {
    if (-not $anc) { continue }
    Write-Host "[NEW] T2 viz $anc" -ForegroundColor Cyan
    Invoke-PythonRcCheck -Label "t2 viz $anc" -PyArgs @(
        "scripts/viz_clot_trigger_t2.py",
        "--anchor", $anc,
        "--checkpoint", $Checkpoint,
        "--kine-ckpt", $KineCkpt,
        "--out", "$VizDir/t2_$anc.png"
    )
}

Write-Host ""
Write-Host "[OK] viz -> $VizDir/t2_$Anchor.png (and $Anchor2)" -ForegroundColor Green
if (-not $SkipEval -and -not $VizOnly) {
    Write-Host "[OK] eval -> $Out" -ForegroundColor Green
}
