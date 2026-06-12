# T0 physics trigger sweep (~15 min LOAO) + viz top legs.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_trigger_t0_physics_sweep.ps1"
#   powershell ... -VizOnly
#   powershell ... -SweepOnly

param(
    [string] $OutDir = "outputs/biochem/clot_trigger/t0_physics_sweep",
    [string] $Val = "patient007",
    [string] $VizAnchor = "patient007",
    [string] $VizAnchor2 = "patient002",
    [int] $TopK = 3,
    [switch] $SubtractT0,
    [double] $DisplayMin = 0.0,
    [switch] $SweepOnly,
    [switch] $VizOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")

$env:PYTHONUNBUFFERED = "1"

if (-not $VizOnly) {
    Write-Host "[NEW] T0 physics sweep (14 legs x 6 anchors, deploy nucleation)" -ForegroundColor Cyan
    $null = Invoke-PythonRcCheck -Label "t0 physics sweep" -PyArgs @(
        "scripts/sweep_clot_trigger_t0_physics.py",
        "--out-dir", $OutDir,
        "--val", $Val
    )
    Write-Host "[OK] sweep -> $OutDir/sweep_index.json" -ForegroundColor Green
}

if (-not $SweepOnly) {
    $VizDir = "outputs/biochem/viz/clot_trigger/t0_sweep"
    Write-Host "[NEW] T0 sweep viz top $TopK legs ($VizAnchor, $VizAnchor2)" -ForegroundColor Cyan
    $vizArgs = @(
        "scripts/viz_clot_trigger_t0_sweep.py",
        "--sweep-dir", $OutDir,
        "--top-k", "$TopK",
        "--anchor", $VizAnchor,
        "--anchor2", $VizAnchor2,
        "--viz-dir", $VizDir,
        "--display-min", "$DisplayMin"
    )
    if ($SubtractT0) {
        $vizArgs += "--subtract-t0"
    }
    $null = Invoke-PythonRcCheck -Label "t0 sweep viz" -PyArgs $vizArgs
    Write-Host "[OK] viz -> $VizDir" -ForegroundColor Green
    $indexPath = Join-Path $RepoRoot "$OutDir/sweep_index.json"
    if (Test-Path $indexPath) {
        Write-Host "[i] ranking:" -ForegroundColor DarkGray
        $null = Invoke-PythonRcCheck -Label "t0 sweep summary" -PyArgs @(
            "scripts/summarize_clot_trigger_t0_physics_sweep.py",
            "--index", (Join-Path $OutDir "sweep_index.json")
        )
    }
}
