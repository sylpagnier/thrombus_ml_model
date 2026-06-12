# Star 3 (T3): pred kine + **cached** teacher species dump (fast, ~T2 speed).
#
# Prereq: run go_clot_trigger_t3_dump_species.ps1 once (or -DumpIfMissing below).
# Requires honest T1 ckpt (retrain with go_clot_trigger_t1.ps1 -Fresh after pivot).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_trigger_t3.ps1"
#   powershell ... -DumpIfMissing
#   powershell ... -VizOnly -Anchor patient007

param(
    [string] $Checkpoint = "outputs/biochem/clot_trigger/t1/clot_trigger_t1_best.pth",
    [string] $DumpDir = "outputs/biochem/anchors_teacher_species",
    [string] $KineCkpt = "outputs/kinematics/kinematics_best.pth",
    [string] $Out = "outputs/biochem/clot_trigger/t3_dumped_species.json",
    [string] $Val = "patient007",
    [string] $Anchor = "patient007",
    [string] $Anchor2 = "patient002",
    [string] $VizDir = "outputs/biochem/viz/clot_trigger",
    [int] $ProgressStep = 5,
    [switch] $DumpIfMissing,
    [switch] $SkipEval,
    [switch] $VizOnly,
    [switch] $NucleationBand,
    [switch] $OracleBand,
    [switch] $Quiet
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")

$env:PYTHONUNBUFFERED = "1"

if (-not (Test-Path (Join-Path $RepoRoot $KineCkpt))) {
    Write-Host "[ERR] missing kinematics ckpt: $KineCkpt" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path (Join-Path $RepoRoot $Checkpoint))) {
    Write-Host "[ERR] missing T1 trigger ckpt: $Checkpoint" -ForegroundColor Red
    exit 1
}

$dumpPath = Join-Path $RepoRoot $DumpDir
$needDump = -not (Test-Path (Join-Path $dumpPath "patient007.pt"))
if ($needDump -and ($DumpIfMissing -or (-not $VizOnly))) {
    Write-Host "[NEW] species cache missing -- running dump" -ForegroundColor Yellow
    & (Join-Path $PSScriptRoot "go_clot_trigger_t3_dump_species.ps1")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot $VizDir) | Out-Null

if (-not $SkipEval -and -not $VizOnly) {
    Write-Host "[NEW] T3 eval (pred kine + dumped species, full-mesh F1)" -ForegroundColor Cyan
    Write-Host "[i]  progress -> $($Out -replace '\.json$','.progress.jsonl')" -ForegroundColor DarkGray
    $evalArgs = @(
        "scripts/eval_clot_trigger_t3_full_stack.py",
        "--species-source", "dumped",
        "--anchor-dir", $DumpDir,
        "--checkpoint", $Checkpoint,
        "--kine-ckpt", $KineCkpt,
        "--out", $Out,
        "--val", $Val,
        "--progress-step", "$ProgressStep"
    )
    if ($Quiet) { $evalArgs += "--quiet" }
    if ($NucleationBand) { $evalArgs += "--nucleation-band" }
    if ($OracleBand) { $evalArgs += "--oracle-band" }
    Invoke-PythonRcCheck -Label "t3 eval dumped" -PyArgs $evalArgs
    Write-Host "[OK] results -> $Out" -ForegroundColor Green
}

foreach ($anc in @($Anchor, $Anchor2)) {
    if (-not $anc) { continue }
    Write-Host "[NEW] T3 viz $anc (full vessel)" -ForegroundColor Cyan
    $vizArgs = @(
        "scripts/viz_clot_trigger_t3.py",
        "--species-source", "dumped",
        "--anchor-dir", $DumpDir,
        "--anchor", $anc,
        "--checkpoint", $Checkpoint,
        "--kine-ckpt", $KineCkpt,
        "--out", "$VizDir/t3_${anc}.png"
    )
    if ($NucleationBand) { $vizArgs += "--nucleation-band" }
    if ($OracleBand) { $vizArgs += "--oracle-band" }
    Invoke-PythonRcCheck -Label "t3 viz $anc" -PyArgs $vizArgs
}

Write-Host ""
Write-Host "[OK] T3 done (dumped species). Live path: go_clot_trigger_t4.ps1" -ForegroundColor Green
