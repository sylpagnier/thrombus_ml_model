# Star 6 (T6): T5 deploy species + full mu/phi -> GINO-DEQ coupling each macro step.
#
# Fast path (T5 pred-flow species dump, ~minutes):
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_trigger_t6.ps1" -SpeciesSource dumped
#
# Honest path (live T5 deploy teacher, hours):
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_trigger_t6.ps1" -SpeciesSource live
#
# Prereq: go_clot_trigger_t5.ps1 (deploy teacher + predkine dump)

param(
    [ValidateSet("dumped", "live")]
    [string] $SpeciesSource = "dumped",
    [string] $Checkpoint = "outputs/biochem/clot_trigger/t1/clot_trigger_t1_best.pth",
    [string] $Teacher = "",
    [string] $DumpDir = "outputs/biochem/anchors_teacher_species_predkine",
    [string] $KineCkpt = "outputs/kinematics/kinematics_best.pth",
    [string] $Out = "",
    [string] $Val = "patient007",
    [string] $Anchor = "patient007",
    [string] $Anchor2 = "patient002",
    [string] $VizDir = "outputs/biochem/viz/clot_trigger",
    [int] $ProgressStep = 5,
    [switch] $SkipEval,
    [switch] $VizOnly,
    [switch] $Quiet
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")

$env:PYTHONUNBUFFERED = "1"

if (-not $Teacher) {
    $Teacher = "outputs/biochem/clot_trigger/t5_deploy_teacher/biochem_teacher_deploy.pth"
    if (-not (Test-Path (Join-Path $RepoRoot $Teacher))) {
        $Teacher = "outputs/biochem/biochem_teacher_best_high_mu.pth"
    }
}

$live = ($SpeciesSource -eq "live")
if (-not (Test-Path (Join-Path $RepoRoot $KineCkpt))) {
    Write-Host "[ERR] missing kinematics ckpt: $KineCkpt" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path (Join-Path $RepoRoot $Checkpoint))) {
    Write-Host "[ERR] missing T1 trigger ckpt: $Checkpoint" -ForegroundColor Red
    exit 1
}

if (-not $live) {
    $dumpPath = Join-Path $RepoRoot $DumpDir
    if (-not (Test-Path (Join-Path $dumpPath "patient007.pt"))) {
        Write-Host "[ERR] missing T5 predkine dump: $DumpDir" -ForegroundColor Red
        Write-Host "[i]  run go_clot_trigger_t5.ps1 first" -ForegroundColor DarkGray
        exit 1
    }
} else {
    $teacherPath = Join-Path $RepoRoot $Teacher
    if (-not (Test-Path $teacherPath)) {
        Write-Host "[ERR] missing teacher: $Teacher (run go_clot_trigger_t5.ps1)" -ForegroundColor Red
        exit 1
    }
}

if (-not $Out) {
    if ($live) {
        $Out = "outputs/biochem/clot_trigger/t6_coupled_live.json"
    } else {
        $Out = "outputs/biochem/clot_trigger/t6_coupled_dumped.json"
    }
}

New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot $VizDir) | Out-Null

if (-not $SkipEval -and -not $VizOnly) {
    Write-Host "[NEW] T6 eval (coupled kine + $SpeciesSource species)" -ForegroundColor Cyan
    $evalArgs = @(
        "scripts/eval_clot_trigger_t3_full_stack.py",
        "--species-source", $SpeciesSource,
        "--coupling", "full",
        "--checkpoint", $Checkpoint,
        "--kine-ckpt", $KineCkpt,
        "--out", $Out,
        "--val", $Val,
        "--progress-step", "$ProgressStep"
    )
    if ($live) {
        $evalArgs += @("--teacher", $Teacher)
    } else {
        $evalArgs += @("--anchor-dir", $DumpDir)
    }
    if ($Quiet) { $evalArgs += "--quiet" }
    Invoke-PythonRcCheck -Label "t6 eval coupled" -PyArgs $evalArgs
    Write-Host "[OK] results -> $Out" -ForegroundColor Green
}

foreach ($anc in @($Anchor, $Anchor2)) {
    if (-not $anc) { continue }
    Write-Host "[NEW] T6 viz $anc" -ForegroundColor Cyan
    $vizArgs = @(
        "scripts/viz_clot_trigger_t3.py",
        "--species-source", $SpeciesSource,
        "--coupling", "full",
        "--anchor", $anc,
        "--checkpoint", $Checkpoint,
        "--teacher", $Teacher,
        "--kine-ckpt", $KineCkpt,
        "--out", "$VizDir/t6_$anc.png"
    )
    if (-not $live) {
        $vizArgs += @("--anchor-dir", $DumpDir)
    }
    Invoke-PythonRcCheck -Label "t6 viz $anc" -PyArgs $vizArgs
}

Write-Host ""
Write-Host "[OK] T6 done (coupled kine feedback)" -ForegroundColor Green
