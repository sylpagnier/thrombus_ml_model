# Star 4 (T4): live GNODE teacher rollout + pred kine (slow; audit / overnight).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_trigger_t4.ps1"
#   powershell ... -VizOnly -Anchor patient007

param(
    [string] $Checkpoint = "outputs/biochem/clot_trigger/t1/clot_trigger_t1_best.pth",
    [string] $Teacher = "outputs/biochem/biochem_teacher_best_high_mu.pth",
    [string] $KineCkpt = "outputs/kinematics/kinematics_best.pth",
    [string] $Out = "outputs/biochem/clot_trigger/t4_live_teacher.json",
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

if (-not (Test-Path (Join-Path $RepoRoot $KineCkpt))) {
    Write-Host "[ERR] missing kinematics ckpt: $KineCkpt" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path (Join-Path $RepoRoot $Checkpoint))) {
    Write-Host "[ERR] missing T1 trigger ckpt: $Checkpoint" -ForegroundColor Red
    exit 1
}

$teacherPath = Join-Path $RepoRoot $Teacher
if (-not (Test-Path $teacherPath)) {
    $fallback = Join-Path $RepoRoot "outputs\biochem\biochem_teacher_last.pth"
    if (Test-Path $fallback) {
        Write-Host "[WARN] using biochem_teacher_last.pth" -ForegroundColor Yellow
        $Teacher = "outputs/biochem/biochem_teacher_last.pth"
    } else {
        Write-Host "[ERR] missing teacher: $Teacher" -ForegroundColor Red
        exit 1
    }
}

New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot $VizDir) | Out-Null

if (-not $SkipEval -and -not $VizOnly) {
    Write-Host "[NEW] T4 eval (pred kine + LIVE teacher species rollout)" -ForegroundColor Cyan
    Write-Host "[WARN] This is slow (~hours). Prefer T3 with dumped cache for daily work." -ForegroundColor Yellow
    $evalArgs = @(
        "scripts/eval_clot_trigger_t3_full_stack.py",
        "--species-source", "live",
        "--checkpoint", $Checkpoint,
        "--teacher", $Teacher,
        "--kine-ckpt", $KineCkpt,
        "--out", $Out,
        "--val", $Val,
        "--progress-step", "$ProgressStep"
    )
    if ($Quiet) { $evalArgs += "--quiet" }
    Invoke-PythonRcCheck -Label "t4 eval live" -PyArgs $evalArgs
    Write-Host "[OK] results -> $Out" -ForegroundColor Green
}

foreach ($anc in @($Anchor, $Anchor2)) {
    if (-not $anc) { continue }
    Write-Host "[NEW] T4 viz $anc" -ForegroundColor Cyan
    Invoke-PythonRcCheck -Label "t4 viz $anc" -PyArgs @(
        "scripts/viz_clot_trigger_t3.py",
        "--species-source", "live",
        "--anchor", $anc,
        "--checkpoint", $Checkpoint,
        "--teacher", $Teacher,
        "--kine-ckpt", $KineCkpt,
        "--out", "$VizDir/t4_$anc.png"
    )
}

Write-Host ""
Write-Host "[OK] T4 done (live teacher)" -ForegroundColor Green
