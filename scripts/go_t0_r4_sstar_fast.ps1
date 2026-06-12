# S-star fast dev loop (T2): preflight + single-axis train/eval/viz (~25 min).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_t0_r4_sstar_fast.ps1" -Axis gate
#   powershell ... -Axis species -SkipPreflight
#   powershell ... -Axis rules -SkipTrain          # s*_G1 rule sweep only
#   powershell ... -Axis dyn -Epochs 20 -Fresh

param(
    [ValidateSet("gate", "species", "dyn", "full", "rules")]
    [string] $Axis = "gate",
    [string] $Anchor = "patient007",
    [string] $ValAnchor = "patient007",
    [string] $Times = "0,15,29,53",
    [int] $Epochs = 20,
    [switch] $Fresh,
    [switch] $SkipPreflight,
    [switch] $SkipTrain,
    [switch] $SkipViz,
    [string] $TeacherCkpt = "outputs/biochem/biochem_teacher_last.pth"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")

$env:PYTHONUNBUFFERED = "1"
$env:CUDA_VISIBLE_DEVICES = if ($env:CUDA_VISIBLE_DEVICES) { $env:CUDA_VISIBLE_DEVICES } else { "0" }

$AxisMap = @{
    gate    = @{ Recipe = "s4_gate_commit"; Step = "s_star_gate"; SubDir = "gate" }
    species = @{ Recipe = "s_star_species"; Step = "s_star_species"; SubDir = "species" }
    dyn     = @{ Recipe = "s_star_dyn"; Step = "s_star_dyn"; SubDir = "dyn" }
    full    = @{ Recipe = "s_star_full"; Step = "s_star_full"; SubDir = "full" }
    rules   = @{ Recipe = "s_star_g0_rules"; Step = "s0"; SubDir = "rules" }
}

$cfg = $AxisMap[$Axis]
$OutDir = Join-Path $RepoRoot "outputs\biochem\t0_r4_sstar\$($cfg.SubDir)"
$Ckpt = Join-Path $OutDir "best.pth"
$PreflightOut = Join-Path $OutDir "preflight.json"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$env:T0_R4_SWEEP_CKPT = $Ckpt
$env:T0_RUNG4_STEP = $cfg.Step

if (-not $SkipPreflight) {
    Write-Host "[NEW] T0 preflight ($Anchor)" -ForegroundColor Cyan
    $pfArgs = @(
        "scripts/diagnose_t0_r4_sweep_preflight.py",
        "--anchor", $Anchor,
        "--out", $PreflightOut
    )
    Invoke-PythonRcCheck -Label "s-star preflight" -PyArgs $pfArgs
}

if ($Axis -eq "rules") {
    $SweepOut = Join-Path $OutDir "s0_sweep.json"
    Write-Host "[NEW] s*_G1 rule sweep ($Anchor) grid=g1" -ForegroundColor Cyan
    Invoke-PythonRcCheck -Label "s0 rule sweep" -PyArgs @(
        "scripts/sweep_t0_r4_s0.py",
        "--anchors", $Anchor,
        "--grid", "g1",
        "--out", $SweepOut
    )
    if ($SkipTrain) { exit 0 }
}

if ($Fresh -and (Test-Path $Ckpt)) {
    Remove-Item $Ckpt -Force
    $jsonSide = Join-Path $OutDir "best.json"
    if (Test-Path $jsonSide) { Remove-Item $jsonSide -Force }
    $logPath = Join-Path $OutDir "train_log.jsonl"
    if (Test-Path $logPath) { Remove-Item $logPath -Force }
}

if (-not $SkipTrain -and $Axis -ne "rules") {
    Write-Host "[NEW] S-star train axis=$Axis recipe=$($cfg.Recipe) val=$ValAnchor ep=$Epochs" -ForegroundColor Cyan
    $trainArgs = @(
        "-m", "src.training.train_t0_r4_sweep_leg",
        "--recipe", $cfg.Recipe,
        "--val-anchor", $ValAnchor,
        "--epochs", "$Epochs",
        "--early-stop", "10",
        "--out", $Ckpt
    )
    Invoke-PythonRcCheck -Label "s-star train" -PyArgs $trainArgs
}

if ($Axis -eq "rules") {
    Write-Host "[OK] rules sweep done (no ML train for rules axis)" -ForegroundColor Green
    exit 0
}

Write-Host "[NEW] S-star eval step=$($cfg.Step) ($Anchor)" -ForegroundColor Cyan
$evalArgs = @(
    "scripts/eval_t0_rung4_step.py",
    "--anchor", $Anchor,
    "--times", $Times,
    "--step", $cfg.Step
)
if ($TeacherCkpt) { $evalArgs += @("--teacher-ckpt", $TeacherCkpt) }
Invoke-PythonRcCheck -Label "s-star eval" -PyArgs $evalArgs

if (-not $SkipViz) {
    Write-Host "[NEW] S-star viz ($Anchor)" -ForegroundColor Cyan
    $vizArgs = @(
        "scripts/viz_t0_rung4_step.py",
        "--anchor", $Anchor,
        "--max-frames", "10",
        "--step", $cfg.Step
    )
    if ($TeacherCkpt) { $vizArgs += @("--teacher-ckpt", $TeacherCkpt) }
    Invoke-PythonRcCheck -Label "s-star viz" -PyArgs $vizArgs
}

Write-Host "[OK] ckpt=$Ckpt" -ForegroundColor Green
Write-Host "[OK] eval=outputs/biochem/clot_trigger/t0_rung4_$($cfg.Step)_${Anchor}.json" -ForegroundColor Green
