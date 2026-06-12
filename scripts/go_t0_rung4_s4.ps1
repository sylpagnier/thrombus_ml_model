# Rung 4 step s4: band GNN gate residual in E(t) + eval + viz.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_t0_rung4_s4.ps1" -Fresh
#   powershell ... -Fresh -Epochs 40 -ValAnchor patient007

param(
    [string] $Anchor = "patient007",
    [string] $Times = "0,7,15,22,27,40,53",
    [string] $ValAnchor = "patient007",
    [int] $Epochs = 40,
    [string] $Ckpt = "outputs/biochem/t0_r4_s4_band_ml/best.pth",
    [switch] $SkipTrain,
    [switch] $Fresh,
    [string] $TeacherCkpt = "outputs/biochem/biochem_teacher_last.pth"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"
$env:CUDA_VISIBLE_DEVICES = if ($env:CUDA_VISIBLE_DEVICES) { $env:CUDA_VISIBLE_DEVICES } else { "0" }
$env:T0_RUNG4_STEP = "s4_band_ml"
$env:T0_R4_S4_CKPT = $Ckpt

$ckptPath = Join-Path $RepoRoot $Ckpt
if ($Fresh -and (Test-Path $ckptPath)) {
    Remove-Item $ckptPath -Force
    $jsonSide = Join-Path (Split-Path $ckptPath) "best.json"
    if (Test-Path $jsonSide) { Remove-Item $jsonSide -Force }
    $logPath = Join-Path (Split-Path $ckptPath) "train_log.jsonl"
    if (Test-Path $logPath) { Remove-Item $logPath -Force }
}

if (-not $SkipTrain) {
    if (-not (Test-Path $ckptPath)) {
        Write-Host "[NEW] Train s4_band_ml (val=$ValAnchor, epochs=$Epochs)" -ForegroundColor Cyan
        $trainArgs = @(
            "-m", "src.training.train_t0_r4_s4_band_ml",
            "--val-anchor", $ValAnchor,
            "--epochs", "$Epochs",
            "--time-stride", "2",
            "--loc-scale", "0.75",
            "--w-fp", "2.0",
            "--early-stop", "12",
            "--out", $Ckpt
        )
        Invoke-PythonRcCheck -Label "rung4 s4 train" -PyArgs $trainArgs
    } else {
        Write-Host "[skip] checkpoint exists: $Ckpt (use -Fresh to retrain)" -ForegroundColor Yellow
    }
}

Write-Host "[NEW] Rung 4.s4 eval ($Anchor)" -ForegroundColor Cyan
$evalArgs = @(
    "scripts/eval_t0_rung4_step.py",
    "--anchor", $Anchor,
    "--times", $Times,
    "--step", "s4_band_ml"
)
if ($TeacherCkpt) { $evalArgs += @("--teacher-ckpt", $TeacherCkpt) }
Invoke-PythonRcCheck -Label "rung4 s4 eval" -PyArgs $evalArgs

Write-Host "[NEW] Rung 4.s4 viz ($Anchor)" -ForegroundColor Cyan
$vizArgs = @(
    "scripts/viz_t0_rung4_step.py",
    "--anchor", $Anchor,
    "--max-frames", "10",
    "--step", "s4_band_ml"
)
if ($TeacherCkpt) { $vizArgs += @("--teacher-ckpt", $TeacherCkpt) }
Invoke-PythonRcCheck -Label "rung4 s4 viz" -PyArgs $vizArgs

Write-Host "[OK] ckpt=$Ckpt" -ForegroundColor Green
Write-Host "[OK] eval=outputs/biochem/clot_trigger/t0_rung4_s4_band_ml_${Anchor}.json" -ForegroundColor Green
Write-Host "[OK] viz=outputs/biochem/viz/clot_trigger/t0_rung4_s4_band_ml_${Anchor}.png" -ForegroundColor Green
