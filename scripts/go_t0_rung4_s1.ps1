# Rung 4 step s1: train residual phi MLP (optional) + eval + viz.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_t0_rung4_s1.ps1"
#   powershell ... -SkipTrain
#   powershell ... -Fresh -Epochs 40 -ValAnchor patient007

param(
    [string] $Anchor = "patient007",
    [string] $Times = "0,7,15,22,27,40,53",
    [string] $ValAnchor = "patient007",
    [int] $Epochs = 40,
    [string] $Ckpt = "outputs/biochem/t0_r4_s1_mlp_phi/best.pth",
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
$env:T0_RUNG4_STEP = "s1_mlp_phi"
$env:T0_R4_S1_CKPT = $Ckpt

$ckptPath = Join-Path $RepoRoot $Ckpt
if ($Fresh -and (Test-Path $ckptPath)) {
    Remove-Item $ckptPath -Force
    $jsonSide = [System.IO.Path]::ChangeExtension($ckptPath, ".json")
    if (Test-Path $jsonSide) { Remove-Item $jsonSide -Force }
}

if (-not $SkipTrain) {
    if (-not (Test-Path $ckptPath)) {
        Write-Host "[NEW] Train s1_mlp_phi (val=$ValAnchor, epochs=$Epochs)" -ForegroundColor Cyan
        $trainArgs = @(
            "-m", "src.training.train_t0_r4_s1_mlp_phi",
            "--val-anchor", $ValAnchor,
            "--epochs", "$Epochs",
            "--out", $Ckpt
        )
        Invoke-PythonRcCheck -Label "rung4 s1 train" -PyArgs $trainArgs
    } else {
        Write-Host "[skip] checkpoint exists: $Ckpt (use -Fresh to retrain)" -ForegroundColor Yellow
    }
}

Write-Host "[NEW] Rung 4.s1 eval ($Anchor)" -ForegroundColor Cyan
$evalArgs = @(
    "scripts/eval_t0_rung4_step.py",
    "--anchor", $Anchor,
    "--times", $Times,
    "--step", "s1_mlp_phi"
)
if ($TeacherCkpt) { $evalArgs += @("--teacher-ckpt", $TeacherCkpt) }
Invoke-PythonRcCheck -Label "rung4 s1 eval" -PyArgs $evalArgs

Write-Host "[NEW] Rung 4.s1 viz ($Anchor)" -ForegroundColor Cyan
$vizArgs = @(
    "scripts/viz_t0_rung4_step.py",
    "--anchor", $Anchor,
    "--max-frames", "10",
    "--step", "s1_mlp_phi"
)
if ($TeacherCkpt) { $vizArgs += @("--teacher-ckpt", $TeacherCkpt) }
Invoke-PythonRcCheck -Label "rung4 s1 viz" -PyArgs $vizArgs

Write-Host "[OK] ckpt=$Ckpt" -ForegroundColor Green
Write-Host "[OK] eval=outputs/biochem/clot_trigger/t0_rung4_s1_mlp_phi_${Anchor}.json" -ForegroundColor Green
Write-Host "[OK] viz=outputs/biochem/viz/clot_trigger/t0_rung4_s1_mlp_phi_${Anchor}.png" -ForegroundColor Green
