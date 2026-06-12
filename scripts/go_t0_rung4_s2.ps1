# Rung 4 step s2: train FI/Mat corrector + eval + viz.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_t0_rung4_s2.ps1"
#   powershell ... -Fresh -Epochs 50 -ValAnchor patient007

param(
    [string] $Anchor = "patient007",
    [string] $Times = "0,7,15,22,27,40,53",
    [string] $ValAnchor = "patient007",
    [int] $Epochs = 50,
    [string] $Mode = "loc",
    [string] $Ckpt = "outputs/biochem/t0_r4_s2_loc/best.pth",
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
$env:T0_RUNG4_STEP = "s2_species"
$env:T0_R4_S2_MODE = $Mode
$env:T0_R4_S2_CKPT = $Ckpt

$ckptPath = Join-Path $RepoRoot $Ckpt
if ($Fresh -and (Test-Path $ckptPath)) {
    Remove-Item $ckptPath -Force
    $jsonSide = [System.IO.Path]::ChangeExtension($ckptPath, ".json")
    if (Test-Path $jsonSide) { Remove-Item $jsonSide -Force }
}

if (-not $SkipTrain) {
    if (-not (Test-Path $ckptPath)) {
        Write-Host "[NEW] Train s2_species mode=$Mode (val=$ValAnchor, epochs=$Epochs)" -ForegroundColor Cyan
        $trainArgs = @(
            "-m", "src.training.train_t0_r4_s2_species",
            "--mode", $Mode,
            "--val-anchor", $ValAnchor,
            "--epochs", "$Epochs",
            "--time-stride", "2",
            "--out", $Ckpt
        )
        Invoke-PythonRcCheck -Label "rung4 s2 train" -PyArgs $trainArgs
    } else {
        Write-Host "[skip] checkpoint exists: $Ckpt (use -Fresh to retrain)" -ForegroundColor Yellow
    }
}

Write-Host "[NEW] Rung 4.s2 eval ($Anchor)" -ForegroundColor Cyan
$evalArgs = @(
    "scripts/eval_t0_rung4_step.py",
    "--anchor", $Anchor,
    "--times", $Times,
    "--step", "s2_species"
)
if ($TeacherCkpt) { $evalArgs += @("--teacher-ckpt", $TeacherCkpt) }
Invoke-PythonRcCheck -Label "rung4 s2 eval" -PyArgs $evalArgs

Write-Host "[NEW] Rung 4.s2 viz ($Anchor)" -ForegroundColor Cyan
$vizArgs = @(
    "scripts/viz_t0_rung4_step.py",
    "--anchor", $Anchor,
    "--max-frames", "10",
    "--step", "s2_species"
)
if ($TeacherCkpt) { $vizArgs += @("--teacher-ckpt", $TeacherCkpt) }
Invoke-PythonRcCheck -Label "rung4 s2 viz" -PyArgs $vizArgs

Write-Host "[OK] ckpt=$Ckpt" -ForegroundColor Green
Write-Host "[OK] eval=outputs/biochem/clot_trigger/t0_rung4_s2_species_${Anchor}.json" -ForegroundColor Green
Write-Host "[OK] viz=outputs/biochem/viz/clot_trigger/t0_rung4_s2_species_${Anchor}.png" -ForegroundColor Green
