# Rung 4 mini-ladder (CUDA): deploy Step s0 default, eval + viz vs R2/R4 teacher.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_t0_rung4_step.ps1"
#   powershell ... -Step s0 -Anchor patient007
#   powershell ... -Step s0_oracle_fi_mat   # audit ceiling only

param(
    [string] $Anchor = "patient007",
    [string] $Times = "0,27,53",
    [string] $Step = "s0",
    [string] $TeacherCkpt = "outputs/biochem/biochem_teacher_last.pth"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"
$env:CUDA_VISIBLE_DEVICES = if ($env:CUDA_VISIBLE_DEVICES) { $env:CUDA_VISIBLE_DEVICES } else { "0" }

Write-Host "[NEW] Rung 4.$Step eval ($Anchor)" -ForegroundColor Cyan
$evalArgs = @(
    "scripts/eval_t0_rung4_step.py",
    "--anchor", $Anchor,
    "--times", $Times,
    "--step", $Step
)
if ($TeacherCkpt) { $evalArgs += @("--teacher-ckpt", $TeacherCkpt) }
Invoke-PythonRcCheck -Label "rung4 step eval" -PyArgs $evalArgs

Write-Host "[NEW] Rung 4.$Step viz ($Anchor)" -ForegroundColor Cyan
$vizArgs = @(
    "scripts/viz_t0_rung4_step.py",
    "--anchor", $Anchor,
    "--max-frames", "10",
    "--step", $Step
)
if ($TeacherCkpt) { $vizArgs += @("--teacher-ckpt", $TeacherCkpt) }
Invoke-PythonRcCheck -Label "rung4 step viz" -PyArgs $vizArgs

Write-Host "[OK] eval=outputs/biochem/clot_trigger/t0_rung4_${Step}_${Anchor}.json" -ForegroundColor Green
Write-Host "[OK] viz=outputs/biochem/viz/clot_trigger/t0_rung4_${Step}_${Anchor}.png" -ForegroundColor Green
