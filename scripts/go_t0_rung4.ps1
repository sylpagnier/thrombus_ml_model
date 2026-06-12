# Rung 4: GT flow + pred teacher species + proxy gamma (eval + clot viz vs R2).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_t0_rung4.ps1"
#   powershell ... -TeacherCkpt outputs/biochem/biochem_teacher_last.pth

param(
    [string] $Anchor = "patient007",
    [string] $Times = "0,27,53",
    [string] $TeacherCkpt = "",
    [int] $TeacherTimeStride = 6,
    [string] $SpeciesDump = "",
    [string] $EvalOut = "outputs/biochem/clot_trigger/t0_rung1234_eval.json",
    [string] $VizOut = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

$evalOutR4 = "outputs/biochem/clot_trigger/t0_rung4_eval.json"
$evalArgs = @(
    "scripts/eval_t0_rung4.py",
    "--anchor", $Anchor,
    "--times", $Times,
    "--out", $evalOutR4
)
if ($TeacherCkpt) { $evalArgs += @("--teacher-ckpt", $TeacherCkpt) }
$evalArgs += @("--teacher-time-stride", "$TeacherTimeStride")
if ($SpeciesDump) { $evalArgs += @("--species-dump", $SpeciesDump) }

Write-Host "[NEW] Rung 4 eval ($Anchor) -- uses species dump if present else live rollout" -ForegroundColor Cyan
Invoke-PythonRcCheck -Label "rung1234 eval" -PyArgs $evalArgs

$vizArgs = @("scripts/viz_t0_rung124.py", "--anchor", $Anchor, "--max-frames", "10")
if ($TeacherCkpt) { $vizArgs += @("--teacher-ckpt", $TeacherCkpt) }
$vizArgs += @("--teacher-time-stride", "$TeacherTimeStride")
if ($SpeciesDump) { $vizArgs += @("--species-dump", $SpeciesDump) }
if ($VizOut) { $vizArgs += @("--out", $VizOut) }

Write-Host "[NEW] Rung 2+4 clot viz ($Anchor)" -ForegroundColor Cyan
Invoke-PythonRcCheck -Label "rung124 viz" -PyArgs $vizArgs

Write-Host "[OK] eval=$evalOutR4 viz=outputs/biochem/viz/clot_trigger/t0_rung124_$Anchor.png" -ForegroundColor Green
