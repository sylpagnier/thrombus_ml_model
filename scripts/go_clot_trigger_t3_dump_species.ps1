# One-time (or refresh): cache GNODE teacher species onto anchor graphs for T3 eval.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_trigger_t3_dump_species.ps1"
#   powershell ... -Force
#   powershell ... -Teacher outputs/biochem/biochem_teacher_best_high_mu.pth

param(
    [string] $Teacher = "outputs/biochem/biochem_teacher_best_high_mu.pth",
    [string] $OutDir = "outputs/biochem/anchors_teacher_species",
    [int] $TimeStride = 6,
    [int] $MinSteps = 8,
    [switch] $Force
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")

$env:PYTHONUNBUFFERED = "1"

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

Write-Host "[NEW] dump teacher species -> $OutDir (stride=$TimeStride min_steps=$MinSteps)" -ForegroundColor Cyan
$dumpArgs = @(
    "scripts/dump_teacher_species_to_anchors.py",
    "--teacher", $Teacher,
    "--out-dir", $OutDir,
    "--device", "cuda",
    "--time-stride", "$TimeStride",
    "--min-steps", "$MinSteps"
)
if ($Force) { $dumpArgs += "--force" }
Invoke-PythonRcCheck -Label "dump teacher species" -PyArgs $dumpArgs

Write-Host "[OK] dumped anchors -> $OutDir" -ForegroundColor Green
Write-Host "[i]  next: go_clot_trigger_t3.ps1 (fast eval uses this cache)" -ForegroundColor DarkGray
