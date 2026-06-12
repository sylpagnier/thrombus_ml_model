# Species teacher health checks (passive FI/Mat on anchor graphs, CUDA).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_t0_species_diagnostics.ps1"

param(
    [string] $Device = "cuda",
    [string] $Only = "patient007"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

$ckpts = @(
    "outputs/biochem/biochem_teacher_last.pth",
    "outputs/biochem/biochem_teacher_passive_align_locked.pth",
    "outputs/biochem/biochem_teacher_passive_xy_locked.pth"
)

$outDir = Join-Path $RepoRoot "outputs/biochem/diagnostics"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$summary = Join-Path $outDir "species_teacher_health.txt"
"[NEW] species teacher diagnostics $(Get-Date -Format o)" | Set-Content $summary

foreach ($ck in $ckpts) {
    $path = Join-Path $RepoRoot $ck
    if (-not (Test-Path $path)) {
        "[MISS] $ck" | Add-Content $summary
        Write-Host "[WARN] missing $ck" -ForegroundColor Yellow
        continue
    }
    Write-Host "[NEW] passive species eval: $ck" -ForegroundColor Cyan
    $log = Join-Path $outDir ("species_" + ([IO.Path]::GetFileNameWithoutExtension($ck)) + ".log")
    try {
        $pyArgs = @(
            "scripts/eval_passive_species_anchors.py",
            "--checkpoint", $ck,
            "--device", $Device,
            "--split", "all"
        )
        if ($Only) { $pyArgs += @("--only", $Only) }
        Invoke-PythonRcCheck -Label "species $ck" -PyArgs $pyArgs 2>&1 | Tee-Object -FilePath $log
        Get-Content $log | Add-Content $summary
    } catch {
        "[ERR] $ck : $_" | Add-Content $summary
        Write-Host "[ERR] $ck failed" -ForegroundColor Red
    }
}

Write-Host "[OK] summary -> $summary" -ForegroundColor Green
