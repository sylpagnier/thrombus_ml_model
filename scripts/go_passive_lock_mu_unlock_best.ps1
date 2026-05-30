# Lock all-truth-best weights from mu-unlock probe for finetune / bridge init.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_passive_lock_mu_unlock_best.ps1"
#   powershell ... -SourceCkpt outputs/biochem/biochem_teacher_passive_mu_unlock_best.pth

param(
    [string] $SourceCkpt = "outputs/biochem/biochem_teacher_passive_mu_unlock_best.pth",
    [string] $DestCkpt = "outputs/biochem/biochem_teacher_passive_mu_unlock_best.pth",
    [string] $RunNote = "passive_mu_unlock_best_locked",
    [string] $ManifestPath = "outputs/biochem/passive_mu_unlock_best_manifest.json",
    [switch] $UseLastFallback
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")

$src = Join-Path $RepoRoot $SourceCkpt
if (-not (Test-Path $src)) {
    if ($UseLastFallback) {
        $fallback = Join-Path $RepoRoot "outputs/biochem/biochem_teacher_last.pth"
        if (Test-Path $fallback) {
            Write-Host "[WARN] Missing $SourceCkpt; using last.pth (may not be all-truth-best epoch)" -ForegroundColor Yellow
            $src = $fallback
        }
    }
}
if (-not (Test-Path $src)) {
    Write-Host "[ERR] Missing source ckpt: $src" -ForegroundColor Red
    Write-Host "[i]  Re-run go_passive_mu_unlock_probe.ps1 (saves best each improvement) or pass -UseLastFallback" -ForegroundColor Yellow
    exit 1
}

$dst = Join-Path $RepoRoot $DestCkpt
$dstDir = Split-Path -Parent $dst
if (-not (Test-Path $dstDir)) {
    New-Item -ItemType Directory -Path $dstDir -Force | Out-Null
}
if ($src -ne $dst) {
    Copy-Item $src $dst -Force
}
Copy-Item $dst (Join-Path $RepoRoot "outputs\biochem\biochem_teacher_best_high_mu.pth") -Force

$manifest = @{
    locked_at_utc  = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    run_note       = $RunNote
    source_ckpt    = $SourceCkpt.Replace("\", "/")
    dest_ckpt      = $DestCkpt.Replace("\", "/")
    source_bytes   = (Get-Item $src).Length
    dest_bytes     = (Get-Item $dst).Length
    also_copied_to = "outputs/biochem/biochem_teacher_best_high_mu.pth"
}
$manifestDir = Split-Path -Parent (Join-Path $RepoRoot $ManifestPath)
if (-not (Test-Path $manifestDir)) {
    New-Item -ItemType Directory -Path $manifestDir -Force | Out-Null
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -Path (Join-Path $RepoRoot $ManifestPath) -Encoding utf8

Write-Host "[OK] Locked passive mu-unlock best:" -ForegroundColor Green
Write-Host "     $DestCkpt" -ForegroundColor Green
Write-Host "[i]  Next: go_passive_mu_unlock_finetune.ps1" -ForegroundColor Cyan
exit 0
