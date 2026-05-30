# Lock the passive-align teacher (biochem_teacher_last.pth from align probe) as canonical passive teacher.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_passive_lock_align_ckpt.ps1"
#   powershell ... -SourceCkpt outputs/biochem/biochem_teacher_last.pth -RunNote m3_align_transport_union

param(
    [string] $SourceCkpt = "outputs/biochem/biochem_teacher_last.pth",
    [string] $DestCkpt = "outputs/biochem/biochem_teacher_passive_align_locked.pth",
    [string] $RunNote = "passive_align_locked",
    [string] $ManifestPath = "outputs/biochem/passive_align_locked_manifest.json"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$src = Join-Path $RepoRoot $SourceCkpt
$dst = Join-Path $RepoRoot $DestCkpt
if (-not (Test-Path $src)) {
    Write-Host "[ERR] Missing source ckpt: $src" -ForegroundColor Red
    Write-Host "[i]  Finish align probe first (go_m3_align_probe.ps1) so biochem_teacher_last.pth exists." -ForegroundColor Yellow
    exit 1
}

$dstDir = Split-Path -Parent $dst
if (-not (Test-Path $dstDir)) {
    New-Item -ItemType Directory -Path $dstDir -Force | Out-Null
}

Copy-Item $src $dst -Force
# init-from-best path used by align / 20ep / bridge launchers
Copy-Item $dst (Join-Path $RepoRoot "outputs\biochem\biochem_teacher_best_high_mu.pth") -Force

$manifest = @{
    locked_at_utc   = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    run_note        = $RunNote
    source_ckpt     = $SourceCkpt.Replace("\", "/")
    dest_ckpt       = $DestCkpt.Replace("\", "/")
    source_bytes    = (Get-Item $src).Length
    dest_bytes      = (Get-Item $dst).Length
    also_copied_to  = "outputs/biochem/biochem_teacher_best_high_mu.pth"
}
$manifestDir = Split-Path -Parent (Join-Path $RepoRoot $ManifestPath)
if (-not (Test-Path $manifestDir)) {
    New-Item -ItemType Directory -Path $manifestDir -Force | Out-Null
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -Path (Join-Path $RepoRoot $ManifestPath) -Encoding utf8

$env:BIOCHEM_RUN_NOTE = $RunNote
Write-Host "[OK] Locked passive teacher:" -ForegroundColor Green
Write-Host "     $DestCkpt" -ForegroundColor Green
Write-Host "[i]  Manifest: $ManifestPath | BIOCHEM_RUN_NOTE=$RunNote" -ForegroundColor Cyan
Write-Host "[i]  Next: go_passive_align_20ep.ps1" -ForegroundColor Cyan
