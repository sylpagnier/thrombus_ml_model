# HemoRGP Customer Predict App
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_customer_predict.ps1
#   powershell ... -Cpu
#
# Put geometries in customer_geometries\ then Browse or pick Inbox in the GUI.

param(
    [switch] $Cpu
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Inbox = Join-Path $RepoRoot "customer_geometries"
if (-not (Test-Path $Inbox)) {
    New-Item -ItemType Directory -Path $Inbox | Out-Null
}

Write-Host "[i] HemoRGP Predict" -ForegroundColor Cyan
Write-Host "[i] Deploy: WC_v8 compound (WC_v7 wall + frontier-h1 growth)" -ForegroundColor DarkGray
Write-Host "[i] Wall-only fallback: `$env:CUSTOMER_COMPOUND='0'" -ForegroundColor DarkGray
Write-Host "[i] Geometries folder: $Inbox" -ForegroundColor DarkGray
Write-Host "[i] Use Open folder or Browse (starts in that folder)" -ForegroundColor DarkGray

$pyArgs = @("-u", "-m", "src.tools.customer_predict_app")
if ($Cpu) {
    $pyArgs += "--cpu"
    Write-Host "[WARN] CPU mode (slow). CUDA is recommended." -ForegroundColor Yellow
}

# Direct python (no Write-Host pipe) so the matplotlib GUI stays interactive.
& python @pyArgs
$rc = if ($null -ne $LASTEXITCODE) { [int]$LASTEXITCODE } else { 0 }
if ($rc -ne 0) {
    Write-Host "[ERR] customer_predict_app exited $rc" -ForegroundColor Red
    exit $rc
}
exit 0
