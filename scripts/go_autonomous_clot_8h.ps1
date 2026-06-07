# 8-hour autonomous clot deploy: 4h Lane A dump track + 4h forecast ladder track.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_autonomous_clot_8h.ps1"
#   powershell ... -LaneHours 3 -Hours 6
#   powershell ... -LaneOnly

param(
    [double] $Hours = 8.0,
    [double] $LaneHours = 4.0,
    [string] $OutDir = "outputs/biochem/autonomy_clot_8h",
    [switch] $LaneOnly,
    [switch] $ForecastOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runOut = Join-Path $OutDir "run_$stamp"
New-Item -ItemType Directory -Force -Path $runOut | Out-Null

$logFile = Join-Path $runOut "autonomy_console.log"
Write-Host "[NEW] 8h autonomous clot deploy -> $runOut" -ForegroundColor Cyan
Write-Host "[i]  lane=${LaneHours}h forecast=$([math]::Max(0, $Hours - $LaneHours))h" -ForegroundColor DarkGray

$pyArgs = @(
    "scripts/autonomous_clot_8h.py",
    "--hours", "$Hours",
    "--lane-hours", "$LaneHours",
    "--out", ($runOut -replace '\\', '/')
)
if ($LaneOnly) { $pyArgs += "--lane-only" }
if ($ForecastOnly) { $pyArgs += "--forecast-only" }

# Tee console to log (no Write-Host pipe on python - tqdm safe via direct redirect in py)
python @pyArgs 2>&1 | Tee-Object -FilePath $logFile
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Symlink latest
$latest = Join-Path $OutDir "latest"
if (Test-Path $latest) { Remove-Item -Force $latest }
New-Item -ItemType SymbolicLink -Path $latest -Target $runOut -ErrorAction SilentlyContinue | Out-Null
if (-not (Test-Path $latest)) {
    Copy-Item -Force (Join-Path $runOut "autonomy_report.jsonl") (Join-Path $OutDir "latest_report.jsonl") -ErrorAction SilentlyContinue
}

Write-Host "[OK]  autonomy complete -> $runOut/autonomy_report.jsonl" -ForegroundColor Green
