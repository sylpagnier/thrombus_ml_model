# Wait for in-flight round2, finish promote/threshold, then run round3 (~4h).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_gt_flow_chain_r2finish_r3_4h.ps1"
#   ... -Round2Pid 22148

param(
    [int] $Round2Pid = 0,
    [int] $PollSec = 120,
    [switch] $SkipRound2Finish,
    [switch] $SkipRound3
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if ($Round2Pid -gt 0) {
    Write-Host "[i] waiting for round2 pid=$Round2Pid ..." -ForegroundColor Cyan
    while (Get-Process -Id $Round2Pid -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds $PollSec
    }
    Write-Host "[OK] round2 process ended" -ForegroundColor Green
} else {
  # Heuristic: wait until round2 log shows promote or teacher_refresh completes
    $log = Join-Path $RepoRoot "outputs\biochem\gt_flow_round2_4h\round2_log.jsonl"
    if (Test-Path $log) {
        Write-Host "[i] waiting for round2 promote in log (poll ${PollSec}s) ..." -ForegroundColor Cyan
        for ($i = 0; $i -lt 120; $i++) {
            $tail = Get-Content $log -Tail 5 -ErrorAction SilentlyContinue | Out-String
            if ($tail -match '"step":"promote"' -or $tail -match 'round2 complete') {
                break
            }
            Start-Sleep -Seconds $PollSec
        }
    }
}

if (-not $SkipRound2Finish) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "go_gt_flow_finish_round2.ps1")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (-not $SkipRound3) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "go_gt_flow_round3_4h.ps1")
    exit $LASTEXITCODE
}

Write-Host "[OK] chain done" -ForegroundColor Green
