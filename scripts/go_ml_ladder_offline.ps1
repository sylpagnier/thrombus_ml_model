<#
    Offline runner: the controlled MeshGraphNet + cGNODE ladder.

    Start it and walk away. Resumable -- skips any {flow}_seed{n}.json already on disk.
    Sequential (standing constraint 5.5: NO CONCURRENT GPU JOBS on this 4 GB card).

    Each run trains four rungs (base / +MGN / +cGNODE / +both) on identical splits and
    seeds, refuses to proceed unless the base first matches the hard physics model on
    FIT+DEV (exit code 2), and opens SEALED exactly once.

    Output plumbing avoids two Windows-PowerShell traps: Tee-Object holds a file handle
    for the life of the pipeline (collides with the next iteration), and a native command
    writing to stderr surfaces as NativeCommandError even on success. Both are handled by
    stringifying each line with "$_" and appending via Add-Content.

    Usage:  powershell -ExecutionPolicy Bypass -File scripts\go_ml_ladder_offline.ps1
            powershell -ExecutionPolicy Bypass -File scripts\go_ml_ladder_offline.ps1 -Seeds 3
#>
param(
    [int]$Seeds = 3,
    [int]$Epochs = 16
)

$ErrorActionPreference = "Continue"
Set-Location (Split-Path -Parent $PSScriptRoot)
New-Item -ItemType Directory -Force -Path "outputs/ml_ladder" | Out-Null
$log = "outputs/ml_ladder/offline_run.log"

function Say($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Write-Host $line -ForegroundColor Cyan
    Add-Content -Path $log -Value $line -Encoding utf8
}

function Invoke-Logged($argList, $outFile) {
    if (Test-Path $outFile) { Remove-Item $outFile -Force -ErrorAction SilentlyContinue }
    # -u: unbuffered, so the log streams instead of appearing only at exit
    & python -u @argList 2>&1 | ForEach-Object {
        $line = "$_"
        Write-Host $line
        Add-Content -Path $outFile -Value $line -Encoding utf8
    }
    return $LASTEXITCODE
}

Say "=== ML ladder offline run (seeds=$Seeds epochs=$Epochs) ==="
$t0 = Get-Date

foreach ($flow in @("gt", "pred")) {
    for ($s = 0; $s -lt $Seeds; $s++) {
        $tag = "${flow}_seed${s}"
        if (Test-Path "outputs/ml_ladder/$tag.json") { Say "skip $tag"; continue }
        Say "running flow=$flow seed=$s"
        $code = Invoke-Logged @(
            "scripts/train_ml_ladder.py", "--flow", $flow, "--seed", "$s",
            "--epochs", "$Epochs", "--tag", $tag
        ) "outputs/ml_ladder/${tag}.log"
        if ($code -eq 2) { Say "PARITY GATE FAILED for $tag" }
        elseif ($code -ne 0) { Say "WARN: $tag exited $code (continuing)" }
        else { Say "done $tag" }
    }
}

Say "aggregating"
Invoke-Logged @("scripts/aggregate_ml_ladder.py") "outputs/ml_ladder/VERDICT.txt" | Out-Null
Say ("=== complete in {0:N0} min -- see outputs/ml_ladder/VERDICT.txt ===" -f `
     ((Get-Date) - $t0).TotalMinutes)
