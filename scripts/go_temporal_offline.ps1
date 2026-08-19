<#
    Offline runner: temporal head only, on a parity-gated physics base.

    Start it and walk away -- no network, no supervision. Resumable: it skips any
    {flow}_seed{n}.json already on disk, so it is safe to kill and restart.

    Runs sequentially (standing constraint 5.5: NO CONCURRENT GPU JOBS -- epoch time went
    650s -> 1900s under contention on this 4 GB card).

      arm A  flow=gt    N seeds
      arm B  flow=pred  N seeds   -- honestly bandaid-free: the temporal head reads only
                                    (mat, mas, d_mat, sr) + edge_index, never data.x

    Each run refuses to train unless the differentiable base first matches the hard
    physics model on FIT+DEV (exit code 2). That gate is the thing whose absence
    invalidated the previous round.

    NOTE ON THE OUTPUT PLUMBING. Two Windows-PowerShell traps are deliberately avoided:
      * ``Tee-Object -FilePath`` holds an open handle for the life of the pipeline, which
        collides with the next iteration and throws "process cannot access the file";
      * a native command writing to stderr (torch prints a UserWarning there) is wrapped
        as an ErrorRecord and surfaces as ``NativeCommandError`` even on success.
    Both are fixed by stringifying every line with "$_" and appending with Add-Content,
    which opens and closes the file per line and never holds a lock.
#>
param(
    [int]$Seeds = 3,
    [int]$Epochs = 20,
    [int]$Patience = 6
)

$ErrorActionPreference = "Continue"
Set-Location (Split-Path -Parent $PSScriptRoot)
New-Item -ItemType Directory -Force -Path "outputs/temporal_only" | Out-Null
$log = "outputs/temporal_only/offline_run.log"

function Say($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Write-Host $line -ForegroundColor Cyan
    Add-Content -Path $log -Value $line -Encoding utf8
}

function Invoke-Logged($argList, $outFile) {
    if (Test-Path $outFile) { Remove-Item $outFile -Force -ErrorAction SilentlyContinue }
    # 2>&1 merges stderr; "$_" flattens ErrorRecords to plain strings so PowerShell does
    # not raise NativeCommandError on torch's stderr warnings.
    & python -u @argList 2>&1 | ForEach-Object {
        $line = "$_"
        Write-Host $line
        Add-Content -Path $outFile -Value $line -Encoding utf8
    }
    return $LASTEXITCODE
}

Say "=== temporal-only offline run starting (seeds=$Seeds epochs=$Epochs) ==="
$t0 = Get-Date

foreach ($flow in @("gt", "pred")) {
    for ($s = 0; $s -lt $Seeds; $s++) {
        $tag = "${flow}_seed${s}"
        if (Test-Path "outputs/temporal_only/$tag.json") {
            Say "skip $tag (already present)"
            continue
        }
        Say "running flow=$flow seed=$s"
        $code = Invoke-Logged @(
            "scripts/sweep_temporal_only.py", "--flow", $flow, "--seed", "$s",
            "--epochs", "$Epochs", "--patience", "$Patience", "--tag", $tag
        ) "outputs/temporal_only/${tag}.log"
        if ($code -eq 2) { Say "PARITY GATE FAILED for $tag -- no training done" }
        elseif ($code -ne 0) { Say "WARN: $tag exited $code (continuing)" }
        else { Say "done $tag" }
    }
}

Say "aggregating"
Invoke-Logged @("scripts/aggregate_temporal_only.py") "outputs/temporal_only/VERDICT.txt" | Out-Null
Say ("=== complete in {0:N0} min -- see outputs/temporal_only/VERDICT.txt ===" -f `
     ((Get-Date) - $t0).TotalMinutes)
