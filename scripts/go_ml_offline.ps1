<#
    Offline runner for the ML-corrector v2 evaluation. Start it and walk away -- it needs
    no network and no supervision, and it prints a verdict table at the end.

    It waits for the v1 seed sweep to finish first: standing constraint 5.5 is NO
    CONCURRENT GPU JOBS (epoch time went 650s -> 1900s under contention on this 4 GB card).

      arm A  flow=gt    3 seeds   -- comparable to the physics bar of 0.9093 on sealed
      arm B  flow=pred  3 seeds   -- deployable; data.x priors rebuilt from u0_pred, so
                                     this is the first arm-B number that is honestly
                                     bandaid-free (docs/PHASE3_HANDOFF.md 0a, corrected)

    Usage:   powershell -ExecutionPolicy Bypass -File scripts\go_ml_offline.ps1
             powershell -ExecutionPolicy Bypass -File scripts\go_ml_offline.ps1 -Seeds 5
#>
param(
    [int]$Seeds = 3,
    [int]$Epochs = 12,
    [int]$Patience = 4,
    [switch]$SkipWait
)

$ErrorActionPreference = "Continue"
Set-Location (Split-Path -Parent $PSScriptRoot)
$log = "outputs/ml_v2/offline_run.log"
New-Item -ItemType Directory -Force -Path "outputs/ml_v2" | Out-Null
if (-not (Test-Path $log)) { New-Item -ItemType File -Path $log | Out-Null }

function Say($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Write-Host $line
    Add-Content -Path $log -Value $line
}

Say "=== offline ML v2 run starting ==="

# --- wait for the v1 seed sweep so the GPU is not shared -----------------------
if (-not $SkipWait) {
    $sentinel = "outputs/ml_clean_protocol/seed3.json"
    if (-not (Test-Path $sentinel)) {
        Say "waiting for the v1 seed sweep to finish ($sentinel) ..."
        $waited = 0
        while ((-not (Test-Path $sentinel)) -and ($waited -lt 10800)) {
            Start-Sleep -Seconds 60
            $waited += 60
        }
        if (Test-Path $sentinel) { Say "v1 sweep done after ${waited}s" }
        else { Say "WARN: gave up waiting after ${waited}s; proceeding anyway" }
    } else {
        Say "v1 sweep already complete"
    }
}

# --- arm A then arm B ----------------------------------------------------------
foreach ($flow in @("gt", "pred")) {
    for ($s = 0; $s -lt $Seeds; $s++) {
        $tag = "${flow}_seed${s}"
        if (Test-Path "outputs/ml_v2/$tag.json") { Say "skip $tag (already present)"; continue }
        Say "running v2  flow=$flow seed=$s"
        $out = "outputs/ml_v2/${tag}.log"
        python scripts/sweep_ml_v2.py --flow $flow --seed $s --epochs $Epochs `
            --patience $Patience --tag $tag *>&1 | Tee-Object -FilePath $out
        if ($LASTEXITCODE -ne 0) { Say "WARN: $tag exited $LASTEXITCODE (continuing)" }
        else { Say "done $tag" }
    }
}

# --- verdict -------------------------------------------------------------------
Say "aggregating"
python scripts/aggregate_ml_results.py *>&1 | Tee-Object -FilePath "outputs/ml_v2/VERDICT.txt"
Say "=== complete -- see outputs/ml_v2/VERDICT.txt ==="
