# CAVO deploy ladder: eval + gate S0 -> S1 -> G1 -> G2 (rule baseline path).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_deploy_ladder.ps1"
#   powershell ... -Rungs "s0+s1" -StopOnFail

param(
    [string] $AnchorDir = "data/processed/graphs_biochem_anchors",
    [string] $Rungs = "",
    [switch] $OnlyS0S1,
    [switch] $StopOnFail,
    [switch] $SkipViz
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")
. (Join-Path $PSScriptRoot "_clot_deploy_binary_base.ps1")
. (Join-Path $PSScriptRoot "_clot_prior_rule_winner_env.ps1")
$env:BIOCHEM_PRIOR_COMSOL_ALIGNED = "1"
$env:BIOCHEM_PRIOR_NORM_MASK = "adjacent"

if ($OnlyS0S1) {
    $rungs = @("s0", "s1")
} elseif ($Rungs -and $Rungs.Trim()) {
    $rungs = @(
        ($Rungs -split "\+|,|\s+") | ForEach-Object { $_.Trim().ToLower() } | Where-Object { $_ }
    )
} else {
    $rungs = @("s0", "s1", "g1", "g2")
}
if ($rungs.Count -lt 1) {
    Write-Host "[ERR] No rungs in -Rungs '$Rungs'" -ForegroundColor Red
    exit 1
}

$passed = @()
$failed = @()

foreach ($rung in $rungs) {
    if ($rung -notin @("s0", "s1", "g1", "g2")) {
        Write-Host "[WARN] skip unknown rung '$rung'" -ForegroundColor Yellow
        continue
    }
    Write-Host ""
    Write-Host "[NEW] ladder rung $($rung.ToUpper()) eval + gate" -ForegroundColor Cyan
    $evalArgs = @(
        "scripts/eval_clot_prior_rule_ladder.py",
        "--stage", $rung,
        "--anchor-dir", $AnchorDir
    )
    Invoke-PythonRcCheck @evalArgs -Label "rule ladder $rung"

    $gateArgs = @(
        "scripts/check_clot_deploy_rung.py",
        "--rung", $rung,
        "--mode", "rule"
    )
    $gateRc = Invoke-PythonRc @gateArgs
    if ($gateRc -eq 0) {
        Write-Host "[OK]  rung $($rung.ToUpper()) PASS" -ForegroundColor Green
        $passed += $rung.ToUpper()
    } else {
        Write-Host "[FAIL] rung $($rung.ToUpper()) did not pass gate" -ForegroundColor Red
        $failed += $rung.ToUpper()
        if ($StopOnFail) {
            break
        }
    }
}

Write-Host ""
Write-Host "[i]  ladder summary passed=$($passed -join ',') failed=$($failed -join ',')" -ForegroundColor DarkGray

$ruleDesc = "prior_p0.80|flux_stag_top20|tie|skip_inlet_q25"
if ($env:CLOT_PHI_PRIOR_RULE_RANK_SDF_MAX) {
    $ruleDesc += "|sdf<=$($env:CLOT_PHI_PRIOR_RULE_RANK_SDF_MAX)"
}

$statusPath = Join-Path $RepoRoot "outputs/biochem/diagnostics/clot_deploy_ladder_status.json"
$status = @{
    passed = $passed
    failed = $failed
    rungs  = $rungs
    rule   = $ruleDesc
    ts     = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
} | ConvertTo-Json
New-Item -ItemType Directory -Force -Path (Split-Path $statusPath) | Out-Null
Set-Content -Path $statusPath -Value $status -Encoding UTF8
Write-Host "[save] $statusPath" -ForegroundColor DarkGray

if (-not $SkipViz -and ($rungs -contains "s1")) {
    Write-Host ""
    Write-Host "[NEW] rule timeline viz (S1 from_t0 keyframes)" -ForegroundColor Cyan
    & (Join-Path $PSScriptRoot "go_clot_deploy_rule_timeline_viz.ps1") -Stage s1 -Anchor patient007 -Keyframes 8
}

if ($failed.Count -gt 0) { exit 1 }
