# S0 + S1 rule ladder only (multistep-winner env).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_deploy_ladder_s0s1.ps1"

param(
    [string] $AnchorDir = "data/processed/graphs_biochem_anchors",
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

$rungs = @("s0", "s1")
$passed = @()
$failed = @()

foreach ($rung in $rungs) {
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
$statusPath = Join-Path $RepoRoot "outputs/biochem/diagnostics/clot_deploy_ladder_status.json"
$existing = @{ passed = @(); failed = @(); rule = $ruleDesc }
if (Test-Path $statusPath) {
    try {
        $prev = Get-Content $statusPath -Raw | ConvertFrom-Json
        if ($prev.passed) { $existing.passed = @($prev.passed) }
        if ($prev.failed) { $existing.failed = @($prev.failed) }
        if ($prev.rule) { $existing.rule = $prev.rule }
    } catch { }
}
foreach ($p in $passed) {
    if ($existing.passed -notcontains $p) { $existing.passed += $p }
}
foreach ($f in $failed) {
    if ($existing.failed -notcontains $f) { $existing.failed += $f }
}
$existing.passed = @($existing.passed | Where-Object { $existing.failed -notcontains $_ })
$status = @{
    passed = @($existing.passed)
    failed = @($existing.failed)
    rungs  = $rungs
    rule   = $ruleDesc
    ts     = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
} | ConvertTo-Json
New-Item -ItemType Directory -Force -Path (Split-Path $statusPath) | Out-Null
Set-Content -Path $statusPath -Value $status -Encoding UTF8
Write-Host "[save] $statusPath" -ForegroundColor DarkGray

if (-not $SkipViz) {
    Write-Host ""
    Write-Host "[NEW] rule timeline viz (S1 from_t0 keyframes)" -ForegroundColor Cyan
    & (Join-Path $PSScriptRoot "go_clot_deploy_rule_timeline_viz.ps1") -Stage s1 -Anchor patient007 -Keyframes 8
}

if ($failed.Count -gt 0) { exit 1 }
