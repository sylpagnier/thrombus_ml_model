# Compatibility stub: 6h A/B launcher redirected to 9h A/B/C.
# Prefer: scripts/go_wc_v7_compound_growth_abc_9h.ps1
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    $Remaining
)
$ErrorActionPreference = "Stop"
Write-Host "[i] Redirecting to go_wc_v7_compound_growth_abc_9h.ps1 (A/B/C; full pipeline ~20-26 h)" -ForegroundColor DarkGray
& (Join-Path $PSScriptRoot "go_wc_v7_compound_growth_abc_9h.ps1") @Remaining
exit $LASTEXITCODE
