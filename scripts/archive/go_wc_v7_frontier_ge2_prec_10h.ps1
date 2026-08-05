# Compatibility shim: 10h recipe resized to 8h.
# Prefer: scripts/go_wc_v7_frontier_ge2_prec_8h.ps1
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    $Remaining
)
Write-Host "[i] Redirecting to go_wc_v7_frontier_ge2_prec_8h.ps1 (~8 h budget)" -ForegroundColor DarkGray
& (Join-Path $PSScriptRoot "go_wc_v7_frontier_ge2_prec_8h.ps1") @Remaining
