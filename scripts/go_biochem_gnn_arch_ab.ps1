# Retired launcher kept as a guardrail stub.
# The GNODE/Neural-ODE pushforward arch A/B was removed from the active stack.
# Use gate A/B tools instead:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_biochem_gnn_gate_ab.ps1

$ErrorActionPreference = "Stop"
Write-Host "[WARN] go_biochem_gnn_arch_ab.ps1 is retired." -ForegroundColor Yellow
Write-Host "[i] GNODE/Neural-ODE pushforward references were removed from active training." -ForegroundColor DarkGray
Write-Host "[i] Use scripts/go_biochem_gnn_gate_ab.ps1 for active deploy A/B checks." -ForegroundColor DarkGray
exit 1
