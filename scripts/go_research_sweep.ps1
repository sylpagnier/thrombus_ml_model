# Geometry-sensitivity research sweeps (locked canonical biochem model).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_research_sweep.ps1 -List
#   powershell ... -File .\scripts\go_research_sweep.ps1 -Sweep 01_stenosis_strength
#   powershell ... -File .\scripts\go_research_sweep.ps1 -All
#   powershell ... -File .\scripts\go_research_sweep.ps1 -Sweep 03_inlet_re -Arm re_450
#   powershell ... -File .\scripts\go_research_sweep.ps1 -Sweep 01_stenosis_strength -Cpu
#
# Model resolution: CustomerDeployPipeline defaults ->
#   outputs/biochem/biochem_gnn/locked/species_gnn_best.pth + WC_v7_clot_phi_mse
# (whatever is currently promoted). Override with -WallCkpt / -MatLeg if needed.

param(
    [string] $Sweep = "",
    [switch] $All,
    [switch] $List,
    [string] $Arm = "",
    [switch] $ForceRebuildMesh,
    [switch] $Cpu,
    [string] $WallCkpt = "",
    [string] $MatLeg = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

Write-Host "[i] Research geometry-sensitivity sweeps" -ForegroundColor Cyan
Write-Host "[i] Model resolver: locked_canonical (promoted biochem baseline at run time)" -ForegroundColor DarkGray

$pyArgs = @("scripts/run_research_sweep.py")
if ($List) {
    $pyArgs += "--list"
}
elseif ($All) {
    $pyArgs += "--all"
}
elseif ($Sweep.Trim()) {
    $pyArgs += @("--sweep", $Sweep.Trim())
}
else {
    Write-Host "[ERR] Pass -Sweep <id>, -All, or -List" -ForegroundColor Red
    Write-Host "[i] Example: -Sweep 01_stenosis_strength" -ForegroundColor DarkGray
    exit 2
}

if ($Arm.Trim()) { $pyArgs += @("--arm", $Arm.Trim()) }
if ($ForceRebuildMesh) { $pyArgs += "--force-rebuild-mesh" }
if ($Cpu) {
    $pyArgs += "--cpu"
    Write-Host "[WARN] CPU mode (slow). CUDA is recommended." -ForegroundColor Yellow
}
if ($WallCkpt.Trim()) { $pyArgs += @("--wall-ckpt", $WallCkpt.Trim()) }
if ($MatLeg.Trim()) { $pyArgs += @("--mat-leg", $MatLeg.Trim()) }

# Direct python for long rollouts (avoid Write-Host pipe buffering).
& python -u @pyArgs
$rc = if ($null -ne $LASTEXITCODE) { [int]$LASTEXITCODE } else { 0 }
if ($rc -ne 0) {
    Write-Host "[ERR] run_research_sweep exited $rc" -ForegroundColor Red
    exit $rc
}
Write-Host "[OK] Done" -ForegroundColor Green
exit 0
