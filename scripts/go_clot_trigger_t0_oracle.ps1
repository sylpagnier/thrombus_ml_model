# Star 1 T0: GT flow + GT species physics trigger (deploy nucleation projection).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_trigger_t0_oracle.ps1"
#   powershell ... -Viz
#   powershell ... -VizOnly
#   powershell ... -Audit
#   powershell ... baseline fields: go_t0_physics_baseline.ps1 -EvalOracle
#   powershell ... -ShowRaw        # viz: add raw gelation debug row
#   powershell ... -OracleBand     # legacy debug only
#   powershell ... -OracleForward  # GT-seed forward (not deploy)

param(
    [string] $Out = "outputs/biochem/clot_trigger/t0_oracle.json",
    [string] $Val = "patient007",
    [string] $VizAnchor = "patient007",
    [string] $VizAnchor2 = "patient002",
    [switch] $PriorGate,
    [switch] $OracleBand,
    [switch] $OracleForward,
    [switch] $ShowRaw,
    [switch] $Audit,
    [switch] $Viz,
    [switch] $VizOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")

$env:CLOT_PHI_PHYSICS_MU_BASE = "comsol_carreau"
$env:CLOT_PHI_PHYSICS_GAMMA_MODE = "max"
$env:CLOT_PHI_PHYSICS_MU_RATIO_MAX = "4"
$env:PYTHONUNBUFFERED = "1"

if (-not $VizOnly) {
    Write-Host "[NEW] T0 physics trigger (pred nucleation deploy, support + full-mesh F1)" -ForegroundColor Cyan
    $pyArgs = @(
        "scripts/eval_clot_trigger_t0_oracle.py",
        "--out", $Out,
        "--val", $Val
    )
    if ($PriorGate) { $pyArgs += "--prior-gate" }
    if ($OracleBand) { $pyArgs += "--oracle-band" }
    if ($OracleForward) { $pyArgs += "--oracle-forward" }
    Invoke-PythonRcCheck -Label "t0 oracle" -PyArgs $pyArgs
    Write-Host "[OK] results -> $Out" -ForegroundColor Green
}

if ($Audit) {
    & (Join-Path $PSScriptRoot "go_clot_trigger_t0_audit.ps1")
}

if ($Viz -or $VizOnly) {
    $VizDir = "outputs/biochem/viz/clot_trigger"
    New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot $VizDir) | Out-Null
    foreach ($anc in @($VizAnchor, $VizAnchor2)) {
        if (-not $anc) { continue }
        Write-Host "[NEW] T0 viz $anc" -ForegroundColor Cyan
        $vizArgs = @(
            "scripts/viz_clot_trigger_t0_oracle.py",
            "--anchor", $anc,
            "--out", "$VizDir/t0_$anc.png"
        )
        if ($PriorGate) { $vizArgs += "--prior-gate" }
        if ($OracleBand) { $vizArgs += "--oracle-band" }
        if ($OracleForward) { $vizArgs += "--oracle-forward" }
        if ($ShowRaw) { $vizArgs += "--show-raw" }
        Invoke-PythonRcCheck -Label "t0 viz $anc" -PyArgs $vizArgs
    }
    Write-Host "[OK] viz -> $VizDir/t0_*.png" -ForegroundColor Green
}
