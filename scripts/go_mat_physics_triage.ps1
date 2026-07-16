# W-centric physics-channel triage (~6h GPU budget).
# Each leg = W (stagnation flow feats) + one targeted COMSOL mechanism channel.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_mat_physics_triage.ps1 -Fresh
#   powershell ... -IncludeControl   # also re-run W_mat_flow_stagnation baseline
#   powershell ... -Legs WA_mat_flow_neighbor_gate,WI_mat_flow_neighbor_geom

param(
    [switch] $Fresh,
    [switch] $IncludeControl,
    [string] $Legs = "",
    [int] $Epochs = 20,
    [int] $EarlyStop = 12,
    [int] $MaxWindows = 64
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

if ($Legs.Trim()) {
    $legList = @($Legs.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })
} else {
    $legList = @(
        "WA_mat_flow_neighbor_gate",
        "WB_mat_flow_geom_rich",
        "WC_mat_flow_dynamic",
        "WD_mat_flow_frontier",
        "WE_mat_flow_thrombin",
        "WF_mat_flow_fg",
        "WG_mat_flow_neighbor_crit",
        "WH_mat_flow_gelation_light",
        "WI_mat_flow_neighbor_geom",
        "WJ_mat_flow_stack"
    )
    if ($IncludeControl) {
        $legList = @("W_mat_flow_stagnation") + $legList
    }
}

$OutRoot = "outputs/biochem/biochem_gnn/mat_physics_triage"
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

Write-Host "[NEW] mat physics triage ($($legList.Count) legs, W + COMSOL channels)" -ForegroundColor Cyan
Write-Host "[i] $Epochs ep / ES $EarlyStop / max_windows $MaxWindows (~$([math]::Round($legList.Count * 28 / 60, 1))h est)" -ForegroundColor DarkGray

Invoke-PythonRcCheck -Label "mat physics triage pytest gate" -PyArgs @(
    "-m", "pytest",
    "src/tests/test_mat_growth_simple_scope.py",
    "src/tests/test_species_flow_feats.py",
    "-q"
)

foreach ($leg in $legList) {
    Write-Host "[leg] $leg" -ForegroundColor Cyan
    $legArgs = @(
        "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", (Join-Path $PSScriptRoot "go_mat_growth_simple.ps1"),
        "-Leg", $leg,
        "-Epochs", "$Epochs",
        "-EarlyStop", "$EarlyStop",
        "-MaxWindows", "$MaxWindows"
    )
    if ($Fresh) { $legArgs += "-Fresh" }
    & powershell @legArgs
    if ($LASTEXITCODE -ne 0) { throw "$leg failed (exit=$LASTEXITCODE)" }
}

Invoke-PythonRcCheck -Label "mat physics triage summary" -PyArgs @(
    "scripts/summarize_mat_only_full.py",
    "--legs", ($legList -join ","),
    "--out", "$OutRoot/mat_physics_triage_summary.json"
)

Write-Host "[OK] summary: $OutRoot/mat_physics_triage_summary.json" -ForegroundColor Green
