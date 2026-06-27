# Fast precision-first architecture triage (~3-6h for 6-8 legs).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_mat_arch_triage.ps1 -Fresh
#   powershell ... -Legs U_mat_frontier_only,S_mat_frontier_nuc

param(
    [switch] $Fresh,
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
        "U_mat_frontier_only",
        "Y_mat_tight_seed",
        "W_mat_flow_stagnation",
        "X_mat_flow_seedfront",
        "V_mat_frontier_geom",
        "S_mat_frontier_nuc",
        "T_mat_frontier_sharp",
        "AB_mat_gelation_aux"
    )
}

$OutRoot = "outputs/biochem/biochem_gnn/mat_arch_triage"
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

Write-Host "[NEW] mat arch triage ($($legList.Count) legs, precision-first recipe)" -ForegroundColor Cyan
Write-Host "[i] $Epochs ep / ES $EarlyStop / max_windows $MaxWindows" -ForegroundColor DarkGray

Invoke-PythonRcCheck -Label "mat arch triage pytest gate" -PyArgs @(
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

Invoke-PythonRcCheck -Label "mat arch triage summary" -PyArgs @(
    "scripts/summarize_mat_only_full.py",
    "--legs", ($legList -join ","),
    "--out", "$OutRoot/mat_arch_triage_summary.json"
)

Write-Host "[OK] summary: $OutRoot/mat_arch_triage_summary.json" -ForegroundColor Green
