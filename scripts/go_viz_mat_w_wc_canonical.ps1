# Viz W vs WC clot ladder on anchor patients (deploy-faithful rollout).
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_viz_mat_w_wc_canonical.ps1
#   powershell ... -File .\scripts\go_viz_mat_w_wc_canonical.ps1 -Patients patient003,patient007

param(
    [string] $Patients = "patient003,patient007",
    [string] $Legs = "W_mat_flow_stagnation,WC_mat_flow_dynamic"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

$legList = @($Legs.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$patientList = @($Patients.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$OutRoot = "outputs/biochem/viz/mat_growth"

Write-Host "[NEW] mat W vs WC ladder viz ($($legList.Count) legs x $($patientList.Count) patients)" -ForegroundColor Cyan

foreach ($leg in $legList) {
    foreach ($patient in $patientList) {
        $outPng = "$OutRoot/clot_ladder_${leg}_${patient}.png"
        Write-Host "[viz] $leg $patient -> $outPng" -ForegroundColor DarkGray
        Invoke-PythonRcCheck -Label "viz $leg $patient" -PyArgs @(
            "scripts/viz_mat_growth_clot_ladder.py",
            "--leg", $leg,
            "--anchor", $patient,
            "--out", $outPng
        )
    }
}

Write-Host "[OK] viz done under $OutRoot" -ForegroundColor Green
