# Off-wall sweep v2 hop-distance visualization.
# Compares the sweep v2 legs against the baseline.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_viz_offwall_v2.ps1
#

param(
    [string[]] $Legs = @("WC_v2_baseline", "WC_v2_convection", "WC_v2_longrange", "WC_v2_label_smooth", "WC_v2_dilation", "WC_v2_longrange_smooth"),
    [string] $CompareLeg = "WC_v2_baseline",
    [string] $Patients = "patient007",
    [int] $MaxHop = 5,
    [string] $Flow = "kinematics"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

# Split legs if passed as a single comma-separated string
$legList = @()
foreach ($l in $Legs) {
    if ($l.Contains(",")) {
        $legList += $l.Split(",") | ForEach-Object { $_.Trim() }
    } else {
        $legList += $l.Trim()
    }
}
$Legs = @($legList | Where-Object { $_ })

$patientList = @($Patients.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$OutRoot = "outputs/biochem/viz/mat_growth"

Write-Host "[i] Off-wall v2 sweep hop viz legs=$($Legs -join ',') compare=$CompareLeg patients=$Patients" -ForegroundColor Cyan

foreach ($leg in $Legs) {
    if ($leg -eq $CompareLeg) {
        continue
    }
    foreach ($patient in $patientList) {
        $outPng = "$OutRoot/pivot3_hop_analysis_${leg}_${patient}.png"
        Write-Host "[viz] $leg vs $CompareLeg anchor=$patient -> $outPng" -ForegroundColor DarkGray
        $pyArgs = @(
            "scripts/viz_pivot3_hop_analysis.py",
            "--leg", $leg,
            "--compare-leg", $CompareLeg,
            "--anchor", $patient,
            "--max-hop", $MaxHop,
            "--flow", $Flow,
            "--out", $outPng
        )
        Invoke-PythonRcCheck -Label "viz offwall v2 $leg $patient" -PyArgs $pyArgs
        Write-Host "[OK] saved $outPng" -ForegroundColor Green
    }
}

Write-Host "[OK] off-wall v2 hop viz done under $OutRoot" -ForegroundColor Green
