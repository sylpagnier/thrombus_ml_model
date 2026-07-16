# Hop-distance visualization for the dynamic occlusion (Pivot 3 / WC_canonical_v2) model.
# Compares the pivot3 ckpt against WC_mat_3hop baseline side-by-side,
# coloured by wall-hop distance, with a bar chart of clot nodes per hop bucket.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_viz_pivot3_hop_analysis.ps1
#   powershell ... -Leg WC_canonical_v2 -Patients patient003,patient007

param(
    [string] $Leg        = "WC_pivot3_occlusion",
    [string] $CompareLeg = "WC_mat_3hop",
    [string] $Patients   = "patient007",
    [int]    $MaxHop     = 5,
    [string] $Flow       = "kinematics"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

$patientList = @($Patients.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$OutRoot = "outputs/biochem/viz/mat_growth"

Write-Host "[i] Pivot-3 hop-distance viz  leg=$Leg  compare=$CompareLeg  patients=$Patients" -ForegroundColor Cyan

foreach ($patient in $patientList) {
    $outPng = "$OutRoot/pivot3_hop_analysis_${Leg}_${patient}.png"
    Write-Host "[viz] $Leg vs $CompareLeg  anchor=$patient -> $outPng" -ForegroundColor DarkGray
    $pyArgs = @(
        "scripts/viz_pivot3_hop_analysis.py",
        "--leg",         $Leg,
        "--compare-leg", $CompareLeg,
        "--anchor",      $patient,
        "--max-hop",     $MaxHop,
        "--flow",        $Flow,
        "--out",         $outPng
    )
    Invoke-PythonRcCheck -Label "viz pivot3 hop $patient" -PyArgs $pyArgs
    Write-Host "[OK] saved $outPng" -ForegroundColor Green
}

Write-Host "[OK] hop viz done under $OutRoot" -ForegroundColor Green
