# Phase 0.2 — pred-kine dump from capped COMSOL anchors (7950 s, full T).
# Run on a second machine / overnight; does NOT block S0-G2 on graphs_biochem_anchors.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_deploy_dump_comsol.ps1"
#   powershell ... -Only "patient001,patient002,patient003,patient004,patient006,patient007"

param(
    [string] $Teacher = "",
    [string] $SrcDir = "data/processed/graphs_biochem_anchors",
    [string] $OutDir = "outputs/biochem/gnode10_sweep/anchors_gnode12_predkine_uvp",
    [string] $Only = "patient001,patient002,patient003,patient004,patient006,patient007",
    [double] $MuRatioMax = 20,
    [switch] $SkipForce
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")
. (Join-Path $PSScriptRoot "_gnode12_env.ps1")

$teacherPath = if ($Teacher) { $Teacher } else { Resolve-Gnode12TeacherCkpt -UserPath "" }
if (-not $teacherPath -or -not (Test-Path $teacherPath)) {
    Write-Host "[ERR] Teacher ckpt missing. Pass -Teacher or run go_gnode11_finish.ps1" -ForegroundColor Red
    exit 1
}

$srcFull = Join-Path $RepoRoot $SrcDir
if (-not (Test-Path $srcFull)) {
    Write-Host "[ERR] Missing COMSOL anchors: $SrcDir" -ForegroundColor Red
    exit 1
}

$ckptRel = Get-Gnode12PathRelativeToRepo -FullPath $teacherPath -RepoRoot $RepoRoot
Set-Gnode12DumpRolloutEnv -MuRatioMax "$MuRatioMax"

$pyArgs = @(
    "scripts/dump_teacher_species_to_anchors.py",
    "--teacher", $ckptRel,
    "--src-dir", ($SrcDir -replace '\\', '/'),
    "--out-dir", ($OutDir -replace '\\', '/'),
    "--device", "cuda",
    "--no-subsample",
    "--write-kine-macro",
    "--mu-ratio-max", "$MuRatioMax"
)
if ($Only) { $pyArgs += @("--only", $Only) }
if (-not $SkipForce) { $pyArgs += "--force" }

Write-Host ""
Write-Host "[NEW] CAVO dump Track B (COMSOL 7950s, pred kine)" -ForegroundColor Cyan
Write-Host "[i]  teacher=$ckptRel" -ForegroundColor DarkGray
Write-Host "[i]  src=$SrcDir out=$OutDir only=$Only" -ForegroundColor DarkGray
Write-Host "[i]  expect T~54 per patient, t_max~7950s (NOT gnode_8h_ladder 30ks cache)" -ForegroundColor DarkGray

Invoke-PythonRcCheck -Label "deploy dump comsol" -PyArgs $pyArgs

Write-Host "[OK]  dump -> $OutDir" -ForegroundColor Green
