# Interactive COMSOL biochem export -> graphs (pick stem, skip if already extracted).
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_extract_biochem_interactive.ps1
#   powershell -File .\scripts\go_extract_biochem_interactive.ps1 -ListOnly

param(
    [string]$Stem = "",
    [switch]$Force,
    [switch]$ListOnly,
    [switch]$NoFromComsol
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$pyArgs = @("-m", "src.tools.extract_biochem_comsol")
if ($Stem) { $pyArgs += @("--stem", $Stem) }
if ($Force) { $pyArgs += "--force" }
if ($ListOnly) { $pyArgs += "--list-only" }
if ($NoFromComsol) { $pyArgs += "--no-from-comsol" }

Write-Host "[NEW] Biochem COMSOL interactive extract (meshes + cfd_results_biochem -> graphs)" -ForegroundColor Cyan
python @pyArgs
