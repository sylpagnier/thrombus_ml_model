# Lock I.3 XY learn-leg teacher as canonical step-2 / M5 init.
#
# Prereq: go_passive_xy_block_pass.ps1 (XY2-learn) -> biochem_teacher_last.pth
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_passive_lock_xy_ckpt.ps1"
#   powershell ... -SourceCkpt outputs/biochem/biochem_teacher_last.pth -RunNote passive_step2_bridge_align_learn

param(
    [string] $SourceCkpt = "outputs/biochem/biochem_teacher_last.pth",
    [string] $DestCkpt = "outputs/biochem/biochem_teacher_passive_xy_locked.pth",
    [string] $RunNote = "passive_step2_bridge_align_learn",
    [string] $ManifestPath = "outputs/biochem/passive_xy_locked_manifest.json"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

& (Join-Path $PSScriptRoot "go_passive_lock_align_ckpt.ps1") `
    -SourceCkpt $SourceCkpt `
    -DestCkpt $DestCkpt `
    -RunNote $RunNote `
    -ManifestPath $ManifestPath
exit $LASTEXITCODE
