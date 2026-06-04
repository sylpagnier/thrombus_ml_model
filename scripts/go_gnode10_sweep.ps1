# GNODE ladder rung 10: predicted-kine sweep (probe -> semi -> final dump+clot-phi).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_gnode10_sweep.ps1 -Fresh
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_gnode10_sweep.ps1 -Resume
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_gnode10_sweep.ps1 -DryRun
#
# Probe-only quick matrix (4ep, skip semi/final):
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_gnode10_sweep.ps1 -SkipFinal -ProbeEpochs 4

param(
    [switch] $Fresh,
    [switch] $Resume,
    [switch] $DryRun,
    [switch] $SkipFinal,
    [string] $InitCkpt = "",
    [int] $ProbeEpochs = 4,
    [int] $SemiEpochs = 8,
    [int] $FinalEpochs = 12,
    [int] $TopN = 3,
    [int] $ClotEpochs = 35,
    [int] $DumpStride = 72
)

$ErrorActionPreference = "Stop"
$args = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $PSScriptRoot "run_gnode10_sweep.ps1")
)
if ($Fresh) { $args += "-Fresh" }
if ($Resume) { $args += "-Resume" }
if ($DryRun) { $args += "-DryRun" }
if ($SkipFinal) { $args += "-SkipFinal" }
if ($InitCkpt) { $args += @("-InitCkpt", $InitCkpt) }
$args += @(
    "-ProbeEpochs", "$ProbeEpochs",
    "-SemiEpochs", "$SemiEpochs",
    "-FinalEpochs", "$FinalEpochs",
    "-TopN", "$TopN",
    "-ClotEpochs", "$ClotEpochs",
    "-DumpStride", "$DumpStride"
)
& powershell @args
exit $LASTEXITCODE
