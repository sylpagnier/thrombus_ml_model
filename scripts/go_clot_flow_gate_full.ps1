# FULL fresh clot-flow gate: wipe prior state -> train a fresh flow-aware teacher (leashed +
# dynamic by default) -> run EVERY ladder rung -> print the consolidated summary. One command,
# definitive, self-contained. Each stage runs in its own process so no gate env leaks between rungs.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_clot_flow_gate_full.ps1
#
#   # variants:
#   powershell ... -Smoke                 # 8-ep teacher, fast end-to-end smoke of the whole pipe
#   powershell ... -Static                # static-flow teacher (skips the #5c dynamic rung)
#   powershell ... -LatentDropout 0.3     # gentler leash

param(
    [double] $LatentDropout = 0.5,   # latent leash prob (0 = off)
    [switch] $Static,                # static-flow teacher (skip Trap C dynamic + #5c rung)
    [int]    $Epochs = 75,
    [int]    $EarlyStop = 24,
    [string] $Graph = "data/processed/graphs_biochem_anchors/patient007.pt",
    [switch] $Smoke
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$dynamic = -not $Static

# Resolve the teacher ckpt path + gate dir (mirror go_species_flow_aware.ps1's RunName logic).
$RunName = "flow_aware"
if ($LatentDropout -gt 0.0) { $RunName = "flow_aware_leashed" }
if ($dynamic) { $RunName = "${RunName}_dynamic" }
$RunRoot = Join-Path $RepoRoot "outputs/biochem/biochem_gnn/$RunName"
$Ckpt = Join-Path $RunRoot "sage/species/best.pth"
$GateRel = "outputs/biochem/corrector_coupling/gate_ladder_$RunName"
$GateDir = Join-Path $RepoRoot $GateRel

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " FRESH clot-flow gate: leash=$LatentDropout dynamic=$dynamic epochs=$Epochs" -ForegroundColor Cyan
Write-Host "   teacher -> $RunRoot" -ForegroundColor DarkGray
Write-Host "   gate    -> $GateDir" -ForegroundColor DarkGray
Write-Host "==================================================================" -ForegroundColor Cyan

# --- FRESH: wipe prior teacher + gate outputs for this config -------------------------------
foreach ($p in @($RunRoot, $GateDir)) {
    if (Test-Path $p) { Write-Host "[clean] $p" -ForegroundColor DarkGray; Remove-Item -Recurse -Force $p }
}

# --- 1) train the fresh teacher -------------------------------------------------------------
$trainArgs = @("-NoProfile", "-ExecutionPolicy", "Bypass",
               "-File", (Join-Path $PSScriptRoot "go_species_flow_aware.ps1"),
               "-LatentDropout", "$LatentDropout", "-Epochs", "$Epochs", "-EarlyStop", "$EarlyStop")
if ($dynamic) { $trainArgs += "-DynamicFlow" }
if ($Smoke) { $trainArgs += "-Smoke" }
Write-Host "`n[stage 1/2] train teacher" -ForegroundColor Green
& powershell @trainArgs
if ($LASTEXITCODE -ne 0) { throw "teacher training failed (exit $LASTEXITCODE)" }
if (-not (Test-Path $Ckpt)) { throw "expected ckpt not found after training: $Ckpt" }

# --- 2) run the full gate ladder ------------------------------------------------------------
$ladderArgs = @("-NoProfile", "-ExecutionPolicy", "Bypass",
                "-File", (Join-Path $PSScriptRoot "go_clot_flow_gate_ladder.ps1"),
                "-Ckpt", $Ckpt, "-Graph", $Graph, "-OutDir", $GateRel)
if (-not $dynamic) { $ladderArgs += "-SkipDynamic" }
Write-Host "`n[stage 2/2] gate ladder" -ForegroundColor Green
& powershell @ladderArgs
if ($LASTEXITCODE -ne 0) { throw "gate ladder failed (exit $LASTEXITCODE)" }

Write-Host "`n[DONE] teacher + full gate ladder complete." -ForegroundColor Green
Write-Host "[i] summary printed above; per-rung JSON under: $GateDir" -ForegroundColor DarkGray
