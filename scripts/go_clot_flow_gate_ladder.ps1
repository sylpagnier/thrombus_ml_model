# One-shot clot-flow gate ladder: runs EVERY diagnostic rung on one species ckpt, writes per-rung
# JSON, then prints a consolidated table + the decisive contrasts. Analyze the summary together.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_clot_flow_gate_ladder.ps1 `
#       -Ckpt outputs/biochem/biochem_gnn/flow_aware_leashed/sage/species/best.pth
#
#   # dynamic teacher (adds the #5c time-varying ceiling rung):
#   powershell ... -Ckpt .../flow_aware_leashed_dynamic/sage/species/best.pth
#
# Rungs (all on the SAME ckpt, frozen z_kin unless noted):
#   ref                ref baseline (flow-active, kine)          <- the common reference B
#   ablate             leash check: flow features zeroed         <- B should DROP if leash bit
#   gt_static  (#5)    perfect STATIC GT flow                    <- static ceiling
#   gt_dynamic (#5c)   perfect TIME-VARYING GT flow              <- dynamic ceiling (Trap C)
#   oracle_mu  (#5b)   corrector around the TRUE COMSOL clot
#   corrector  (#6a)   real corrector, frozen z_kin
#   corrector_resolve  (#6b) real corrector + z_kin re-solve ON

param(
    [Parameter(Mandatory = $true)] [string] $Ckpt,
    [string] $Graph = "data/processed/graphs_biochem_anchors/patient007.pt",
    [string] $Times = "",                          # "" = tool default (0, mid, last)
    [string] $OutDir = "outputs/biochem/corrector_coupling/gate_ladder",
    [switch] $SkipDynamic                          # skip #5c (static-only teacher)
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"
# Cap block splitting so the allocator can reuse freed memory instead of fragmenting it -- this is
# what OOMs the corrector_resolve rung on a 4 GiB card (a small alloc fails while 100s of MiB are
# reserved-but-unallocated). Windows-safe, unlike expandable_segments. Keeps the solve on GPU.
if (-not $env:PYTORCH_CUDA_ALLOC_CONF) { $env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128" }

$OutPath = Join-Path $RepoRoot $OutDir
New-Item -ItemType Directory -Force -Path $OutPath | Out-Null

$timeArgs = @()
if ($Times.Trim()) { $timeArgs = @("--times", $Times) }

# Clean any inherited gate env so each rung is hermetic.
foreach ($v in @("BIOCHEM_CORRECTOR_COUPLING", "BIOCHEM_KINE_RESOLVE_ON_CLOT",
                 "SPECIES_FLOW_FEATS_ABLATE", "SPECIES_FLOW_FEATS_SOURCE",
                 "SPECIES_FLOW_FEATS_DYNAMIC", "SPECIES_ROLLOUT_VEL_SOURCE")) {
    if (Test-Path "Env:\$v") { Remove-Item "Env:\$v" }
}

function Invoke-Rung {
    param([string] $Name, [string[]] $Flags, [hashtable] $Env)
    Write-Host "`n[rung] $Name" -ForegroundColor Cyan
    if ($Env) { foreach ($k in $Env.Keys) { Set-Item "Env:\$k" $Env[$k] } }
    $out = Join-Path $OutPath "$Name.json"
    $pyArgs = @("-m", "src.tools.compare_coupled_mat_rollout",
                "--graph", $Graph, "--species-ckpt", $Ckpt, "--out", $out) + $timeArgs + $Flags
    Invoke-PythonRcCheck -Label "gate:$Name" -PyArgs $pyArgs
    if ($Env) { foreach ($k in $Env.Keys) { if (Test-Path "Env:\$k") { Remove-Item "Env:\$k" } } }
}

# --- ref + leash check ----------------------------------------------------------------------
# ref: the tool's else-branch couples (resolve ON by default); its BASELINE pass is the flow-active
#      reference B we compare everything to. (We read t_last_baseline_f1 for ref/ablate.)
Invoke-Rung -Name "ref" -Flags @() -Env @{ "BIOCHEM_KINE_RESOLVE_ON_CLOT" = "0" }
Invoke-Rung -Name "ablate" -Flags @() -Env @{ "BIOCHEM_KINE_RESOLVE_ON_CLOT" = "0"; "SPECIES_FLOW_FEATS_ABLATE" = "1" }

# --- ceilings (frozen z_kin, no corrector) --------------------------------------------------
Invoke-Rung -Name "gt_static" -Flags @("--gt-flow") -Env @{}
if (-not $SkipDynamic) {
    Invoke-Rung -Name "gt_dynamic" -Flags @("--gt-flow-dynamic") -Env @{}
}

# --- corrector mechanism --------------------------------------------------------------------
Invoke-Rung -Name "oracle_mu" -Flags @("--oracle-mu") -Env @{ "BIOCHEM_KINE_RESOLVE_ON_CLOT" = "0" }
Invoke-Rung -Name "corrector" -Flags @() -Env @{ "BIOCHEM_KINE_RESOLVE_ON_CLOT" = "0" }
Invoke-Rung -Name "corrector_resolve" -Flags @() -Env @{ "BIOCHEM_KINE_RESOLVE_ON_CLOT" = "1" }

# --- consolidated summary -------------------------------------------------------------------
Write-Host "`n[i] all rungs done -> summarizing" -ForegroundColor Green
Invoke-PythonRcCheck -Label "gate summary" -PyArgs @("scripts/summarize_clot_flow_gate.py", "--dir", $OutPath)
Write-Host "[OK] per-rung JSON + summary under: $OutPath" -ForegroundColor Green
