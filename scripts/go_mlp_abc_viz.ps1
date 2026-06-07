# Interactive or headless A/B/C mu coupling viz (same legs as abc_compare_1h).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_mlp_abc_viz.ps1
#   powershell ... -Leg B -Anchor patient007
#   powershell ... -Headless -Anchor patient007          # PNGs only (~1h for all 3 legs)
#   powershell ... -Leg All -Fast                        # interactive, faster rollout

param(
    [ValidateSet("A", "B", "B_deploy", "B_seed_growth", "C", "All")]
    [string] $Leg = "All",
    [string] $TeacherCheckpoint = "outputs\biochem\clot_baseline\teacher_best_high_mu.pth",
    [string] $ClotPhiCheckpoint = "outputs\biochem\clot_baseline\clot_phi_best.pth",
    [string] $Anchor = "patient007",
    [int] $SimEndS = 30000,
    [int] $TimeStride = 5,
    [double] $MuRatioMax = 20,
    [double] $Blend = 1.0,
    [switch] $Headless,
    [switch] $Fast,
    [switch] $FullViz
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")

function Clear-MuCouplingEnv {
    Remove-Item Env:BIOCHEM_MLP_CLOT_INJECT, Env:BIOCHEM_MLP_MU_MAP, Env:BIOCHEM_MU_NEIGHBOR_WALL_ONLY -ErrorAction SilentlyContinue
    Remove-Item Env:BIOCHEM_MU_NEIGHBOR_WALL_MASK, Env:BIOCHEM_MU_NEIGHBOR_WALL_BULK -ErrorAction SilentlyContinue
    Remove-Item Env:BIOCHEM_MLP_MU_MAP_PHI_GATE, Env:BIOCHEM_MLP_MU_MAP_MASK, Env:BIOCHEM_MLP_MU_MAP_BULK -ErrorAction SilentlyContinue
    Remove-Item Env:BIOCHEM_MLP_MU_MAP_GAMMA_THRESH_ND, Env:BIOCHEM_MLP_MU_MAP_GEO_CAP -ErrorAction SilentlyContinue
    Remove-Item Env:BIOCHEM_MLP_CLOT_REGION, Env:BIOCHEM_MLP_CLOT_CKPT -ErrorAction SilentlyContinue
    Remove-Item Env:BIOCHEM_MLP_NEIGHBOR_SEED, Env:BIOCHEM_MLP_NEIGHBOR_REQUIRE_PHI -ErrorAction SilentlyContinue
    Remove-Item Env:BIOCHEM_MLP_MU_MAP_PHI_THRESH -ErrorAction SilentlyContinue
    Remove-Item Env:BIOCHEM_MLP_DEPLOY_DGAMMA_SLICE, Env:BIOCHEM_MLP_DEPLOY_PHI_Q -ErrorAction SilentlyContinue
    Remove-Item Env:BIOCHEM_MLP_NEIGHBOR_GROWTH_ONLY, Env:BIOCHEM_MLP_DEPLOY_MU_EXCESS_SI -ErrorAction SilentlyContinue
    Remove-Item Env:BIOCHEM_MLP_DEPLOY_REQUIRE_MLP_CLOTS -ErrorAction SilentlyContinue
    Remove-Item Env:CLOT_SHAPE_MU_THRESH_SI -ErrorAction SilentlyContinue
    Remove-Item Env:VIZ_ABC_LEG, Env:VIZ_MU_DYNAMIC_RECOMPUTE -ErrorAction SilentlyContinue
}

function Set-MuCouplingLegEnv {
    param([ValidateSet("A", "B", "B_deploy", "B_seed_growth", "C")][string] $LegId)
    Clear-MuCouplingEnv
    switch ($LegId) {
        "A" {
            # Baseline: full-domain GNODE mu head only.
        }
        "B" {
            $env:BIOCHEM_MLP_MU_MAP = "1"
            $env:BIOCHEM_MLP_MU_MAP_PHI_GATE = "1"
            $env:BIOCHEM_MLP_MU_MAP_MASK = "gt_clot"
            $env:BIOCHEM_MLP_MU_MAP_BULK = "cap_low_shear"
            $env:BIOCHEM_MLP_MU_MAP_GAMMA_THRESH_ND = "0.01"
            $env:BIOCHEM_MLP_MU_MAP_GEO_CAP = "0"
            $env:BIOCHEM_MLP_CLOT_CKPT = $ClotPhi
            $env:BIOCHEM_MLP_CLOT_BLEND = "$Blend"
        }
        "B_deploy" {
            # Must set PowerShell env directly (python os.environ does not persist to parent shell).
            $env:BIOCHEM_MLP_MU_MAP = "1"
            $env:BIOCHEM_MLP_MU_MAP_PHI_GATE = "1"
            $env:BIOCHEM_MLP_MU_MAP_MASK = "neighbor"
            $env:BIOCHEM_MLP_MU_MAP_BULK = "cap_low_shear"
            $env:BIOCHEM_MLP_MU_MAP_GAMMA_THRESH_ND = "0.01"
            $env:BIOCHEM_MLP_MU_MAP_GEO_CAP = "0"
            $env:BIOCHEM_MLP_NEIGHBOR_SEED = "pred_clot"
            $env:BIOCHEM_MLP_NEIGHBOR_REQUIRE_PHI = "1"
            $env:BIOCHEM_MLP_MU_MAP_PHI_THRESH = "0.5"
            $env:BIOCHEM_MLP_DEPLOY_DGAMMA_SLICE = "0"
            $env:BIOCHEM_MLP_DEPLOY_PHI_Q = "0"
            $env:BIOCHEM_MLP_NEIGHBOR_GROWTH_ONLY = "0"
            $env:BIOCHEM_MLP_DEPLOY_MU_EXCESS_SI = "0"
            $env:BIOCHEM_MLP_DEPLOY_VISION_RESTRICT = "1"
            $env:BIOCHEM_MLP_DEPLOY_VISION_INIT = "comsol_t0"
            $env:BIOCHEM_MLP_DEPLOY_VISION_GROW = "1"
            $env:BIOCHEM_MLP_DEPLOY_VISION_GROW_HOPS = "1"
            $env:BIOCHEM_MLP_DEPLOY_NO_COMMIT_T0 = "1"
            if (-not $env:CLOT_SHAPE_MU_THRESH_SI) { $env:CLOT_SHAPE_MU_THRESH_SI = "0.055" }
            $env:BIOCHEM_MLP_CLOT_CKPT = $ClotPhi
            $env:BIOCHEM_MLP_CLOT_BLEND = "$Blend"
        }
        "B_seed_growth" {
            $env:BIOCHEM_MLP_MU_MAP = "1"
            $env:BIOCHEM_MLP_MU_MAP_PHI_GATE = "1"
            $env:BIOCHEM_MLP_MU_MAP_MASK = "seed_growth"
            $env:BIOCHEM_MLP_MU_MAP_BULK = "cap_low_shear"
            $env:BIOCHEM_MLP_MU_MAP_GAMMA_THRESH_ND = "0.01"
            $env:BIOCHEM_MLP_MU_MAP_GEO_CAP = "0"
            $env:BIOCHEM_MLP_NEIGHBOR_REQUIRE_PHI = "1"
            $env:BIOCHEM_MLP_MU_MAP_PHI_THRESH = "0.5"
            $env:BIOCHEM_MLP_SEED_GROWTH_INIT = "comsol_t0"
            $env:BIOCHEM_MLP_SEED_GROWTH_HOPS = "1"
            $env:BIOCHEM_MLP_DEPLOY_VISION_RESTRICT = "1"
            $env:BIOCHEM_MLP_DEPLOY_VISION_INIT = "comsol_t0"
            $env:BIOCHEM_MLP_DEPLOY_VISION_GROW = "1"
            $env:BIOCHEM_MLP_DEPLOY_VISION_GROW_HOPS = "1"
            $env:BIOCHEM_MLP_NEIGHBOR_GROWTH_ONLY = "0"
            $env:BIOCHEM_MLP_DEPLOY_DGAMMA_SLICE = "0"
            $env:BIOCHEM_MLP_DEPLOY_PHI_Q = "0"
            $env:BIOCHEM_MLP_DEPLOY_MU_EXCESS_SI = "0"
            $env:BIOCHEM_MLP_DEPLOY_REQUIRE_MLP_CLOTS = "0"
            if (-not $env:CLOT_SHAPE_MU_THRESH_SI) { $env:CLOT_SHAPE_MU_THRESH_SI = "0.055" }
            $env:BIOCHEM_MLP_CLOT_CKPT = $ClotPhi
            $env:BIOCHEM_MLP_CLOT_BLEND = "$Blend"
        }
        "C" {
            $env:BIOCHEM_MU_NEIGHBOR_WALL_ONLY = "1"
            $env:BIOCHEM_MU_NEIGHBOR_WALL_MASK = "gt_clot"
            $env:BIOCHEM_MU_NEIGHBOR_WALL_BULK = "cap_low_shear"
            $env:BIOCHEM_MLP_MU_MAP_GAMMA_THRESH_ND = "0.01"
        }
    }
}

$Teacher = Join-Path $RepoRoot ($TeacherCheckpoint -replace '/', '\')
if (-not (Test-Path $Teacher)) {
    Write-Host "[ERR] Missing teacher ckpt: $TeacherCheckpoint" -ForegroundColor Red
    exit 1
}
$ClotPhi = Join-Path $RepoRoot ($ClotPhiCheckpoint -replace '/', '\')
if (-not (Test-Path $ClotPhi)) {
    Write-Host "[ERR] Missing clot-phi ckpt: $ClotPhiCheckpoint" -ForegroundColor Red
    exit 1
}

$env:BIOCHEM_GT_KINE_VEL = "0"
Remove-Item Env:BIOCHEM_GT_KINE_SKIP_DEQ -ErrorAction SilentlyContinue
$env:BIOCHEM_TEACHER_MU_RATIO_MAX = "$MuRatioMax"

if ($Headless) {
    $legsArg = if ($Leg -eq "All") { "A,B,C" } else { $Leg }
    Write-Host "[NEW] A/B/C headless mu snapshots (legs=$legsArg anchor=$Anchor)" -ForegroundColor Cyan
    $pyArgs = @(
        (Join-Path $RepoRoot "scripts\snapshot_mlp_abc_mu.py"),
        "--teacher-checkpoint", $TeacherCheckpoint,
        "--clot-phi-checkpoint", $ClotPhiCheckpoint,
        "--anchor", $Anchor,
        "--legs", $legsArg,
        "--time-stride", "$TimeStride",
        "--mu-ratio-max", "$MuRatioMax"
    )
    $rc = Invoke-PythonRc @pyArgs
    if ($rc -ne 0) { exit $rc }
    Write-Host "[OK]  outputs\biochem\viz\abc_mu\" -ForegroundColor Green
    exit 0
}

if ($FullViz) {
    $env:VIZ_FAST = "0"
} elseif ($Fast) {
    $env:VIZ_FAST = "1"
} else {
    Remove-Item Env:VIZ_FAST -ErrorAction SilentlyContinue
}

$legs = @()
if ($Leg -eq "All") { $legs = @("A", "B", "C") } else { $legs = @($Leg) }

Write-Host "[NEW] A/B/C interactive viz (matplotlib windows)" -ForegroundColor Cyan
Write-Host "[i]  legs=$($legs -join ',') anchor=$Anchor sim_end=${SimEndS}s" -ForegroundColor DarkGray
Write-Host "[i]  close each figure window to advance to the next leg" -ForegroundColor DarkGray

foreach ($legId in $legs) {
    Set-MuCouplingLegEnv -LegId $legId
    $legNote = switch ($legId) {
        "A" { "baseline full-domain GNODE mu" }
        "B" { "MLP mu map v2 gt_clot + cap_low_shear" }
        "B_deploy" { "MLP mu map deploy neighbor (wall + 1-hop pred clot/phi)" }
        "B_seed_growth" { "MLP seed_growth (GT t=0 vision + 1-hop pred clot expand)" }
        "C" { "GNODE mask-only gt_clot + cap_low_shear" }
    }
    Write-Host ""
    Write-Host "[NEW] Leg $legId -- $legNote" -ForegroundColor Cyan
    $env:VIZ_ABC_LEG = "Leg $legId"
    Remove-Item Env:VIZ_MU_DYNAMIC_RECOMPUTE -ErrorAction SilentlyContinue

    $pyArgs = @(
        "-m", "src.evaluation.visualize_pipeline",
        "--teacher-only",
        "--biochem-checkpoint", $Teacher,
        "--anchor", $Anchor,
        "--sim-end-s", "$SimEndS",
        "--no-sim-end-prompt",
        "--clot-phi-checkpoint", $ClotPhi
    )
    if ($FullViz) { $pyArgs += "--full-viz" }

    $rc = Invoke-PythonRc @pyArgs
    if ($rc -ne 0) { exit $rc }
}

Write-Host "[OK]  A/B/C viz complete." -ForegroundColor Green
