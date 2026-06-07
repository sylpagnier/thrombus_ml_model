# Fair head-to-head: GNODE12 Lane A forward (Leg A) vs MLP mu map v2 (Leg B).
# Same teacher + clot-phi ckpts, pred kine, mu_ratio_max, anchors, rollout stride.
# North-star rank = clot_shape on rollout ch3 (not offline clot-phi F1).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_mlp_ab_fair.ps1 -Fast
#   powershell ... -Anchors "patient007" -TimeStride 5
#   powershell ... -VizOnly -Anchor patient007          # headless PNGs A+B only (~30 min)
#   powershell ... -VizOnly -Interactive -Anchor patient007

param(
    [string] $TeacherCheckpoint = "outputs\biochem\clot_baseline\teacher_best_high_mu.pth",
    [string] $ClotPhiCheckpoint = "outputs\biochem\clot_baseline\clot_phi_best.pth",
    [string] $Anchors = "patient003,patient007,patient006",
    [string] $Anchor = "patient007",
    [int] $TimeStride = 5,
    [double] $MuRatioMax = 20,
    [double] $Blend = 1.0,
    [switch] $Fast,
    [switch] $VizOnly,
    [switch] $Interactive
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")

$env:BIOCHEM_GT_KINE_VEL = "0"
Remove-Item Env:BIOCHEM_GT_KINE_SKIP_DEQ -ErrorAction SilentlyContinue
$env:BIOCHEM_TEACHER_MU_RATIO_MAX = "$MuRatioMax"
$env:BIOCHEM_MLP_CLOT_BLEND = "$Blend"
Remove-Item Env:BIOCHEM_MLP_CLOT_INJECT, Env:BIOCHEM_MLP_MU_MAP, Env:BIOCHEM_MU_NEIGHBOR_WALL_ONLY -ErrorAction SilentlyContinue
if (-not $env:CLOT_SHAPE_MU_THRESH_SI) { $env:CLOT_SHAPE_MU_THRESH_SI = "0.055" }

Write-Host "[NEW] Lane A vs Leg B fair compare (same ckpt stack)" -ForegroundColor Cyan
Write-Host "[i]  A = GNODE teacher mu (Lane A default forward)" -ForegroundColor DarkGray
Write-Host "[i]  B = Carreau bulk + MLP mu map on gt_clot (closed-loop ch3)" -ForegroundColor DarkGray
Write-Host "[i]  teacher=$TeacherCheckpoint" -ForegroundColor DarkGray
Write-Host "[i]  clot-phi=$ClotPhiCheckpoint" -ForegroundColor DarkGray

if ($VizOnly) {
    if ($Interactive) {
        Write-Host "[NEW] interactive viz (close each window to advance)" -ForegroundColor Cyan
        foreach ($leg in @("A", "B")) {
            $rc = & (Join-Path $PSScriptRoot "go_mlp_abc_viz.ps1") `
                -Leg $leg -Anchor $Anchor -TimeStride $TimeStride `
                -TeacherCheckpoint $TeacherCheckpoint -ClotPhiCheckpoint $ClotPhiCheckpoint `
                -MuRatioMax $MuRatioMax -Blend $Blend
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        }
        exit 0
    }
    Write-Host "[NEW] headless PNGs (legs A,B anchor=$Anchor stride=$TimeStride)" -ForegroundColor Cyan
    $pyArgs = @(
        (Join-Path $RepoRoot "scripts\snapshot_mlp_abc_mu.py"),
        "--teacher-checkpoint", $TeacherCheckpoint,
        "--clot-phi-checkpoint", $ClotPhiCheckpoint,
        "--anchor", $Anchor,
        "--legs", "A,B",
        "--time-stride", "$TimeStride",
        "--mu-ratio-max", "$MuRatioMax"
    )
    $rc = Invoke-PythonRc @pyArgs
    if ($rc -ne 0) { exit $rc }
    Write-Host "[OK]  outputs\biochem\viz\abc_mu\leg_A_* and leg_B_*" -ForegroundColor Green
    exit 0
}

$outJson = if ($Fast) {
    "outputs/biochem/mlp_clot_inject_probe/ab_fair_fast.json"
} else {
    "outputs/biochem/mlp_clot_inject_probe/ab_fair_1h.json"
}

Write-Host "[NEW] clot_shape scorecard (legs A,B only)" -ForegroundColor Cyan
Write-Host "[i]  anchors=$Anchors  time_stride=$TimeStride  fast=$($Fast.IsPresent)" -ForegroundColor DarkGray

$pyArgs = @(
    (Join-Path $RepoRoot "scripts\run_mlp_clot_inject_probe.py"),
    "--teacher-checkpoint", $TeacherCheckpoint,
    "--clot-phi-checkpoint", $ClotPhiCheckpoint,
    "--anchors", $Anchors,
    "--legs", "A,B",
    "--time-stride", "$TimeStride",
    "--mu-ratio-max", "$MuRatioMax",
    "--out", $outJson
)
if ($Fast) { $pyArgs += "--fast" }

$rc = Invoke-PythonRc @pyArgs
if ($rc -ne 0) { exit $rc }

Write-Host "[OK]  $outJson" -ForegroundColor Green
Write-Host "[i]  viz: .\scripts\go_mlp_ab_fair.ps1 -VizOnly -Anchor $Anchor" -ForegroundColor DarkGray
Write-Host "[i]  or:  .\scripts\go_mlp_abc_viz.ps1 -Leg A|B -Anchor $Anchor" -ForegroundColor DarkGray
