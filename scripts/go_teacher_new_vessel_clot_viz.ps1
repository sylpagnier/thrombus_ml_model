# Teacher + clot-phi viz via visualize_pipeline (default: patient biochem anchor).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_teacher_new_vessel_clot_viz.ps1
#   powershell ... -Anchor patient003
#   powershell ... -Synthetic -Seed 7
#   powershell ... -SimEndS 30000          # default; skip prompt
#   powershell ... -Prompt                 # ask simulation length [s]
#   powershell ... -Checkpoint outputs\biochem\clot_baseline\teacher_best_high_mu.pth

param(
    [string] $Checkpoint = "",
    [string] $Anchor = "patient007",
    [int] $Seed = 42,
    [int] $SimEndS = 30000,
    [switch] $Prompt,
    [double] $MuRatioMax = 20,
    [switch] $Synthetic,
    [switch] $FullViz,
    [switch] $ReuseSynthetic,
    [switch] $DeployMuMap
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")

function Resolve-TeacherDeployCkpt {
    param([string] $UserPath = "")
    if ($UserPath) {
        $p = if ([System.IO.Path]::IsPathRooted($UserPath)) { $UserPath } else { Join-Path $RepoRoot $UserPath }
        if (Test-Path $p) { return $p }
    }
    foreach ($rel in @(
            "outputs\biochem\clot_baseline\teacher_best_high_mu.pth",
            "outputs\biochem\gnode10_sweep\gnode12_mu_unlock\biochem_teacher_best_high_mu.pth",
            "outputs\biochem\gnode10_sweep\gnode12_lane_a_promoted\biochem_teacher_best_high_mu.pth",
            "outputs\biochem\sweep_mu_complexity_6h\FULL_step2\biochem_teacher_best_high_mu.pth",
            "outputs\biochem\biochem_teacher_best_high_mu.pth"
        )) {
        $c = Join-Path $RepoRoot $rel
        if (Test-Path $c) { return $c }
    }
    return $null
}

$Ckpt = Resolve-TeacherDeployCkpt -UserPath $Checkpoint
if (-not $Ckpt) {
    Write-Host "[ERR] No teacher checkpoint found. Train or pass -Checkpoint." -ForegroundColor Red
    exit 1
}

$ClotPhiCkpt = Join-Path $RepoRoot "outputs\biochem\clot_baseline\clot_phi_best.pth"
if (-not (Test-Path $ClotPhiCkpt)) {
    $ClotPhiCkpt = Join-Path $RepoRoot "outputs\biochem\passive_species_focus_compare\gnode12_lane_a_clotphi\clot_phi_best.pth"
}

Write-Host "[NEW] Teacher + clot-phi viz (visualize_pipeline)" -ForegroundColor Cyan
Write-Host "[i]  teacher=$Ckpt" -ForegroundColor DarkGray
if (Test-Path $ClotPhiCkpt) {
    Write-Host "[i]  clot-phi=$ClotPhiCkpt" -ForegroundColor DarkGray
} else {
    Write-Host "[WARN] No clot-phi ckpt; mu_eff panel only (train go_baseline_clot.ps1)" -ForegroundColor Yellow
}

$env:BIOCHEM_GT_KINE_VEL = "0"
Remove-Item Env:BIOCHEM_GT_KINE_SKIP_DEQ -ErrorAction SilentlyContinue
$env:BIOCHEM_TEACHER_MU_RATIO_MAX = "$MuRatioMax"

$pyArgs = @(
    "-m", "src.evaluation.visualize_pipeline",
    "--teacher-only",
    "--biochem-checkpoint", $Ckpt,
    "--no-sim-end-prompt"
)

if ($Prompt) {
    $pyArgs = $pyArgs | Where-Object { $_ -ne "--no-sim-end-prompt" }
} else {
    $pyArgs += @("--sim-end-s", "$SimEndS")
    Write-Host "[i]  sim_end=${SimEndS}s (use -Prompt to ask, or -SimEndS N)" -ForegroundColor DarkGray
}

if ($Synthetic) {
    Write-Host "[i]  synthetic vessel seed=$Seed" -ForegroundColor DarkGray
    $pyArgs += @("--synthetic", "--seed", "$Seed")
    if ($ReuseSynthetic) {
        $pyArgs += "--reuse"
    } else {
        $pyArgs += "--regenerate"
    }
} else {
    Write-Host "[i]  anchor=$Anchor" -ForegroundColor DarkGray
    $pyArgs += @("--anchor", $Anchor)
}

if ($FullViz) {
    $pyArgs += "--full-viz"
}
if (Test-Path $ClotPhiCkpt) {
    $pyArgs += @("--clot-phi-checkpoint", $ClotPhiCkpt)
}
if ($DeployMuMap) {
    $pyArgs += "--deploy-mu-map"
    Write-Host "[i]  deploy-mu-map: wired mlp_band closed-loop (matches dynamic mu panel)" -ForegroundColor DarkGray
}

Write-Host "[viz] Time idx slider drives Flow window + Clot MLP map. Close windows to exit." -ForegroundColor DarkGray
$rc = Invoke-PythonRc @pyArgs
if ($rc -ne 0) { exit $rc }

Write-Host "[OK]  Done." -ForegroundColor Green
