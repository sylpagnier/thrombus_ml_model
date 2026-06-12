# Fast-track F1-F6: Step 5a mu readout, 5b coupled smoke, 3b continuous-time spike.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_ml_fast_track.ps1"
#   powershell ... -SkipF1 -SkipF3 -SkipF5Rollout
#   Default phi shell: step1_a35 (best clot ML). F1 inc40 is optional hand-rule baseline.

param(
    [string] $Anchor = "patient007",
    [string] $AnchorDir = "data/processed/graphs_biochem_anchors",
    [string] $Step0Json = "outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json",
    [string] $Step1Ckpt = "outputs/biochem/sweep_clot_ml_physics_6h/step1_a35/clot_ml_step1_best.pth",
    [string] $LaneBCkpt = "",
    [string] $KineCkpt = "outputs/kinematics/kinematics_best.pth",
    [double] $SimEndScale = 1.5,
    [switch] $SkipF1,
    [switch] $SkipF3,
    [switch] $SkipF5Rollout,
    [switch] $SkipTests
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_clot_prior_rule_winner_env.ps1")
. (Join-Path $PSScriptRoot "_python_rc.ps1")

$env:CLOT_TEMPORAL_VEL_SOURCE = "kinematics"
$env:CLOT_PHI_KINE_CKPT = $KineCkpt
$env:CLOT_PHI_MINIMAL_FEATURES = "1"
$env:PYTHONUNBUFFERED = "1"

$OutRoot = "outputs/biochem/clot_ml_ladder/fast_track"
New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot $OutRoot) | Out-Null

function Invoke-Fast {
    param([string] $Label, [string[]] $PyArgs)
    Write-Host ""
    Write-Host "[NEW] $Label" -ForegroundColor Cyan
    Invoke-PythonRcCheck -Label $Label -PyArgs $PyArgs
}

Write-Host "[NEW] clot ML fast-track F1-F6 (ML-first: step1_a35)" -ForegroundColor Cyan
Write-Host "[i] anchor=$Anchor step1=$Step1Ckpt kine=$KineCkpt" -ForegroundColor DarkGray

# F1: inc40 hand-rule baseline (optional)
if (-not $SkipF1) {
    Invoke-Fast "F1 step5a inc40 baseline" @(
        "scripts/eval_clot_ml_step5a_mu_readout.py",
        "--shell", "inc40",
        "--anchor-dir", $AnchorDir,
        "--step0-json", $Step0Json,
        "--kine-ckpt", $KineCkpt,
        "--out", "$OutRoot/step5a_inc40_summary.json"
    )
}

# F2: step1 shell LOAO (best clot ML phi)
Invoke-Fast "F2 step5a step1" @(
    "scripts/eval_clot_ml_step5a_mu_readout.py",
    "--shell", "step1",
    "--anchor-dir", $AnchorDir,
    "--step0-json", $Step0Json,
    "--step1-ckpt", $Step1Ckpt,
    "--kine-ckpt", $KineCkpt,
    "--out", "$OutRoot/step5a_step1_summary.json"
)

# F3: lane_b upper bound (optional -- needs deploy ckpt)
if (-not $SkipF3) {
    $laneArgs = @(
        "scripts/eval_clot_ml_step5a_mu_readout.py",
        "--shell", "lane_b",
        "--anchor-dir", $AnchorDir,
        "--kine-ckpt", $KineCkpt,
        "--out", "$OutRoot/step5a_lane_b_summary.json"
    )
    if ($LaneBCkpt) {
        $laneArgs += @("--lane-b-ckpt", $LaneBCkpt)
    }
    try {
        Invoke-Fast "F3 step5a lane_b" $laneArgs
    }
    catch {
        Write-Host "[WARN] F3 lane_b skipped: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

# F4: coupled kine smoke (step1 phi -> mu prior)
Invoke-Fast "F4 step5b smoke step1" @(
    "scripts/smoke_clot_ml_step5b_coupled_kine.py",
    "--anchor", $Anchor,
    "--anchor-dir", $AnchorDir,
    "--shell", "step1",
    "--step0-json", $Step0Json,
    "--step1-ckpt", $Step1Ckpt,
    "--kine-ckpt", $KineCkpt,
    "--out", "$OutRoot/step5b_smoke_step1.json"
)

# F5: 3b tau spike
$spikeArgs = @(
    "scripts/spike_clot_ml_step3b_continuous_time.py",
    "--anchor", $Anchor,
    "--anchor-dir", $AnchorDir,
    "--step0-json", $Step0Json,
    "--sim-end-scale", "$SimEndScale",
    "--out", "$OutRoot/step3b_spike.json"
)
if (-not $SkipF5Rollout) {
    $spikeArgs += "--compare-rollout"
}
Invoke-Fast "F5 step3b spike" $spikeArgs

# F6: unit tests
if (-not $SkipTests) {
    Invoke-Fast "F6 pytest step3b" @(
        "-m", "pytest", "src/tests/test_clot_ml_step3b_continuous_time.py", "-q"
    )
}

Write-Host ""
Write-Host "[OK] fast-track complete -> $OutRoot" -ForegroundColor Green
