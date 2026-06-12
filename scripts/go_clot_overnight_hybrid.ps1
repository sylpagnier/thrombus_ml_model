# Overnight: GPU teacher species dump -> hybrid sweep -> promote winner -> timeline viz.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_overnight_hybrid.ps1"
#   powershell ... -SkipDump -SkipArchSweep   # hybrid-only if dump already done
#   powershell ... -Teacher outputs/biochem/biochem_teacher_passive_xy_locked.pth

param(
    [string] $Teacher = "",
    [string] $Anchor = "patient007",
    [int] $Keyframes = 8,
    [switch] $SkipDump,
    [switch] $SkipArchSweep,
    [switch] $SkipHybridSweep,
    [switch] $SkipViz,
    [switch] $SkipSpeciesEval
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")

$LogDir = Join-Path $RepoRoot "outputs\biochem\diagnostics"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogFile = Join-Path $LogDir "clot_overnight_hybrid.log"
function Log([string]$Msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Msg"
    Add-Content -Path $LogFile -Value $line
    Write-Host $line
}

Log "[NEW] clot overnight hybrid pipeline"

if (-not $Teacher) {
    $candidates = @(
        "outputs\biochem\biochem_teacher_passive_xy_locked.pth",
        "outputs\biochem\biochem_teacher_passive_align_locked.pth",
        "outputs\biochem\biochem_teacher_passive_species_locked.pth",
        "outputs\biochem\sweep_mu_complexity_6h\FULL_step2\biochem_teacher_best_high_mu.pth",
        "outputs\biochem\biochem_teacher_last.pth"
    )
    foreach ($c in $candidates) {
        $p = Join-Path $RepoRoot $c
        if (Test-Path $p) {
            $Teacher = $c
            break
        }
    }
}

if (-not $SkipDump) {
    if (-not $Teacher -or -not (Test-Path (Join-Path $RepoRoot $Teacher))) {
        Log "[ERR] no teacher checkpoint found; set -Teacher or place passive_xy_locked.pth"
        exit 1
    }
    Log "[i] phase 1 GPU: dump teacher species ($Teacher)"
    Invoke-PythonRcCheck -Label "dump teacher species" -PyArgs @(
        "scripts/dump_teacher_species_to_anchors.py",
        "--teacher", $Teacher,
        "--out-dir", "outputs/biochem/anchors_teacher_species",
        "--device", "cuda",
        "--time-stride", "6",
        "--min-steps", "12",
        "--force"
    )
} else {
    Log "[skip] species dump"
}

if (-not $SkipSpeciesEval -and $Teacher -and (Test-Path (Join-Path $RepoRoot $Teacher))) {
    Log "[i] phase 1b GPU: passive species anchor eval"
    Invoke-PythonRcCheck -Label "passive species eval" -PyArgs @(
        "scripts/eval_passive_species_anchors.py",
        "--checkpoint", $Teacher,
        "--device", "cuda"
    ) | Out-Null
    if ($LASTEXITCODE -ne 0) { Log "[WARN] passive species eval failed (exit=$LASTEXITCODE); continuing" }
}

if (-not $SkipArchSweep) {
    Log "[i] phase 2 CPU: full architecture sweep (resume, fixed JSON)"
    Invoke-PythonRcCheck -Label "architecture sweep" -PyArgs @(
        "scripts/sweep_clot_rule_architectures.py",
        "--resume"
    )
} else {
    Log "[skip] architecture sweep"
}

if (-not $SkipHybridSweep) {
    $bakedDir = Join-Path $RepoRoot "outputs\biochem\anchors_teacher_species"
    if (-not (Get-ChildItem $bakedDir -Filter "*.pt" -ErrorAction SilentlyContinue)) {
        Log "[ERR] no baked anchors in $bakedDir (run dump first)"
        exit 1
    }
    Log "[i] phase 3 CPU: hybrid teacher-species rule sweep"
    Invoke-PythonRcCheck -Label "hybrid species sweep" -PyArgs @(
        "scripts/sweep_clot_rule_architectures.py",
        "--hybrid-species",
        "--resume"
    )
} else {
    Log "[skip] hybrid species sweep"
}

Log "[i] phase 4: promote hybrid winner env"
Invoke-PythonRcCheck -Label "promote winner" -PyArgs @(
    "scripts/promote_clot_architecture_winner.py",
    "--json", "outputs/biochem/diagnostics/clot_hybrid_species_sweep.json"
)

if (-not $SkipViz) {
    Log "[i] phase 5: timeline viz ($Anchor)"
    . (Join-Path $PSScriptRoot "_clot_prior_rule_winner_env.ps1")
    $archEnv = Join-Path $PSScriptRoot "_clot_architecture_winner_env.ps1"
    if (Test-Path $archEnv) { . $archEnv }
    $vizAnchorDir = "data/processed/graphs_biochem_anchors"
    $spw = $env:CLOT_LOCALIZED_SPECIES_WEIGHT
    if ($spw -and [double]$spw -gt 0) {
        $vizAnchorDir = "outputs/biochem/anchors_teacher_species"
        Log "[i] viz uses baked teacher-species anchors (species_weight=$spw)"
    }
    & (Join-Path $PSScriptRoot "go_clot_temporal_rule_timeline_viz.ps1") `
        -Anchor $Anchor -AnchorDir $vizAnchorDir -Keyframes $Keyframes
} else {
    Log "[skip] viz"
}

Log "[OK] overnight hybrid pipeline complete"
Log "[save] log $LogFile"
