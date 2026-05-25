# ~10h kinematics recovery sweep on the MAIN graph tree (graphs_kinematics/newtonian).
# Goal: find a recipe that approaches Apr-2026 val Rel L2 ~0.10 (target < 0.05 on full val).
#
# Prerequisite (once): python -m src.data_gen.backfill_kinematics_geometry_level
# Do NOT set KINEMATICS_GRAPH_RHEOLOGY_DIR (uses pre-A/B cohort).
#
# Morning leaderboard:
#   Get-Content outputs\kinematics\sweep_recovery_12h\manifest.jsonl | % { $_ | ConvertFrom-Json } |
#     Sort-Object { [double]$_.best_rel_l2 } | ft leg_id,best_rel_l2,best_l0,best_l1,best_l2,best_epoch,n_graphs

param(
    [string[]] $Legs = @(),
    [double]$TargetHours = 10.0,
    [double]$EstMinPerEpoch = 13.0,
    [switch] $DryRun,
    [switch] $Force,
    [string[]] $ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# April best used ~2000 graphs; cap avoids RAM blow-up when 3000 .pt files exist on disk.
$env:KINEMATICS_GRAPH_CAP = "2000"
# Quiet logs: no tqdm percent spam (epoch + val lines only).
$env:KINEMATICS_QUIET = "1"

$hostName = $env:COMPUTERNAME
$SweepDir = Join-Path $RepoRoot "outputs\kinematics\sweep_recovery_12h"
$ManifestPath = Join-Path $SweepDir "manifest.jsonl"
$GraphDir = Join-Path $RepoRoot "data\processed\graphs_kinematics\newtonian"
$SummaryPath = Join-Path $RepoRoot "outputs\reports\training\kinematics\recovery12h_summary.txt"

if (-not (Test-Path $SweepDir)) {
    New-Item -ItemType Directory -Path $SweepDir -Force | Out-Null
}
if (-not (Test-Path (Split-Path $SummaryPath))) {
    New-Item -ItemType Directory -Path (Split-Path $SummaryPath) -Force | Out-Null
}

function Clear-KinematicsSweepEnv {
    Remove-Item Env:KINEMATICS_GRAPH_RHEOLOGY_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:KINEMATICS_BEND_SIGN_MODE -ErrorAction SilentlyContinue
    Remove-Item Env:KINEMATICS_OUTPUT_DIR -ErrorAction SilentlyContinue
}

function Invoke-TrainPython {
    param([string[]]$Cmd)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & python @Cmd *>&1 | ForEach-Object { Write-Host $_ }
    $code = [int]$LASTEXITCODE
    $ErrorActionPreference = $prev
    return $code
}

function Get-GraphCount {
    if (-not (Test-Path $GraphDir)) { return 0 }
    return @(Get-ChildItem -Path $GraphDir -Filter "vessel_*.pt" -File).Count
}

function Get-BestValFromJsonl {
    param([string]$JsonlPath)
    if (-not (Test-Path $JsonlPath)) { return $null }
    $best = $null
    $bestRel = [double]::PositiveInfinity
    foreach ($line in Get-Content $JsonlPath) {
        if (-not $line.Trim()) { continue }
        $row = $line | ConvertFrom-Json
        $rel = $row.rel_l2
        if ($null -eq $rel) { continue }
        $relD = [double]$rel
        # Prefer stage-3 Carreau rows when present.
        $stage = 0
        if ($null -ne $row.stage) { $stage = [int]$row.stage }
        if ($stage -lt 3) { continue }
        if ($relD -lt $bestRel) {
            $bestRel = $relD
            $best = $row
        }
    }
    if ($null -eq $best) {
        foreach ($line in Get-Content $JsonlPath) {
            if (-not $line.Trim()) { continue }
            $row = $line | ConvertFrom-Json
            $relD = [double]$row.rel_l2
            if ($relD -lt $bestRel) {
                $bestRel = $relD
                $best = $row
            }
        }
    }
    return $best
}

function Save-LegArtifacts {
    param(
        [string]$LegId,
        [string]$Title,
        [string]$Hypothesis,
        [int]$GraphCount
    )
    $legDir = Join-Path $SweepDir $LegId
    $outDir = Join-Path $legDir "kinematics_out"
    $srcBest = Join-Path $outDir "kinematics_best.pth"
    if (-not (Test-Path $srcBest)) {
        throw "Missing $srcBest after leg $LegId"
    }
    Copy-Item $srcBest (Join-Path $legDir "kinematics_best.pth") -Force
    foreach ($name in @("kinematics_architecture.json", "kinematics_validation.jsonl")) {
        $src = Join-Path $outDir $name
        if (Test-Path $src) {
            Copy-Item $src (Join-Path $legDir $name) -Force
        }
    }
    $best = Get-BestValFromJsonl -JsonlPath (Join-Path $legDir "kinematics_validation.jsonl")
    $rel = ""; $l0 = ""; $l1 = ""; $l2 = ""; $ep = ""; $cont = ""; $comp = ""
    if ($best) {
        $rel = [string]$best.rel_l2
        $ep = [string]$best.epoch
        $cont = [string]$best.continuity
        $comp = [string]$best.composite
        if ($best.PSObject.Properties.Name -contains "rel_l2_level_0") { $l0 = [string]$best.rel_l2_level_0 }
        if ($best.PSObject.Properties.Name -contains "rel_l2_level_1") { $l1 = [string]$best.rel_l2_level_1 }
        if ($best.PSObject.Properties.Name -contains "rel_l2_level_2") { $l2 = [string]$best.rel_l2_level_2 }
    }
    $manifestRow = @{
        leg_id = $LegId
        title = $Title
        hypothesis = $Hypothesis
        host = $hostName
        n_graphs = $GraphCount
        best_rel_l2 = $rel
        best_l0 = $l0
        best_l1 = $l1
        best_l2 = $l2
        best_epoch = $ep
        best_continuity = $cont
        best_composite = $comp
        checkpoint = (Join-Path $legDir "kinematics_best.pth").Replace("\", "/")
    } | ConvertTo-Json -Compress
    Add-Content -Path $ManifestPath -Value $manifestRow
}

function Get-ScaledLegSchedule {
    param(
        [hashtable]$Def,
        [int]$LegEpochs
    )
    $base = [int]$Def.Epochs
    if ($base -lt 1) { $base = 1 }
    $epochs = [int][math]::Max(4, $LegEpochs)
    $adam = [int][math]::Min($epochs - 1, [math]::Max(2, [math]::Round($Def.Adam * $epochs / $base)))
    $s1 = [int][math]::Max(1, [math]::Round($Def.S1 * $epochs / $base))
    $s2 = [int][math]::Max($s1 + 1, [math]::Round($Def.S2 * $epochs / $base))
    if ($s2 -ge $epochs) { $s2 = $epochs - 1 }
    $l0l1 = [int][math]::Min([int]$Def.L0L1Only, [math]::Max(0, $s1 - 1))
    $hard = [int]$Def.HardMine
    if ($hard -gt $s1) { $hard = [math]::Max(1, $s1) }
    return @{
        Epochs = $epochs
        Adam = $adam
        S1 = $s1
        S2 = $s2
        L0L1Only = $l0l1
        HardMine = $hard
    }
}

# Reference epoch counts (full recipe); scaled to TargetHours at runtime.
$LegCatalog = [ordered]@{
    A0_april_ratio = @{
        Title = "April-2026 stage ratios (40/60), no L0/L1 warmstart"
        Hypothesis = "Match LadHyX curriculum timing; L2 in train from epoch 0"
        Epochs = 55; Adam = 47; S1 = 22; S2 = 33
        L0L1Only = 0; HardMine = 16; NoGeom = $false
        Shuffle = $false; WData = 500; WMu = 10; WWss = 10
    }
    F0_foundation = @{
        Title = "Default geometry curriculum (6 ep L0/L1-only)"
        Hypothesis = "Current best-practice foundation recipe"
        Epochs = 42; Adam = 36; S1 = 17; S2 = 25
        L0L1Only = 6; HardMine = 16; NoGeom = $false
        Shuffle = $false; WData = 500; WMu = 10; WWss = 10
    }
    F1_long_l0l1 = @{
        Title = "Long L0/L1 warmstart (12 ep), delayed hard mining"
        Hypothesis = "Fix L1 before introducing L2 sampling"
        Epochs = 45; Adam = 38; S1 = 18; S2 = 27
        L0L1Only = 12; HardMine = 24; NoGeom = $false
        Shuffle = $false; WData = 500; WMu = 10; WWss = 10
    }
    F2_no_curriculum = @{
        Title = "Uniform sampling (pre-curriculum behavior)"
        Hypothesis = "Old uniform cohort may generalize better on mixed val"
        Epochs = 40; Adam = 34; S1 = 16; S2 = 24
        L0L1Only = 0; HardMine = 4; NoGeom = $true
        Shuffle = $false; WData = 500; WMu = 10; WWss = 10
    }
    F3_l1_warm_mining = @{
        Title = "L0/L1-only 10 ep + early hard mining @12"
        Hypothesis = "Anchor hard negatives while L1 still dominates val"
        Epochs = 48; Adam = 41; S1 = 19; S2 = 29
        L0L1Only = 10; HardMine = 12; NoGeom = $false
        Shuffle = $false; WData = 500; WMu = 10; WWss = 10
    }
    S0_shuffle_full = @{
        Title = "Shuffled graph load order (no limit-data)"
        Hypothesis = "Avoid sorted-prefix bias from vessel_*.pt lex order"
        Epochs = 40; Adam = 34; S1 = 17; S2 = 25
        L0L1Only = 6; HardMine = 16; NoGeom = $false
        Shuffle = $true; WData = 500; WMu = 10; WWss = 10
    }
    H0_data_heavy = @{
        Title = "Higher data + WSS weights (800 / 15)"
        Hypothesis = "Pull velocity fit harder vs mu/PDE (Apr used strong data_u)"
        Epochs = 38; Adam = 32; S1 = 15; S2 = 23
        L0L1Only = 6; HardMine = 16; NoGeom = $false
        Shuffle = $false; WData = 800; WMu = 10; WWss = 15
    }
    H1_low_mu = @{
        Title = "Lower mu supervision weight (5)"
        Hypothesis = "Reduce early mu overfit on physics-only graphs"
        Epochs = 38; Adam = 32; S1 = 15; S2 = 23
        L0L1Only = 6; HardMine = 16; NoGeom = $false
        Shuffle = $false; WData = 500; WMu = 5; WWss = 10
    }
}

$DefaultLegOrder = @(
    "A0_april_ratio",
    "F0_foundation",
    "F1_long_l0l1",
    "F2_no_curriculum",
    "F3_l1_warm_mining",
    "S0_shuffle_full",
    "H0_data_heavy",
    "H1_low_mu"
)

if ($Legs.Count -eq 0) { $Legs = @($DefaultLegOrder) }

$nGraphs = Get-GraphCount
if ($nGraphs -lt 50) {
    Write-Host "WARNING: only $nGraphs graphs in $GraphDir - run backfill / check data path." -ForegroundColor Red
}

$refTotalEp = 0
foreach ($lid in $Legs) { $refTotalEp += [int]$LegCatalog[$lid].Epochs }
$budgetEp = [int][math]::Floor(($TargetHours * 60.0 / $EstMinPerEpoch) * 0.9)
if ($budgetEp -lt 8) { $budgetEp = 8 }

$LegEpochs = @{}
$scaledTotal = 0
foreach ($lid in $Legs) {
    $ref = [int]$LegCatalog[$lid].Epochs
    $ep = [int][math]::Max(4, [math]::Round($budgetEp * $ref / [double]$refTotalEp))
    $LegEpochs[$lid] = $ep
    $scaledTotal += $ep
}
$estHours = ($scaledTotal * $EstMinPerEpoch) / 60.0

Write-Host ""
Write-Host "Kinematics recovery sweep (~${TargetHours}h target)" -ForegroundColor Cyan
Write-Host "  graphs: $nGraphs on disk; KINEMATICS_GRAPH_CAP=$($env:KINEMATICS_GRAPH_CAP)" -ForegroundColor DarkGray
Write-Host "  legs:   $($Legs.Count)  scaled_epochs=$scaledTotal (ref=$refTotalEp, budget~$budgetEp)" -ForegroundColor DarkGray
Write-Host "  est:    ~$([math]::Round($estHours,1))h @ ${EstMinPerEpoch}m/ep (P2200 + ~1800 train steps/ep)" -ForegroundColor DarkGray
Write-Host "  logs:   KINEMATICS_QUIET=1 (no tqdm; val + epoch summary only)" -ForegroundColor DarkGray
Write-Host "  archive: $SweepDir" -ForegroundColor DarkGray
Write-Host "  target:  val Rel L2 < 0.05 (Apr best ~0.10 @ ep 84 / 2000 graphs)" -ForegroundColor DarkGray
Write-Host ""

Add-Content -Path $SummaryPath -Value "BATCH_START host=$hostName ts=$(Get-Date -Format o) n_graphs=$nGraphs legs=$($Legs -join ',')"

$legIndex = 0
foreach ($legId in $Legs) {
    if (-not $LegCatalog.Contains($legId)) { throw "Unknown leg: $legId" }
    $def = $LegCatalog[$legId]
    $legIndex++
    $legDir = Join-Path $SweepDir $legId
    $legCkpt = Join-Path $legDir "kinematics_best.pth"
    if ((Test-Path $legCkpt) -and (-not $Force)) {
        Write-Host "SKIP $legId (kinematics_best.pth exists; use -Force)" -ForegroundColor DarkGray
        continue
    }

    Write-Host ""
    Write-Host "========== [$legIndex/$($Legs.Count)] $legId ==========" -ForegroundColor Yellow
    Write-Host "  $($def.Title)" -ForegroundColor DarkGray

    Clear-KinematicsSweepEnv
    $env:KINEMATICS_GRAPH_CAP = "2000"
    $env:KINEMATICS_QUIET = "1"
    $outDir = Join-Path $legDir "kinematics_out"
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null
    $env:KINEMATICS_OUTPUT_DIR = $outDir

    $sched = Get-ScaledLegSchedule -Def $def -LegEpochs $LegEpochs[$legId]
    Write-Host "  schedule: epochs=$($sched.Epochs) adam=$($sched.Adam) s1=$($sched.S1) s2=$($sched.S2) l0l1=$($sched.L0L1Only)" -ForegroundColor DarkGray

    $cmd = @(
        "-m", "src.training.train_kinematics_predictor",
        "--fresh", "--no-prompt", "--quiet",
        "--epochs", "$([int]$sched.Epochs)",
        "--adam-epochs", "$([int]$sched.Adam)",
        "--stage1-end-epoch", "$([int]$sched.S1)",
        "--stage2-end-epoch", "$([int]$sched.S2)",
        "--l0l1-only-epochs", "$([int]$sched.L0L1Only)",
        "--hard-mining-start-epoch", "$([int]$sched.HardMine)",
        "--weight-data", "$([double]$def.WData)",
        "--weight-mu", "$([double]$def.WMu)",
        "--weight-wss", "$([double]$def.WWss)"
    )
    if ($def.NoGeom) { $cmd += "--no-geometry-curriculum" }
    else { $cmd += @("--geometry-phase", "auto") }
    if ($def.Shuffle) { $cmd += @("--shuffle-graphs", "--graph-load-seed", "42") }
    if ($ExtraArgs.Count -gt 0) { $cmd += $ExtraArgs }

    if ($DryRun) {
        Write-Host "DryRun: python $($cmd -join ' ')" -ForegroundColor DarkGray
        continue
    }

    $t0 = Get-Date
    $code = Invoke-TrainPython -Cmd $cmd
    if ($code -ne 0) { throw "FAIL leg=$legId exit=$code" }
    Save-LegArtifacts -LegId $legId -Title $def.Title -Hypothesis $def.Hypothesis -GraphCount $nGraphs
    $mins = [int]((Get-Date) - $t0).TotalMinutes
    Write-Host "OK $legId ${mins}m" -ForegroundColor Green
    python -c "import gc; gc.collect(); import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>$null
    Clear-KinematicsSweepEnv
}

Write-Host ""
Write-Host "Sweep complete | manifest: $ManifestPath" -ForegroundColor Green
Write-Host @"
Leaderboard:
  Get-Content '$ManifestPath' | % { `$_ | ConvertFrom-Json } |
    Sort-Object { [double]`$_.best_rel_l2 } |
    Format-Table leg_id, best_rel_l2, best_l0, best_l1, best_l2, best_epoch, n_graphs
"@ -ForegroundColor DarkGray
