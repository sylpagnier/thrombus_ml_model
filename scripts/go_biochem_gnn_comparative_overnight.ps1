# Overnight comparative sweep: train remaining legs, eval all, write verdict.
#
# Default plan (GA/GB already done on disk):
#   Phase 1: train+eval S1, S3
#   Phase 2: summary GA, GB, S1, S3 (+ locked baseline in eval json)
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_biochem_gnn_comparative_overnight.ps1
#   powershell ... -TrainLegs "S1,S3" -SummaryLegs "GA,GB,S1,S3"

param(
    [string] $TrainLegs = "S1,S3",
    [string] $SummaryLegs = "GA,GB,S1,S3",
    [double] $TargetHours = 8.0,
    [int]    $Epochs = 40,
    [int]    $EarlyStop = 14,
    [switch] $Fresh
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$RunRoot = Join-Path $RepoRoot "outputs/biochem/biochem_gnn/comparative_6h"
$LogDir = Join-Path $RunRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDir "overnight_$ts.log"

function Log([string]$Msg) {
    $line = "[$(Get-Date -Format 'HH:mm:ss')] $Msg"
    Add-Content -Path $LogFile -Value $line -Encoding UTF8
    Write-Host $line
}

Log "=== comparative overnight start ==="
Log "log=$LogFile"
Log "train legs=$TrainLegs summary legs=$SummaryLegs epochs=$Epochs"

$launcher = Join-Path $PSScriptRoot "go_biochem_gnn_comparative_6h.ps1"
$freshArg = if ($Fresh) { "-Fresh" } else { "" }
$skipCompleted = if ($Fresh) { "" } else { "-SkipCompleted" }

# Phase 1: train + eval requested legs
Log "--- phase 1: train+eval ---"
$trainArgs = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $launcher,
    "-TargetHours", "$TargetHours",
    "-Legs", $TrainLegs,
    "-Epochs", "$Epochs",
    "-EarlyStop", "$EarlyStop"
)
if ($skipCompleted) { $trainArgs += $skipCompleted }
if ($freshArg) { $trainArgs += $freshArg }

& powershell @trainArgs 2>&1 | ForEach-Object { Log $_ }
if ($LASTEXITCODE -ne 0) {
    Log "[ERR] phase 1 failed exit=$LASTEXITCODE"
    exit $LASTEXITCODE
}
Log "[OK] phase 1 done"

# Phase 2: refresh eval + merged summary for all legs
Log "--- phase 2: summary all legs ---"
$sumArgs = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $launcher,
    "-Legs", $SummaryLegs,
    "-SkipTrain",
    "-SkipCompleted",
    "-SummaryAll"
)
& powershell @sumArgs 2>&1 | ForEach-Object { Log $_ }
if ($LASTEXITCODE -ne 0) {
    Log "[ERR] phase 2 failed exit=$LASTEXITCODE"
    exit $LASTEXITCODE
}

# Phase 3: human-readable verdict
Log "--- phase 3: verdict ---"
python -c @"
import json
from pathlib import Path
root = Path(r'$RepoRoot')
run = root / 'outputs/biochem/biochem_gnn/comparative_6h'
summary_path = run / 'comparative_summary.json'
out = run / 'comparative_verdict.md'
s = json.loads(summary_path.read_text(encoding='utf-8'))
legs = s.get('legs') or {}
wk = s.get('winner_key') or 'deploy_clot_score'
lines = ['# Biochem GNN comparative verdict', '', f'Winner key: {wk}', '', '## Leg scores (patient007 deploy)', '', '| Leg | deploy_clot_score | clot_f1_main | clot_guiding_main | holdout_mean_clot_f1 |', '|-----|-------------------|--------------|-------------------|----------------------|']
for leg, m in sorted(legs.items()):
    lines.append(f\"| {leg} | {float(m.get('deploy_clot_score',0)):.3f} | {float(m.get('clot_f1_main',0)):.3f} | {float(m.get('clot_guiding_main',0)):.3f} | {float(m.get('holdout_mean_clot_f1_main',0)):.3f} |\")
lines += ['', f\"Gate winner: {s.get('gate_winner')} ({float(s.get('gate_winner_score',0)):.4f})\", f\"Species winner: {s.get('species_winner')} ({float(s.get('species_winner_score',0)):.4f})\", '', 'Primary metric: deploy_frozen p007 clot F1 @ t=200 + holdout mean.']
out.write_text(chr(10).join(lines)+chr(10), encoding='utf-8')
print('[OK] wrote', out)
"@ 2>&1 | ForEach-Object { Log $_ }

Log "=== comparative overnight done ==="
Log "summary=$RunRoot/comparative_summary.json"
Log "verdict=$RunRoot/comparative_verdict.md"
