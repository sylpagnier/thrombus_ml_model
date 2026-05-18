# Install CUDA-enabled PyTorch into the active venv (Windows).
# Requires Python 3.10–3.13 (not 3.14+). From repo root:
#   .\scripts\install_torch_cuda.ps1

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$pyVer = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$minor = [int]($pyVer.Split(".")[1])
$major = [int]($pyVer.Split(".")[0])

if ($major -ne 3 -or $minor -lt 10 -or $minor -gt 13) {
    Write-Host "This venv uses Python $pyVer. PyTorch CUDA wheels need Python 3.10–3.13." -ForegroundColor Red
    Write-Host @"

Create a 3.11 venv (example — adjust path if py launcher differs):

  py -3.11 -m venv .venv
  .\.venv\Scripts\Activate.ps1
  python -m pip install -U pip
  .\scripts\install_torch_cuda.ps1

In PyCharm: Settings → Project → Python Interpreter → Add → 3.11 → .venv
"@ -ForegroundColor Yellow
    exit 1
}

Write-Host "Python $pyVer — installing torch+cu124 ..." -ForegroundColor Cyan
python -m pip uninstall -y torch torchvision torchaudio 2>$null
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE -ne 0) {
    Write-Host "cu124 failed; trying cu121 ..." -ForegroundColor Yellow
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
}

python -c @"
import torch
print('torch', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
"@
