# Install PyTorch with CUDA so Tier 3 training can use the GPU.
# Using --index-url (not --extra-index-url) forces the cu124 wheel; otherwise pip
# often picks a CPU-only build from PyPI (e.g. 2.x.x+cpu).
# Requires: NVIDIA driver (nvidia-smi).

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "Installing PyTorch with CUDA 12.4 (cu124) from download.pytorch.org..."
py -3 -m pip install torch --upgrade --index-url https://download.pytorch.org/whl/cu124

Write-Host ""
py -3 -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('torch.version.cuda', torch.version.cuda)"
