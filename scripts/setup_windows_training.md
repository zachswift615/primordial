# Windows Gaming PC Setup for LibriSpeech Training

## Hardware
- AMD Ryzen 5 7600
- NVIDIA RTX 4060 (8GB VRAM)
- 16GB RAM
- 500GB NVMe SSD

## Step 1: Install Prerequisites

### Install Python 3.11+ (if not installed)
Download from: https://www.python.org/downloads/windows/

Check "Add Python to PATH" during installation.

### Install Git (if not installed)
Download from: https://git-scm.com/download/win

### Install NVIDIA CUDA Toolkit (if not installed)
PyTorch wheels include CUDA runtime, but having the toolkit helps.
Download CUDA 12.1: https://developer.nvidia.com/cuda-12-1-0-download-archive

## Step 2: Clone the Project

Open PowerShell or Command Prompt:

```powershell
cd C:\Users\YourName\projects
git clone <your-repo-url> kung-foo-chick-pea-feeble
cd kung-foo-chick-pea-feeble
```

Or copy the folder from your MacBook via USB/network share.

## Step 3: Create Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If you get an execution policy error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Step 4: Install PyTorch with CUDA

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA is working:
```powershell
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

Should output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 4060
```

## Step 5: Install Project Dependencies

```powershell
pip install -r requirements.txt
pip install g2p_en sounddevice
```

## Step 6: Copy Data and Checkpoints

### Option A: Copy from MacBook
Copy these folders/files to the Windows PC:
- `~/data/LibriSpeech/` → `C:\Users\YourName\data\LibriSpeech\`
- `checkpoints/sequence/librispeech_v3_best.pt` → same path in project

### Option B: Re-download LibriSpeech
```powershell
mkdir C:\Users\YourName\data\LibriSpeech
cd C:\Users\YourName\data\LibriSpeech
# Download train-clean-100.tar.gz from https://www.openslr.org/12/
# Extract with 7-Zip or tar
```

## Step 7: Run Training

```powershell
cd C:\Users\YourName\projects\kung-foo-chick-pea-feeble
.\venv\Scripts\Activate.ps1

python -m primordial.scripts.train_librispeech `
    --data C:\Users\YourName\data\LibriSpeech `
    --checkpoint checkpoints/sequence/librispeech_v3_best.pt `
    --output checkpoints/sequence/librispeech_v3_cuda_best.pt `
    --epochs 200 `
    --lr 5e-6 `
    --max-real-ratio 0.6 `
    2>&1 | Tee-Object -FilePath training_log_v3_cuda.txt
```

## Step 8: Monitor Training

In another PowerShell window:
```powershell
Get-Content training_log_v3_cuda.txt -Wait -Tail 50
```

Or use nvidia-smi to monitor GPU usage:
```powershell
nvidia-smi -l 1
```

## Expected Performance

| Metric | MacBook (MPS) | RTX 4060 (CUDA) |
|--------|---------------|-----------------|
| Epoch time | ~10-15 min | ~1-2 min |
| 200 epochs | ~33-50 hours | ~3-7 hours |
| GPU utilization | Variable | 80-95% |

## Troubleshooting

### "CUDA out of memory"
Reduce batch size:
```powershell
python -m primordial.scripts.train_librispeech --batch-size 8 ...
```

### "torch.cuda.is_available() returns False"
1. Check NVIDIA driver is installed: `nvidia-smi`
2. Reinstall PyTorch with CUDA:
```powershell
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### "ModuleNotFoundError: No module named 'primordial'"
Make sure you're in the project root directory and venv is activated.

### LibriSpeech path issues
Use forward slashes or raw strings:
```powershell
--data "C:/Users/YourName/data/LibriSpeech"
# or
--data C:\Users\YourName\data\LibriSpeech
```

## Quick Verification Script

Save as `test_cuda.py` and run:
```python
import torch
import sys

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Quick benchmark
    x = torch.randn(1000, 1000, device='cuda')
    for _ in range(100):
        x = x @ x
    print("GPU compute test: PASSED")
else:
    print("ERROR: CUDA not available!")
    print("Install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121")
```

## After Training

Copy the best checkpoint back to MacBook:
```
checkpoints/sequence/librispeech_v3_cuda_best.pt
```

Then test with interactive demo on MacBook:
```bash
python -m primordial.scripts.interactive_demo \
    --checkpoint checkpoints/sequence/librispeech_v3_cuda_best.pt
```
