# Windows + RTX 4050 Optimized Setup

Quick setup guide optimized for Windows PC with NVIDIA RTX 4050 GPU (6GB VRAM).

## 🎮 GPU Specifications

- **GPU**: NVIDIA RTX 4050
- **VRAM**: 6GB
- **Recommended Settings**: Mixed precision, optimized batch sizes

## 🚀 Installation

### 1. Install Python and CUDA

```powershell
# Install Python 3.9-3.10 from python.org
# Download: https://www.python.org/downloads/

# Verify Python
python --version

# Install CUDA Toolkit 11.8 (for PyTorch)
# Download: https://developer.nvidia.com/cuda-11-8-0-download-archive
```

### 2. Install PyTorch with CUDA

```powershell
# Navigate to project
cd C:\path\to\MedContinualLearning

# Install PyTorch with CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

## ⚙️ Optimized Configuration for RTX 4050

### Edit config.py

```python
# config.py - RTX 4050 Optimized Settings

TRAINING_CONFIG = {
    'learning_rate': 0.001,
    'batch_size': 12,              # Optimized for 6GB VRAM
    'iters_per_task': 500,

    # Enable mixed precision
    'use_amp': True,                # Automatic Mixed Precision

    'device': 'cuda',
    'random_seed': 42,
}

CONTINUAL_LEARNING_CONFIG = {
    'num_tasks': 2,

    'ewc': {
        'lambda': 5000.0,
        'fisher_samples': 150,      # Reduced for faster computation
        'fisher_batch_size': 1,
    },

    'replay': {
        'buffer_size_per_class': 40,  # Reduced for memory
    },

    'combined': {
        'ewc_lambda': 5000.0,
        'buffer_size_per_class': 40,
    }
}
```

## 🔧 Memory-Optimized Notebook Settings

Edit in notebook Section 9:

```python
# RTX 4050 Optimized Settings
NUM_TASKS = 2
NUM_CLASSES = 2
CLASSES_PER_TASK = NUM_CLASSES // NUM_TASKS

ITERS_PER_TASK = 500
LEARNING_RATE = 0.001
BATCH_SIZE = 12                    # Optimized for 6GB

EWC_LAMBDA = 5000.0
BUFFER_SIZE_PER_CLASS = 40         # Reduced for memory
FISHER_SAMPLES = 150               # Faster computation

# Enable mixed precision training
USE_MIXED_PRECISION = True
```

## 🚄 Enable Mixed Precision Training

Add to notebook after imports:

```python
from torch.cuda.amp import autocast, GradScaler

# Initialize scaler for mixed precision
scaler = GradScaler()
```

Update training functions:

```python
def train_baseline(model, dataset, iters, lr, batch_size, device, use_amp=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=use_amp)

    model.train()
    model.to(device)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=2, pin_memory=True)  # num_workers=2 for Windows

    iters_left = 0
    progress_bar = tqdm.tqdm(range(1, iters + 1), desc="Training")

    for batch_idx in progress_bar:
        if iters_left == 0:
            data_iter = iter(data_loader)
            iters_left = len(data_loader)

        images, labels = next(data_iter)
        images, labels = images.to(device), labels.to(device)
        iters_left -= 1

        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Mixed precision backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
```

## 📊 Performance Expectations

| Configuration | Time (RTX 4050) | Memory Usage |
|---------------|-----------------|--------------|
| Batch Size 12 | ~25-35 min | ~4.5-5GB |
| Batch Size 8 | ~35-45 min | ~3.5-4GB |
| Batch Size 16 | ~20-30 min | ~5.5-6GB (may OOM) |

## 🐛 Troubleshooting

### CUDA Out of Memory

**Option 1**: Reduce batch size
```python
BATCH_SIZE = 8  # More conservative
```

**Option 2**: Clear cache between tasks
```python
# Add after each task
torch.cuda.empty_cache()
```

**Option 3**: Reduce buffer size
```python
BUFFER_SIZE_PER_CLASS = 30
```

### Slow Data Loading on Windows

```python
# In DataLoader
DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=2,      # Windows works best with 2
    pin_memory=True,    # Faster GPU transfer
    persistent_workers=True  # Keep workers alive
)
```

### GPU Not Detected

```powershell
# Check CUDA installation
nvidia-smi

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 🎯 Running on Windows

### Method 1: Jupyter Notebook

```powershell
# Start Jupyter
jupyter lab

# Open: rsna_continual_learning_kaggle.ipynb
# Modify settings in Section 9 as shown above
# Run all cells
```

### Method 2: Command Line

```powershell
# Quick test
python quickstart.py --preset debug --method baseline

# Full run with optimized settings
python quickstart.py --method combined --batch_size 12 --iters 500
```

## 💾 Monitoring GPU

### PowerShell

```powershell
# Monitor GPU usage
nvidia-smi -l 1  # Updates every 1 second

# Or install gpustat
pip install gpustat
gpustat -cp -i 1
```

### In Python

```python
import torch

# Check memory
print(f"Allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved(0)/1024**3:.2f} GB")

# Clear cache
torch.cuda.empty_cache()
```

## 🔥 Performance Tips

### 1. Enable TensorFloat-32
```python
# Add at beginning of notebook
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### 2. Benchmark Mode
```python
# For consistent input sizes
torch.backends.cudnn.benchmark = True
```

### 3. Gradient Accumulation (if still OOM)
```python
accumulation_steps = 2
BATCH_SIZE = 6  # Effective batch size = 6 * 2 = 12

for batch_idx, (images, labels) in enumerate(data_loader):
    outputs = model(images)
    loss = criterion(outputs, labels) / accumulation_steps
    loss.backward()

    if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## 📈 Optimal Settings Summary

```python
# Best settings for RTX 4050 (6GB VRAM)
{
    'batch_size': 12,
    'num_workers': 2,
    'pin_memory': True,
    'use_amp': True,
    'buffer_size_per_class': 40,
    'fisher_samples': 150,
}
```

## ⏱️ Expected Runtime

- **Debug (50 iters)**: ~3-5 minutes
- **Default (500 iters)**: ~25-35 minutes
- **High Quality (2000 iters)**: ~90-120 minutes

All times with batch_size=12 and mixed precision.

## 🔗 Windows-Specific Paths

```python
# Dataset path (Windows style)
data_root = r"C:\Users\YourName\.cache\kagglehub\datasets\..."

# Output path
output_dir = r"C:\path\to\MedContinualLearning\outputs"
```

## ✅ Quick Start Checklist

- [ ] Install Python 3.9+
- [ ] Install CUDA Toolkit 11.8
- [ ] Install PyTorch with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
- [ ] Install requirements: `pip install -r requirements.txt`
- [ ] Verify GPU: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Update notebook settings (Section 9) with optimized values
- [ ] Run notebook with mixed precision enabled

## 🚀 Ready to Run!

```powershell
cd C:\path\to\MedContinualLearning
jupyter lab
# Open rsna_continual_learning_kaggle.ipynb
# Use optimized settings above
# Run all cells
```

---

**Optimized for your RTX 4050!** Batch size 12 with mixed precision gives best performance. 🎮
