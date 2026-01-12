# Local Setup Guide

Quick guide to run the project on your local machine.

## 🚀 Quick Start (5 minutes)

### 1. Install Dependencies

```bash
cd /Users/ismaila/Documents/E-Others/AI_Dev/MedContinualLearning

# Install requirements
pip install -r requirements.txt

# Or with conda
conda create -n rsna_cl python=3.9
conda activate rsna_cl
pip install -r requirements.txt
```

### 2. Download Dataset

```python
# Open Python/IPython
import kagglehub

# Download dataset
path = kagglehub.dataset_download(
    "ashery/rsna-2023-abdominal-trauma-processed-dataset"
)
print(f"Dataset downloaded to: {path}")
# Outputs something like: /Users/yourusername/.cache/kagglehub/datasets/...
```

### 3. Run Notebook

```bash
# Start Jupyter
jupyter lab

# Open: rsna_continual_learning_kaggle.ipynb
# Run all cells
```

## 💻 Alternative: Run from Python Script

If you prefer scripts over notebooks:

```python
# run_experiment.py
import torch
from pathlib import Path

# Import from project files
from config import get_config
from utils import set_seed, resolve_data_root, get_device

# Setup
set_seed(42)
device = get_device()

# Get dataset
data_root = resolve_data_root(
    kaggle_dataset="ashery/rsna-2023-abdominal-trauma-processed-dataset"
)

print(f"Dataset at: {data_root}")
print(f"Device: {device}")

# Now load your data and run experiments
# (Copy code from notebook sections as needed)
```

Run it:
```bash
python run_experiment.py
```

## 🔧 Configuration

### Option 1: Edit config.py

```python
# config.py
DATA_CONFIG = {
    'kaggle_dataset': 'ashery/rsna-2023-abdominal-trauma-processed-dataset',
    'file_extension': 'png',
    'target_size': (256, 256),
}

# For faster testing
TRAINING_CONFIG['iters_per_task'] = 100  # Reduce iterations
TRAINING_CONFIG['batch_size'] = 16
```

### Option 2: Use Presets

```python
from config import get_config

# Fast testing
config = get_config(preset='debug')

# Quality results
config = get_config(preset='highquality')
```

## 🎯 Using CLI

```bash
# Quick test
python quickstart.py \
    --kaggle_dataset ashery/rsna-2023-abdominal-trauma-processed-dataset \
    --method ewc \
    --preset debug

# Full run
python quickstart.py \
    --kaggle_dataset ashery/rsna-2023-abdominal-trauma-processed-dataset \
    --method combined \
    --iters 1000 \
    --batch_size 16
```

## 📊 Expected Output

```
Using device: cuda
Dataset downloaded to: /Users/.../.cache/kagglehub/...
Loaded 1234 training images
Loaded 567 test images
Number of classes: 2

Task 1: Classes [0], 617 samples
Task 2: Classes [1], 617 samples

Training on Task 1...
100%|██████████| 500/500 [02:15<00:00,  3.69it/s, loss=0.3245]

Evaluating after Task 1:
  Task 1: 87.34%
  Task 2: 52.11%

Training on Task 2...
...
```

## 🐛 Troubleshooting

### Can't import kagglehub

```bash
pip install kagglehub
```

### CUDA Out of Memory

```python
# In config.py or notebook
TRAINING_CONFIG['batch_size'] = 8  # Reduce
TRAINING_CONFIG['device'] = 'cpu'  # Or use CPU
```

### Dataset Already Downloaded

kagglehub caches datasets. To find location:

```python
import kagglehub
path = kagglehub.dataset_download("ashery/rsna-2023-abdominal-trauma-processed-dataset")
print(path)  # Will use cached version if available
```

### Slow Training

```bash
# Use GPU
nvidia-smi  # Check GPU available

# Reduce iterations for testing
python quickstart.py --preset debug  # Only 50 iters
```

## 📂 Project Structure After Setup

```
MedContinualLearning/
├── rsna_continual_learning_kaggle.ipynb   # Main notebook
├── config.py                               # Configuration
├── utils.py                                # Utilities
├── quickstart.py                          # CLI tool
├── requirements.txt                        # Dependencies
├── outputs/                               # Created automatically
│   ├── checkpoints/
│   ├── figures/
│   ├── results/
│   └── logs/
└── ~/.cache/kagglehub/                    # Dataset cache (auto)
```

## 🔬 Development Workflow

### 1. Quick Test (5 min)
```bash
# Test if everything works
python quickstart.py --preset debug --method baseline
```

### 2. Single Method (15-30 min)
```bash
# Run one method fully
python quickstart.py --method ewc --iters 500
```

### 3. Full Comparison (1-2 hours)
```bash
# Run all methods
for method in baseline ewc replay combined; do
    python quickstart.py --method $method --iters 500
done
```

### 4. Notebook Exploration
```bash
# Interactive analysis
jupyter lab rsna_continual_learning_kaggle.ipynb
```

## 📈 Checking Results

Results are saved in `outputs/`:

```python
# Load results
import json

with open('outputs/results/experiment_ewc.json', 'r') as f:
    results = json.load(f)

print(results['accuracy_matrix'])
```

## 🎓 Learning Path

1. **Start**: Run notebook with debug preset (50 iters)
2. **Understand**: Read through notebook sections
3. **Modify**: Change hyperparameters in Section 9
4. **Compare**: Run all 4 methods
5. **Analyze**: Study forgetting matrices
6. **Extend**: Add your own methods

## ⚡ Performance Tips

### Faster Training
- Use GPU (10-20x faster than CPU)
- Increase batch size (if GPU memory allows)
- Use `num_workers=4` in DataLoader (local only)

### Better Results
- Increase `ITERS_PER_TASK`
- Increase `FISHER_SAMPLES` for EWC
- Increase `BUFFER_SIZE_PER_CLASS` for Replay
- Try different `EWC_LAMBDA` values

### Memory Optimization
- Reduce batch size
- Reduce buffer size
- Use mixed precision training
- Clear GPU cache: `torch.cuda.empty_cache()`

## 🔗 Next Steps

1. ✅ Setup complete
2. → Run notebook with debug preset
3. → Understand catastrophic forgetting
4. → Try different methods
5. → Compare results
6. → Adjust hyperparameters
7. → Upload to Kaggle (when ready)

---

**Ready to run!** Just follow steps 1-3 above. 🚀
