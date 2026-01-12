# RSNA Continual Learning with Medical Images

Continual learning implementation for RSNA medical images demonstrating catastrophic forgetting and mitigation methods. Ready for Kaggle with preprocessed dataset.

## 🎯 Overview

This project demonstrates continual learning on medical images using the RSNA 2023 Abdominal Trauma dataset. It implements four methods:

- **Baseline**: Naive fine-tuning (demonstrates catastrophic forgetting)
- **EWC**: Elastic Weight Consolidation (parameter regularization)
- **Experience Replay**: Rehearsal-based approach
- **EWC + Replay**: Combined method (best performance)

## 📁 Project Structure

```
MedContinualLearning/
├── rsna_continual_learning_kaggle.ipynb   # Main Kaggle-ready notebook
├── quickstart.py                          # CLI interface
├── config.py                              # Configuration
├── utils.py                               # Utility functions
├── requirements.txt                       # Dependencies
└── outputs/                               # Auto-created outputs
```

## 🚀 Quick Start (Kaggle)

### 1. Install Dependencies

```python
!pip install kagglehub torch torchvision tqdm matplotlib seaborn scikit-learn opencv-python
```

### 2. Download Dataset

```python
import kagglehub

path = kagglehub.dataset_download(
    "ashery/rsna-2023-abdominal-trauma-processed-dataset"
)
print(f"Dataset path: {path}")
```

### 3. Run Notebook

Open `rsna_continual_learning_kaggle.ipynb` and run all cells. The dataset is already preprocessed as PNGs.

## 💻 Local Setup

### Installation

```bash
# Clone/download repository
cd MedContinualLearning

# Install dependencies
pip install -r requirements.txt
```

### Using Kaggle Dataset Locally

```python
from utils import resolve_data_root

data_root = resolve_data_root(
    kaggle_dataset="ashery/rsna-2023-abdominal-trauma-processed-dataset"
)
```

### Using Local Dataset

```python
data_root = "/path/to/your/dataset"
```

Expected structure:
```
dataset_root/
├── train/
│   ├── class_0/
│   │   ├── image1.png
│   │   └── ...
│   └── class_1/
│       └── ...
└── test/
    ├── class_0/
    └── class_1/
```

### CLI Usage

```bash
# Using Kaggle dataset
python quickstart.py \
    --kaggle_dataset ashery/rsna-2023-abdominal-trauma-processed-dataset \
    --method ewc \
    --preset debug

# Using local data
python quickstart.py \
    --data_root /path/to/data \
    --method replay \
    --iters 1000
```

## 📊 Methods Explained

### Baseline (Naive Fine-tuning)
Sequential training without forgetting prevention.
- **Forgetting**: 20-40% accuracy drop
- **Memory**: None
- **Use**: Baseline comparison

### EWC (Elastic Weight Consolidation)
Protects important parameters using Fisher Information Matrix.

**Formula**: Loss = Task Loss + (λ/2) Σ F_i(θ_i - θ_i*)²

- **Forgetting**: 5-15%
- **Memory**: Low (only Fisher matrix)
- **Hyperparameter**: λ = 5000 (regularization strength)

### Experience Replay
Stores representative samples from previous tasks.
- **Forgetting**: 2-10%
- **Memory**: Buffer (50 samples/class default)
- **Privacy**: Stores raw data

### EWC + Replay (Combined)
Combines both approaches for best results.
- **Forgetting**: 0-5%
- **Memory**: Buffer + Fisher matrix
- **Performance**: Best overall

## ⚙️ Configuration

Edit `config.py` or pass command-line arguments:

### Key Parameters

```python
# Data
DATA_CONFIG['kaggle_dataset'] = "ashery/rsna-2023-abdominal-trauma-processed-dataset"
DATA_CONFIG['file_extension'] = 'png'
DATA_CONFIG['target_size'] = (256, 256)

# Training
TRAINING_CONFIG['learning_rate'] = 0.001
TRAINING_CONFIG['batch_size'] = 16
TRAINING_CONFIG['iters_per_task'] = 500

# Continual Learning
CONTINUAL_LEARNING_CONFIG['num_tasks'] = 2
CONTINUAL_LEARNING_CONFIG['ewc']['lambda'] = 5000.0
CONTINUAL_LEARNING_CONFIG['replay']['buffer_size_per_class'] = 50
```

### Presets

```python
from config import get_config

# Fast debugging (50 iters, small buffer)
config = get_config(preset='debug')

# High quality (2000 iters, large buffer)
config = get_config(preset='highquality')

# Memory efficient (small batch, small buffer)
config = get_config(preset='memory_efficient')
```

## 📈 Expected Results

| Method | Avg Accuracy | Forgetting |
|--------|--------------|------------|
| Baseline | 60-70% | 25-35% |
| EWC | 75-85% | 8-15% |
| Replay | 80-90% | 3-8% |
| EWC+Replay | 85-95% | 0-5% |

## 🔧 Hyperparameter Tuning

### EWC Lambda

```
Too low (<100):      Still significant forgetting
Recommended (5000):  Good balance
Too high (>50000):   Poor new task learning
```

Search: [100, 500, 1000, 5000, 10000]

### Buffer Size

```
Too small (<10):     Limited protection
Recommended (50):    Good balance
Too large (>500):    High memory, slow
```

Adjust based on dataset size and available memory.

## 📝 Key Metrics

### Average Accuracy
Mean accuracy across all tasks after training: `(1/T) Σ A_T,t`

### Forgetting
Average drop in accuracy on previous tasks: `(1/T-1) Σ (max A_i,t - A_T,t)`

### Forward Transfer
Zero-shot performance on future tasks (positive = good generalization)

### Backward Transfer
Effect of new tasks on old tasks (negative = forgetting)

## 🔬 Research Quality Features

✅ Reproducible (fixed random seeds)
✅ Proper evaluation metrics
✅ Multiple baselines
✅ Comprehensive documentation
✅ Modular and extensible
✅ Visualization tools

## 🐛 Troubleshooting

### CUDA Out of Memory

```python
# Reduce batch size
config['training']['batch_size'] = 8

# Or use CPU
config['training']['device'] = 'cpu'
```

### Dataset Not Found

```python
# Verify path
import os
print(os.listdir(data_root))

# Check structure
print(os.listdir(os.path.join(data_root, 'train')))
```

### Slow Training

```python
# Use GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Increase num_workers (local only, not Kaggle)
DataLoader(dataset, num_workers=4)
```

## 📚 References

### Continual Learning

1. **EWC**: Kirkpatrick et al. (2017). "Overcoming catastrophic forgetting in neural networks." PNAS.
   - Introduces Fisher Information-based parameter protection

2. **Online EWC**: Schwarz et al. (2018). "Progress & Compress: A scalable framework for continual learning." ICML.
   - Improves EWC with running averages

3. **Survey**: van de Ven & Tolias (2019). "Three scenarios for continual learning." arXiv:1904.07734.
   - Comprehensive overview of continual learning approaches

### Medical Imaging

4. **RSNA Dataset**: RSNA 2023 Abdominal Trauma Detection
   - Preprocessed CT images for trauma detection

## 🎓 Learning Resources

### Understanding the Code

1. **Start here**: Open `rsna_continual_learning_kaggle.ipynb`
2. **Run experiments**: Execute cells sequentially
3. **Modify hyperparameters**: Edit Section 9
4. **Visualize results**: See Section 11

### Extending the Project

```python
# Add new CL method
def train_with_new_method(model, dataset, ...):
    # Your implementation
    pass

# Add to experiments
model_new = MedicalImageClassifier(...)
train_with_new_method(model_new, ...)
```

### Custom Datasets

```python
# Modify load_rsna_data() function
def load_custom_data(data_root):
    # Load your image paths and labels
    return train_paths, train_labels, test_paths, test_labels
```

## 🤝 Contributing

Improvements welcome! Focus areas:
- Additional continual learning methods (LwF, iCaRL, PackNet)
- Better memory selection strategies
- Advanced architectures (ResNet, EfficientNet)
- Multi-task evaluation metrics

## 📄 License

MIT License - Feel free to use for research and education.

## 📧 Contact

For questions or issues:
- Check code comments in notebook
- Review configuration in `config.py`
- Inspect utilities in `utils.py`

## 🎯 Quick Reference

### Import from Utils

```python
from utils import (
    set_seed,                  # Reproducibility
    resolve_data_root,         # Kaggle/local data
    save_checkpoint,           # Model saving
    ExperimentLogger,          # Logging
    calculate_metrics,         # CL metrics
    get_device                 # GPU/CPU
)
```

### Load Configuration

```python
from config import get_config, print_config

config = get_config(preset='debug')
print_config(config)
```

### Run Experiment

```python
# In notebook
1. Set DATA_ROOT to dataset path
2. Run all cells in order
3. View results in Section 11

# CLI
python quickstart.py --kaggle_dataset <dataset-id> --method ewc
```

## ✨ Key Features

- 🎯 **Kaggle-Ready**: Works out-of-box with kagglehub
- 📊 **Four Methods**: Baseline, EWC, Replay, Combined
- 🔬 **Research-Quality**: Proper metrics and evaluation
- 📈 **Visualizations**: Accuracy plots, forgetting matrices
- ⚙️ **Configurable**: Easy hyperparameter tuning
- 📝 **Well-Documented**: Inline comments and explanations
- 🚀 **Fast**: Optimized for quick experiments

---

**Ready to run?** Open `rsna_continual_learning_kaggle.ipynb` and start experimenting! 🚀
