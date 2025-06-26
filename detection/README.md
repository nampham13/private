# 🧠 3D Anomaly Detection Pipeline

A comprehensive, user-friendly framework for 3D medical image anomaly detection using deep learning.

## 📁 Project Structure

```
detection/
├── 📦 src/                          # Source code
│   ├── 🔧 core/                     # Core pipeline functionality
│   │   ├── pipeline.py              # Main pipeline class
│   │   ├── data_loader.py           # Data loading & preprocessing
│   │   ├── patch_extraction.py      # 3D patch extraction
│   │   ├── inference.py             # Inference pipeline
│   │   ├── evaluation.py            # Evaluation metrics
│   │   └── thresholding.py          # Anomaly thresholding
│   │
│   ├── 🧠 models/                   # Deep learning models
│   │   ├── feature_extractor.py     # 3D CNN architectures
│   │   ├── feature_embedding.py     # Feature embedding layers
│   │   ├── autoencoder.py           # Autoencoder variants
│   │   └── training.py              # Training utilities
│   │
│   └── 🛠️ utils/                    # Utility functions
│
├── 📚 examples/                     # Usage examples
│   ├── basic/                       # Simple examples
│   │   ├── demo_pipeline.py         # Visual demonstration
│   │   └── quick_start.py           # Quick start guide
│   └── advanced/                    # Advanced examples
│       └── example_usage.py         # Full PyTorch example
│
├── 🧪 tests/                        # Test suite
│   ├── unit/                        # Unit tests
│   └── integration/                 # Integration tests
│
├── 📖 docs/                         # Documentation
│   ├── COMPREHENSIVE_README.md      # Detailed documentation
│   ├── PYTORCH_SETUP.md            # Setup instructions
│   └── api/                         # API documentation
│
├── ⚙️ config/                       # Configuration files
│   └── requirements.txt             # Dependencies
│
└── 📊 results/                      # Generated results
    └── anomaly_detection_results.png
```

## 🚀 Quick Start

### 1. Basic Demo (No PyTorch required)
```bash
cd detection/examples/basic
python demo_pipeline.py
```

### 2. Quick Start Guide
```bash
cd detection/examples/basic
python quick_start.py
```

### 3. Full Pipeline (Requires PyTorch)
```bash
# Install PyTorch first (see docs/PYTORCH_SETUP.md)
cd detection/examples/advanced
python example_usage.py
```

## 📖 Usage Examples

### Basic Usage
```python
# Import from organized structure
from detection.src.core import AnomalyDetectionPipeline
from detection.src.core import load_and_normalize_volume

# Create pipeline
pipeline = AnomalyDetectionPipeline(
    patch_size=(8, 8, 8),
    feature_extractor='ResNet3D_18',
    autoencoder_type='vae'
)

# Load and process data
normal_volumes = [load_and_normalize_volume(path) for path in normal_paths]

# Train pipeline
pipeline.fit(normal_volumes)

# Detect anomalies
test_volume = load_and_normalize_volume('test.nii.gz')
anomaly_scores = pipeline.score_map(test_volume)
binary_mask = pipeline.predict(test_volume)
```

### Advanced Usage
```python
# Import specific models
from detection.src.models import ResNet3D, VariationalAutoencoder
from detection.src.models import train_autoencoder

# Custom model configuration
feature_extractor = ResNet3D(block_type='basic', layers=[2, 2, 2, 2])
autoencoder = VariationalAutoencoder(input_dim=512, latent_dim=64)

# Custom training
trained_model = train_autoencoder(
    autoencoder, 
    train_data, 
    epochs=100,
    early_stopping=True
)
```

## 🎯 Key Features

- **🏗️ Modular Design**: Organized into logical components
- **📚 User-Friendly**: Clear folder structure and documentation
- **🧠 Advanced Models**: 3D CNNs, VAEs, ResNet architectures
- **🔧 Easy Configuration**: Simple parameter adjustment
- **📊 Comprehensive Evaluation**: Multiple metrics and visualizations
- **🎨 Rich Examples**: From basic demos to advanced usage
- **🧪 Test Suite**: Comprehensive testing framework

## 📋 Components Overview

### Core Pipeline (`src/core/`)
1. **Pipeline**: Main orchestrator class with `fit()`, `predict()`, `score_map()`
2. **Data Loader**: NIfTI, NumPy, DICOM support with normalization
3. **Patch Extraction**: Overlapping 3D patches with coordinate tracking
4. **Inference**: End-to-end anomaly scoring pipeline
5. **Evaluation**: ROC-AUC, IoU, Dice coefficient, PR-AUC
6. **Thresholding**: Statistical and adaptive threshold methods

### Deep Learning Models (`src/models/`)
1. **Feature Extractors**: Simple CNN, ResNet3D-18/34/50
2. **Feature Embedding**: Multiple embedding strategies  
3. **Autoencoders**: Standard, VAE, Denoising, Contractive
4. **Training**: Professional training loops with callbacks

## 🛠️ Installation & Setup

See detailed instructions in:
- `docs/PYTORCH_SETUP.md` - PyTorch installation guide
- `docs/COMPREHENSIVE_README.md` - Complete documentation

## 🎉 Getting Started

1. **Explore Examples**: Start with `examples/basic/demo_pipeline.py`
2. **Read Documentation**: Check `docs/` folder for guides
3. **Run Tests**: Use `tests/` to validate installation
4. **Customize**: Modify components in `src/` for your needs

## 📞 Support

- 📖 Documentation: `docs/COMPREHENSIVE_README.md`
- 🔧 Setup Help: `docs/PYTORCH_SETUP.md`  
- 🎯 Quick Start: `examples/basic/quick_start.py`
- 🧪 Validation: `tests/integration/comprehensive_test.py`

**Your 3D anomaly detection pipeline is ready for production use!** 🚀
