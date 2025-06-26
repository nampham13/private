# 3D Anomaly Detection Pipeline

A comprehensive PyTorch-based framework for 3D medical image anomaly detection using autoencoders. This pipeline provides end-to-end functionality for building, training, and evaluating 3D anomaly detection models.

## Features

- **🧠 Multiple 3D CNN Architectures**: Simple CNN, 3D ResNet variants
- **🔍 Flexible Patch Extraction**: Multi-scale overlapping 3D patches
- **🎯 Feature Embedding**: Various methods (flattening, pooling, MLP projection)
- **🔄 Multiple Autoencoder Types**: Standard, Variational (VAE), Denoising
- **📊 Comprehensive Evaluation**: ROC-AUC, PR-AUC, IoU, Dice coefficient
- **🎚️ Adaptive Thresholding**: Otsu, Triangle, Yen methods
- **📈 Visualization Tools**: Training curves, ROC curves, score distributions
- **💾 Modular Pipeline**: Easy save/load, configurable components

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd 3d-anomaly-detection

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from pipeline import create_simple_pipeline
from data_loader import load_dataset

# Create pipeline
pipeline = create_simple_pipeline()

# Load training data (normal samples only)
train_volumes = load_dataset(train_paths, target_shape=(64, 64, 64))

# Fit pipeline
pipeline.fit(train_volumes)

# Load test data
test_volume = load_and_normalize_volume('test_scan.nii.gz', target_shape=(64, 64, 64))

# Get anomaly score map
score_map, metadata = pipeline.score_map(test_volume)

# Get binary prediction
binary_mask = pipeline.predict(test_volume)

# Evaluate on test set with ground truth
metrics = pipeline.evaluate(test_volumes, ground_truth_masks)
```

## Pipeline Components

### 1. Data Loading (`data_loader.py`)
Load and normalize 3D volumes from various formats:
- **NIfTI files** (`.nii`, `.nii.gz`)
- **NumPy arrays** (`.npy`, `.npz`)
- **Other medical formats** via SimpleITK

Features:
- Multiple normalization methods (z-score, min-max, percentile)
- Automatic resizing with trilinear interpolation
- Optional data augmentation (rotation, flipping, noise)

### 2. Patch Extraction (`patch_extraction.py`)
Extract overlapping 3D patches at multiple scales:
- Configurable patch sizes and strides
- Coordinate tracking for reconstruction
- Memory-efficient processing

### 3. Feature Extraction (`feature_extractor.py`)
3D CNN architectures for feature extraction:
- **Simple 3D CNN**: Lightweight model for quick prototyping
- **3D ResNet-18/34/50**: Deeper networks with residual connections
- **Feature map extraction**: Access intermediate layer outputs

### 4. Feature Embedding (`feature_embedding.py`)
Convert patches to fixed-length feature vectors:
- **Flattening**: Direct spatial flattening
- **Pooling**: Average/max/adaptive pooling
- **MLP Projection**: Learnable dimensionality reduction

### 5. Autoencoders (`autoencoder.py`)
Multiple autoencoder architectures:
- **Standard Autoencoder**: Basic reconstruction-based anomaly detection
- **Variational Autoencoder (VAE)**: Probabilistic latent representations
- **Denoising Autoencoder**: Robust to input noise
- **Contractive Autoencoder**: Regularized representations

### 6. Training (`training.py`)
Comprehensive training framework:
- Adam optimizer with configurable schedules
- Early stopping and model checkpointing
- Training curve visualization
- Support for validation monitoring

### 7. Inference (`inference.py`)
End-to-end inference pipeline:
- Feature extraction from test volumes
- Reconstruction error computation
- 3D anomaly score map generation
- Batch processing capabilities

### 8. Thresholding (`thresholding.py`)
Threshold computation and binary mask generation:
- **Statistical methods**: Percentile, sigma, MAD, IQR
- **Adaptive methods**: Otsu, Triangle, Yen
- **Multi-threshold analysis**: Performance across threshold ranges
- **Morphological post-processing**: Noise reduction

### 9. Evaluation (`evaluation.py`)
Comprehensive evaluation metrics:
- **Pixel-level**: ROC-AUC, PR-AUC, IoU, Dice coefficient
- **Volume-level**: Volume-wise classification metrics
- **Visualization**: ROC curves, PR curves, score distributions

### 10. Modular Pipeline (`pipeline.py`)
Complete end-to-end pipeline:
- **Easy configuration**: Pre-configured setups for common use cases
- **Save/Load functionality**: Persistent model storage
- **Batch processing**: Handle multiple volumes efficiently

## Usage Examples

### Basic Usage
```python
# Create a simple pipeline
pipeline = create_simple_pipeline(
    patch_size=(8, 8, 8),
    hidden_dims=[128, 64],
    latent_dim=32
)

# Fit on normal training data
pipeline.fit(train_volumes)

# Predict on test volume
binary_mask = pipeline.predict(test_volume)
```

### Advanced Configuration
```python
# Custom pipeline configuration
feature_config = {
    'type': 'resnet18',
    'args': {'in_channels': 1}
}

embedding_config = {
    'input_channels': 512,
    'patch_size': (4, 4, 4),
    'embedding_method': 'mlp',
    'mlp_hidden_dims': [256, 128],
    'output_dim': 64
}

autoencoder_config = {
    'type': 'variational',
    'args': {
        'input_dim': 64,
        'hidden_dims': [32, 16],
        'latent_dim': 8,
        'beta': 0.5
    }
}

pipeline = AnomalyDetectionPipeline(
    feature_extractor_config=feature_config,
    embedding_config=embedding_config,
    autoencoder_config=autoencoder_config,
    patch_config=patch_config
)
```

### Evaluation and Visualization
```python
# Comprehensive evaluation
metrics = pipeline.evaluate(test_volumes, ground_truth_masks)

# Plot evaluation curves
from evaluation import plot_roc_curves, plot_precision_recall_curves

results = [("Method", ground_truth, predictions)]
plot_roc_curves(results)
plot_precision_recall_curves(results)

# Threshold analysis
from thresholding import multi_threshold_analysis, visualize_threshold_analysis

analysis = multi_threshold_analysis(score_map, threshold_range=(0.0, 1.0))
visualize_threshold_analysis(analysis)
```

## Example Results

The pipeline provides comprehensive metrics for anomaly detection performance:

```
Pixel-level metrics:
* AUC: 0.8542
* PR-AUC: 0.7891
* IoU: 0.6234
* Dice: 0.7678
* F1: 0.7456

Volume-level metrics:
* Volume AUC: 0.9123
* Volume F1: 0.8567
```

## File Structure

```
3d-anomaly-detection/
├── requirements.txt          # Dependencies
├── __init__.py              # Package initialization
├── data_loader.py           # Data loading and preprocessing
├── patch_extraction.py      # 3D patch extraction utilities
├── feature_extractor.py     # 3D CNN architectures
├── feature_embedding.py     # Feature embedding methods
├── autoencoder.py          # Autoencoder models
├── training.py             # Training utilities
├── inference.py            # Inference pipeline
├── thresholding.py         # Thresholding methods
├── evaluation.py           # Evaluation metrics
├── pipeline.py             # Complete pipeline
├── example_usage.py        # Example usage script
└── README.md              # This file
```

## Running the Example

```bash
# Run the complete demonstration
python example_usage.py
```

This will:
1. Generate synthetic 3D volumes with anomalies
2. Train the pipeline on normal data
3. Evaluate on test data with ground truth
4. Generate visualization plots
5. Save the trained pipeline

## Supported Data Formats

- **NIfTI**: `.nii`, `.nii.gz` (via nibabel)
- **NumPy**: `.npy`, `.npz`
- **Medical formats**: DICOM, MetaImage, etc. (via SimpleITK)

## Dependencies

- PyTorch >= 2.0.0
- NumPy >= 1.21.0
- nibabel >= 5.1.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.5.0
- scipy >= 1.9.0
- SimpleITK >= 2.2.0
- tqdm >= 4.64.0

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{3d_anomaly_detection_pipeline,
  title={3D Anomaly Detection Pipeline: A Comprehensive PyTorch Framework},
  author={3D Anomaly Detection Team},
  year={2024},
  url={https://github.com/your-repo/3d-anomaly-detection}
}
```
