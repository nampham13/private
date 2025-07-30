import torch
import numpy as np
from src.core.pipeline import AnomalyDetectionPipeline
from src.models.feature_extractor import FeatureExtractor3D, Simple3DCNN

# Example configuration for pipeline components
feature_extractor_config = {
    'type': 'simple',  # or 'resnet18', 'resnet34', etc.
    'args': {
        'in_channels': 1  # Change if your data has more channels
    }
}
embedding_config = {
    'input_channels': 8,  # Match output channels of feature extractor
    'patch_size': (8, 8, 8),
    'embedding_method': 'flatten',  # or 'mlp', 'pool'
    'output_dim': 128  # Only needed for 'mlp'
}
autoencoder_config = {
    'type': 'standard',
    'args': {
        'input_dim': 8*8*8*8,  # input_channels * patch_size product
        'hidden_dims': [256, 128],
        'latent_dim': 32,
        'activation': 'relu',
        'dropout_rate': 0.1,
        'batch_norm': True
    }
}
patch_config = {
    'patch_size': (8, 8, 8),
    'stride': (4, 4, 4),
    'max_patches': 100,
    'sampling_strategy': 'uniform'
}
training_config = {
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'scheduler_type': 'step',
    'scheduler_params': {'step_size': 10, 'gamma': 0.5},
    'early_stopping_patience': 10,
    'plot_losses': True
}

def load_npz_volumes(npz_path):
    data = np.load(npz_path)
    volumes = []
    for key in data.keys():
        arr = data[key]
        if arr.ndim == 4:
            volumes.append(torch.from_numpy(arr).float())
    return volumes

if __name__ == "__main__":
    npz_path = r"d:\final_year_project\private\detection\0001_rgbd.npz"
    train_volumes = load_npz_volumes(npz_path)

    pipeline = AnomalyDetectionPipeline(
        feature_extractor_config=feature_extractor_config,
        embedding_config=embedding_config,
        autoencoder_config=autoencoder_config,
        patch_config=patch_config,
        training_config=training_config,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    pipeline.fit(train_volumes)

    # Save trained autoencoder
    torch.save(pipeline.autoencoder.state_dict(), "autoencoder_trained.pth")
    print("Training complete. Model saved as autoencoder_trained.pth.")
