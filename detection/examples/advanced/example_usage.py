#!/usr/bin/env python3
"""
Example usage of the 3D Anomaly Detection Pipeline
==================================================

This script demonstrates how to use the complete pipeline for 3D anomaly detection.
It includes data loading, training, inference, and evaluation.
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple

# Import pipeline components
from pipeline import create_simple_pipeline, create_resnet_pipeline, AnomalyDetectionPipeline
from data_loader import load_and_normalize_volume, load_dataset
from evaluation import plot_roc_curves, plot_precision_recall_curves, plot_score_distributions
from thresholding import multi_threshold_analysis, visualize_threshold_analysis


def generate_synthetic_data(num_volumes: int = 10, 
                          volume_shape: Tuple[int, int, int] = (64, 64, 64),
                          anomaly_ratio: float = 0.3) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Generate synthetic 3D volumes for demonstration.
    
    Args:
        num_volumes: Number of volumes to generate
        volume_shape: Shape of each volume (D, H, W)
        anomaly_ratio: Ratio of volumes with anomalies
    
    Returns:
        Tuple of (volumes, ground_truth_masks)
    """
    volumes = []
    masks = []
    
    for i in range(num_volumes):
        # Generate base volume with noise
        volume = torch.randn(1, *volume_shape) * 0.1
        
        # Add anatomical-like structures
        d, h, w = volume_shape
        center_d, center_h, center_w = d // 2, h // 2, w // 2
        
        # Add some Gaussian blobs to simulate anatomy
        for _ in range(3):
            blob_d = np.random.randint(center_d - 10, center_d + 10)
            blob_h = np.random.randint(center_h - 10, center_h + 10)
            blob_w = np.random.randint(center_w - 10, center_w + 10)
            
            sigma = np.random.uniform(3, 8)
            intensity = np.random.uniform(0.5, 1.0)
            
            # Create coordinate grids
            d_coords, h_coords, w_coords = torch.meshgrid(
                torch.arange(d, dtype=torch.float32),
                torch.arange(h, dtype=torch.float32),
                torch.arange(w, dtype=torch.float32),
                indexing='ij'
            )
            
            # Add Gaussian blob
            blob = intensity * torch.exp(-((d_coords - blob_d)**2 + 
                                        (h_coords - blob_h)**2 + 
                                        (w_coords - blob_w)**2) / (2 * sigma**2))
            
            volume[0] += blob
        
        # Create anomaly mask
        mask = torch.zeros(volume_shape)
        
        # Add anomalies to some volumes
        if i < int(num_volumes * anomaly_ratio):
            # Add small spherical anomalies
            num_anomalies = np.random.randint(1, 4)
            
            for _ in range(num_anomalies):
                anom_d = np.random.randint(10, d - 10)
                anom_h = np.random.randint(10, h - 10)
                anom_w = np.random.randint(10, w - 10)
                anom_radius = np.random.uniform(3, 8)
                anom_intensity = np.random.uniform(0.8, 1.5)
                
                # Create spherical anomaly
                d_coords, h_coords, w_coords = torch.meshgrid(
                    torch.arange(d, dtype=torch.float32),
                    torch.arange(h, dtype=torch.float32),
                    torch.arange(w, dtype=torch.float32),
                    indexing='ij'
                )
                
                distance = torch.sqrt((d_coords - anom_d)**2 + 
                                    (h_coords - anom_h)**2 + 
                                    (w_coords - anom_w)**2)
                
                anomaly_mask = distance <= anom_radius
                
                # Add to volume and mask
                volume[0][anomaly_mask] += anom_intensity
                mask[anomaly_mask] = 1.0
        
        volumes.append(volume)
        masks.append(mask)
    
    return volumes, masks


def demonstrate_simple_pipeline():
    """Demonstrate the simple pipeline on synthetic data."""
    print("=" * 60)
    print("Demonstrating Simple 3D Anomaly Detection Pipeline")
    print("=" * 60)
    
    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    train_volumes, _ = generate_synthetic_data(num_volumes=20, anomaly_ratio=0.0)  # Normal only
    test_volumes, test_masks = generate_synthetic_data(num_volumes=10, anomaly_ratio=0.5)
    
    print(f"   - Training volumes: {len(train_volumes)} (normal only)")
    print(f"   - Test volumes: {len(test_volumes)} (50% with anomalies)")
    
    # Create and configure pipeline
    print("\n2. Creating pipeline...")
    pipeline = create_simple_pipeline(
        patch_size=(8, 8, 8),
        hidden_dims=[128, 64],
        latent_dim=32
    )
    
    print(f"   - Pipeline: {pipeline}")
    
    # Fit pipeline on normal training data
    print("\n3. Fitting pipeline on training data...")
    pipeline.fit(
        train_volumes[:15],  # Training set
        val_volumes=train_volumes[15:],  # Validation set
        threshold_method='percentile',
        threshold_percentile=95.0
    )
    
    print(f"   - Fitted successfully with threshold: {pipeline.get_threshold():.6f}")
    
    # Test inference on single volume
    print("\n4. Testing inference on single volume...")
    test_volume = test_volumes[0]
    
    # Get anomaly score map
    score_map, metadata = pipeline.score_map(test_volume)
    print(f"   - Score map shape: {score_map.shape}")
    print(f"   - Score range: [{score_map.min():.4f}, {score_map.max():.4f}]")
    print(f"   - Number of patches processed: {metadata['num_patches']}")
    
    # Get binary prediction
    binary_prediction = pipeline.predict(test_volume)
    print(f"   - Anomaly pixels detected: {binary_prediction.sum().item()}")
    
    # Evaluate on test set
    print("\n5. Evaluating on test set...")
    metrics = pipeline.evaluate(test_volumes, test_masks)
    
    print("   - Pixel-level metrics:")
    print(f"     * AUC: {metrics['pixel_auc']:.4f}")
    print(f"     * PR-AUC: {metrics['pixel_pr_auc']:.4f}")
    print(f"     * IoU: {metrics['pixel_iou']:.4f}")
    print(f"     * Dice: {metrics['pixel_dice']:.4f}")
    print(f"     * F1: {metrics['pixel_f1']:.4f}")
    
    if 'volume_auc' in metrics:
        print("   - Volume-level metrics:")
        print(f"     * Volume AUC: {metrics['volume_auc']:.4f}")
        print(f"     * Volume F1: {metrics['volume_f1']:.4f}")
    
    # Save pipeline
    print("\n6. Saving pipeline...")
    save_path = Path("./saved_pipeline")
    pipeline.save(save_path)
    print(f"   - Pipeline saved to: {save_path}")
    
    # Load pipeline
    print("\n7. Loading pipeline...")
    loaded_pipeline = AnomalyDetectionPipeline.load(save_path)
    print(f"   - Loaded pipeline: {loaded_pipeline}")
    
    # Verify loaded pipeline works
    test_score_map, _ = loaded_pipeline.score_map(test_volume)
    print(f"   - Loaded pipeline score map matches: {torch.allclose(score_map, test_score_map)}")
    
    return pipeline, test_volumes, test_masks, metrics


def demonstrate_threshold_analysis(pipeline, test_volumes, test_masks):
    """Demonstrate threshold analysis."""
    print("\n" + "=" * 60)
    print("Demonstrating Threshold Analysis")
    print("=" * 60)
    
    # Get score maps for test volumes
    print("\n1. Generating score maps...")
    score_maps = []
    for i, volume in enumerate(test_volumes):
        score_map, _ = pipeline.score_map(volume)
        score_maps.append(score_map)
        print(f"   - Processed volume {i+1}/{len(test_volumes)}")
    
    # Combine all scores for analysis
    all_scores = torch.cat([score.flatten() for score in score_maps])
    all_masks = torch.cat([mask.flatten() for mask in test_masks])
    
    # Threshold analysis
    print("\n2. Performing threshold analysis...")
    combined_score_map = torch.stack(score_maps).mean(dim=0)  # Average score map
    
    threshold_range = (all_scores.min().item(), all_scores.max().item())
    analysis_results = multi_threshold_analysis(
        combined_score_map,
        threshold_range=threshold_range,
        num_thresholds=20
    )
    
    print(f"   - Analyzed {len(analysis_results['thresholds'])} thresholds")
    print(f"   - Threshold range: [{threshold_range[0]:.4f}, {threshold_range[1]:.4f}]")
    
    # Visualize threshold analysis
    print("\n3. Visualizing threshold analysis...")
    try:
        visualize_threshold_analysis(analysis_results, save_path="threshold_analysis.png")
        print("   - Threshold analysis plot saved as 'threshold_analysis.png'")
    except Exception as e:
        print(f"   - Could not create visualization: {e}")
    
    return analysis_results


def demonstrate_evaluation_plots(test_volumes, test_masks, pipeline):
    """Demonstrate evaluation visualizations."""
    print("\n" + "=" * 60)
    print("Demonstrating Evaluation Visualizations")
    print("=" * 60)
    
    # Get predictions and scores
    print("\n1. Computing predictions and scores...")
    all_scores = []
    all_masks = []
    
    for volume, mask in zip(test_volumes, test_masks):
        score_map, _ = pipeline.score_map(volume)
        all_scores.append(score_map.flatten())
        all_masks.append(mask.flatten())
    
    combined_scores = torch.cat(all_scores)
    combined_masks = torch.cat(all_masks)
    
    # Separate normal and anomaly scores
    normal_scores = combined_scores[combined_masks == 0]
    anomaly_scores = combined_scores[combined_masks == 1]
    
    print(f"   - Normal scores: {len(normal_scores)}")
    print(f"   - Anomaly scores: {len(anomaly_scores)}")
    
    # Plot score distributions
    print("\n2. Plotting score distributions...")
    try:
        plot_score_distributions(
            normal_scores,
            anomaly_scores,
            threshold=pipeline.get_threshold(),
            save_path="score_distributions.png"
        )
        print("   - Score distribution plot saved as 'score_distributions.png'")
    except Exception as e:
        print(f"   - Could not create score distribution plot: {e}")
    
    # Plot ROC curve
    print("\n3. Plotting ROC curve...")
    try:
        results = [("Pipeline", combined_masks, combined_scores)]
        plot_roc_curves(results, save_path="roc_curve.png")
        print("   - ROC curve plot saved as 'roc_curve.png'")
    except Exception as e:
        print(f"   - Could not create ROC curve plot: {e}")
    
    # Plot Precision-Recall curve
    print("\n4. Plotting Precision-Recall curve...")
    try:
        plot_precision_recall_curves(results, save_path="pr_curve.png")
        print("   - PR curve plot saved as 'pr_curve.png'")
    except Exception as e:
        print(f"   - Could not create PR curve plot: {e}")


def main():
    """Run the complete demonstration."""
    print("3D Anomaly Detection Pipeline Demonstration")
    print("==========================================")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # Run simple pipeline demonstration
        pipeline, test_volumes, test_masks, metrics = demonstrate_simple_pipeline()
        
        # Run threshold analysis
        analysis_results = demonstrate_threshold_analysis(pipeline, test_volumes, test_masks)
        
        # Run evaluation visualizations
        demonstrate_evaluation_plots(test_volumes, test_masks, pipeline)
        
        print("\n" + "=" * 60)
        print("Demonstration completed successfully!")
        print("=" * 60)
        print("\nFiles generated:")
        print("- saved_pipeline/: Complete pipeline save")
        print("- threshold_analysis.png: Threshold analysis plots")
        print("- score_distributions.png: Score distribution plots")
        print("- roc_curve.png: ROC curve")
        print("- pr_curve.png: Precision-Recall curve")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
