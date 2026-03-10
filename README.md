# Point Cloud Pretraining for Calorimeter Hits

This project implements a Masked Point Modeling (MPM) approach using the **Point-MAE** architecture to learn representations of high-energy physics calorimeter hits. The system treats calorimeter and tracker hits uniformly as a heterogeneous point cloud, capable of scaling to full-event contexts.

## Strategy: Hierarchical Patch Tokenization

To effectively learn the local and global physics of particle showers, we use a multi-stage hierarchical approach:

1.  **Neighborhood Sampling (Dataset):** For events with high hit counts, we sample a "seed" hit (weighted by energy) and its $K$ nearest neighbors. This ensures the model focuses on dense, physics-rich regions.
2.  **Patch Tokenization (Model):** Within the sampled context, we divide hits into irregular spatial patches using **Farthest Point Sampling (FPS)** and **K-Nearest Neighbors (KNN)**.
3.  **PointNet Embedding:** Each patch is embedded into a single token using a lightweight **PointNet** (MLPs + Max Pooling). This maintains permutation invariance and captures local geometric context.
4.  **Asymmetric Masked Autoencoder:** A heavy Transformer encoder processes only visible patches, while a lightweight decoder reconstructs the coordinates, energy, and density features of all hits within the masked patches.

## Optimized Preprocessing

To avoid GPU idling during data loading, the project uses an `IterableDataset` with worker-aware partitioning. This offloads shard loading and event-level processing to **background CPU workers** via the PyTorch `DataLoader`. Geometric tokenization (FPS/KNN) and multi-scale density computation are performed on the **GPU** during the model's forward pass to leverage parallel hardware.

## Quantitative Validation: Fidelity vs. Density

We evaluate the model's ability to use context through **Density-Conditioned Reconstruction Loss**. 
*   **Metric:** Correlation between Mean Absolute Error (MAE) and local hit density.
*   **Success Criterion:** A negative correlation indicates the Transformer is successfully utilizing neighboring hits to "fill in" missing information in dense shower regions.

## Project Structure

*   `dataset.py`: Multi-scale feature computation and spatially-aware sampling (`NeighborhoodCalorimeterDataset`).
*   `train_example.py`: Main training script with hierarchical patching and GPU-accelerated tokenization.
*   `compute_all_representations.py`: Sliding-window inference for full-event embedding.
*   `visualize_scan.py`: Analysis of hyperparameter scans.

## Installation & Requirements

This project uses `pixi` for environment management.

```bash
# Install dependencies
pixi install
```

## Usage Instructions

All commands should be run with the local library path set:
`export LD_LIBRARY_PATH=$PWD/.pixi/envs/default/lib`

### 1. Training a Model

Train with default patch-level tokenization and neighborhood sampling:

```bash
pixi run python train_example.py \
    --neighborhood True \
    --n_patches 64 \
    --k_neighbors 32 \
    --num_hits 1024 \
    --epochs 20 \
    --output_dir results/patch_experiment
```

### 2. Computing Full Event Representations

```bash
pixi run python compute_all_representations.py \
    --checkpoint results/patch_experiment/checkpoint_h1024_patches.pth \
    --num_hits 1024 \
    --embed_dim 128
```

## Results & Plots

*   **`reconstruction_epoch_{N}_ev{idx}.png`**: Side-by-side comparison of true hits (red=masked) vs model predictions (green).
*   **`point_cloud_epoch_{N}_ev{idx}.png`**: 3D visualization of hits colored by their DBSCAN clustering in the latent space.
*   **`representation_metrics_evolution.png`**: Tracking of PCA entropy, silhouette scores, and density correlations over time.
*   **`fidelity_vs_density_epoch_{N}.png`**: Heatmap showing how reconstruction accuracy scales with local hit density.
