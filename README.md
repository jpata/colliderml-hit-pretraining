# Point Cloud Pretraining for Calorimeter Hits

This project implements and compares two state-of-the-art self-supervised learning (SSL) approaches for high-energy physics calorimeter hits: **Point-MAE** (reconstruction-based) and **PANDA** (distillation-based). The system treats calorimeter and tracker hits as a heterogeneous point cloud, capable of scaling to full-event contexts.

## Comparison of Pretraining Strategies

| Feature | Point-MAE (Reconstruction) | PANDA (Self-Distillation) |
| :--- | :--- | :--- |
| **Core Objective** | Reconstruct masked coordinates/features | Align student-teacher prototype distributions |
| **Backbone** | Standard Transformer (Patch-based) | Point Transformer V3 (Sparse 3D + Attention) |
| **Masking Strategy**| Random Patch Masking (FPS/KNN) | Voxel-Grid Patch Masking (Scheduled size/ratio) |
| **Representations** | Patch-level latent tokens | Per-point hierarchical multi-scale embeddings |
| **View Strategy** | Single view with random masking | Multi-view (Global crops, Local crops, Masked) |
| **Loss Function** | MSE (Coord, Energy, Density) | Cross-Entropy (Sinkhorn-Knopp centered) |

---

## 1. Point-MAE: Hierarchical Patch Tokenization

To effectively learn the local and global physics of particle showers via reconstruction:

1.  **Neighborhood Sampling:** For events with high hit counts, we sample a "seed" hit (weighted by energy) and its $K$ nearest neighbors.
2.  **Patch Tokenization:** Within the sampled context, we divide hits into irregular spatial patches using **Farthest Point Sampling (FPS)** and **K-Nearest Neighbors (KNN)**.
3.  **PointNet Embedding:** Each patch is embedded into a single token using a lightweight **PointNet**.
4.  **Asymmetric MAE:** A heavy Transformer encoder processes only visible patches, while a lightweight decoder reconstructs the coordinates, energy, and density features of masked patches.

## 2. PANDA: Self-Distillation of Sensor-Level Representations

PANDA (*Point Attention and Distillation*) targets reusable sensor-level representations through clustering:

1.  **PTv3 Backbone:** Uses a five-stage **Point Transformer V3** (hierarchical sparse 3D encoder) that operates directly on voxels across multiple scales (3mm to 48mm).
2.  **Voxel-Grid Patch Masking:** Instead of random points, we mask entire voxel-grid cubes. We use **Difficulty Scheduling**:
    *   **Patch Size:** Increases from 2.1cm (7 voxels) to 15cm (50 voxels).
    *   **Mask Ratio:** Increases from 50% to 90% over the first 5% of training.
3.  **Prototype-Based Distillation:** A Teacher network maps global views to assignments over 4096 learned prototypes. A Student network is trained to predict consistent prototype distributions across masked and local views using a Sinkhorn-Knopp centered cross-entropy loss.

---

## Optimized Preprocessing

To avoid GPU idling, the project uses an `IterableDataset` with worker-aware partitioning.
*   **Spatial Sorting:** Hits are sorted using a **Morton (Z-order) curve** to ensure spatial locality and cache efficiency.
*   **GPU-Accelerated Tokenization:** Geometric tokenization (FPS/KNN) and density computation are performed on the **GPU** during the forward pass.

## Project Structure

*   `src/dataset.py`: Multi-scale feature computation and spatially-aware sampling.
*   `src/hilbert.py`: Vectorized Morton/Hilbert indexing for spatial data sorting.
*   `train_example.py`: Point-MAE training script with hierarchical patching.
*   `scripts/train_panda.py`: PANDA training script with PTv3 and self-distillation.
*   `compute_all_representations.py`: Sliding-window inference for full-event embedding.

## Installation & Requirements

This project uses `pixi` for environment management.

```bash
# Install dependencies
pixi install
```

## Usage Instructions

### Training Point-MAE (Reconstruction)

```bash
pixi run -e local python train_example.py \
    --neighborhood True \
    --n_patches 128 \
    --k_neighbors 64 \
    --num_hits 2048 \
    --epochs 20 \
    --batch_size 64 \
    --output_dir results/mae_experiment
```

### Training PANDA (Self-Distillation)

```bash
pixi run -e local python scripts/train_panda.py \
    --num_hits 2048 \
    --epochs 10 \
    --batch_size 48 \
    --lr 2.6e-3 \
    --n_local_views 4 \
    --output_dir results/panda_pretrain
```

## Results & Plots

*   **`reconstruction_...`**: (Point-MAE) Side-by-side comparison of true hits vs model predictions.
*   **`embeddings_viz_...`**: (PANDA) 3D visualization of hits and 2D UMAP projection colored by DBSCAN clustering in the latent space.
*   **`representation_metrics_evolution.png`**: Tracking of PCA entropy, silhouette scores, and density-loss correlations.
