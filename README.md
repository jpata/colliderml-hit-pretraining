# Point Cloud Pretraining for Calorimeter Hits

This project implements a Masked Point Modeling (MPM) approach using Transformers to learn representations of high-energy physics calorimeter hits. The system is designed to scale from sparse samplings to full-event contexts containing tens of thousands of hits.

## Strategy: Hierarchical Neighborhood Encoding

To effectively learn from large context windows (10,000+ hits) while training on fixed-size buffers (e.g., 256 hits), we use a two-part strategy:

1.  **Neighborhood Training:** Instead of global random sampling, we sample a "seed" hit (weighted by energy) and its $K$ nearest neighbors. This ensures the Transformer learns the dense local physics of particle showers.
2.  **Sliding-Window Inference:** To compute representations for *all* hits in an event, we process the full hit cloud using overlapping spatial windows and average the resulting latent vectors.

## Quantitative Validation: Fidelity vs. Density

We evaluate the model's ability to use context through **Density-Conditioned Reconstruction Loss**. 
*   **Metric:** Mean Squared Error (MSE) of reconstructed masked hits binned by local hit density.
*   **Success Criterion:** A negative correlation between MSE and density, indicating the Transformer is successfully utilizing neighboring hits to "fill in" missing information in dense shower regions.

## Project Structure

*   `dataset.py`: Contains `NeighborhoodCalorimeterDataset` for spatially-aware sampling.
*   `train_example.py`: The main training script. Features:
    *   Efficient Transformer backbone (using FlashAttention via `scaled_dot_product_attention`).
    *   Automated generation of **Fidelity vs. Density** plots.
    *   Support for neighborhood-based training.
*   `compute_all_representations.py`: Utility to generate embeddings for every hit in a full event using the sliding-window strategy.
*   `visualize_scan.py`: Tools for analyzing hyperparameter scans.

## Installation & Requirements

This project uses `pixi` for environment management.

```bash
# Install dependencies
pixi install
```

## Usage Instructions

All commands should be run with the local library path set to ensure compatibility with the environment's C++ libraries:
`export LD_LIBRARY_PATH=$PWD/.pixi/envs/default/lib`

### 1. Training a Model

Train a model using the neighborhood sampling strategy:

```bash
pixi run python train_example.py \
    --num_hits 256 \
    --embed_dim 32 \
    --epochs 10 \
    --use_neighborhood \
    --output_dir results/my_experiment
```

### 2. Computing Full Event Representations

Once trained, compute embeddings for all hits in the dataset:

```bash
pixi run python compute_all_representations.py \
    --checkpoint results/my_experiment/checkpoint_h256_e32_neighTrue.pth \
    --num_hits 256 \
    --max_events 5 \
    --output_file full_embeddings.pt
```

### 3. Hyperparameter Scanning

The project supports automated parameter scans via `Snakefile`:

```bash
pixi run snakemake --cores 4
```

## Results & Plots

*   **`umap_epoch_N.png`**: UMAP visualization of the latent space.
*   **`fidelity_vs_density_epoch_N.png`**: Heatmap showing how reconstruction accuracy improves with local hit density.
*   **`full_event_embeddings.pt`**: Serialized tensor containing (Hit Coords, Latent Embedding) for downstream tasks.
