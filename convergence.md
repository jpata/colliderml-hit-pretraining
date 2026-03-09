# Convergence Study Instructions

This document outlines the procedure for the iterative convergence study of the Point Cloud Pretraining model.

## Goal
Achieve convergence of the validation loss for the Masked Point Modeling task.

## Current Strategy
*   **Base Model:** Transformer encoder with PointNet hit embeddings.
*   **Task:** Masked Point Modeling (reconstruct x, y, z, e of 75% masked hits).
*   **Loss:** SmoothL1Loss on masked hits.
*   **Optimization:** Adam with Cosine Annealing, LR=3e-4.
*   **Data:** `NeighborhoodCalorimeterDataset` (**neighborhood=True**) with **256 hits** (initial) and 2000 events. This ensures that the model primarily uses local information (spatial correlations within a shower) to reconstruct masked hits.
*   **Additional Metrics:** Monitor reconstruction fidelity vs density, 3D embedding clusters, and separate coordinate vs energy loss.

## Energy Reconstruction Improvement Strategies
Recent iterations identified energy reconstruction as a primary bottleneck. Consider the following:
*   **Log-scale Energy:** Train on `log10(energy + epsilon)` to better handle the high dynamic range of calorimeter hits.
*   **Weighted Loss:** Apply a weight to the energy loss component proportional to the hit energy (or log-energy) to force the model to prioritize high-energy shower cores.
*   **Auxiliary Loss:** Implement a total energy conservation loss (sum of reconstructed vs sum of true energy in the window).

## Iterative Process (Up to 20 iterations)

1.  **Run Training:**
    Execute the training task using pixi:
    ```bash
    pixi run train
    ```
    This will run `train_example.py` with parameters defined in `pixi.toml`.

2.  **Evaluate Convergence and Performance:**
    *   Inspect `results/loss.csv` to see `train_loss`, `val_loss`, `coord_loss`, and `energy_loss`.
    *   Assess the density-fidelity correlation and 3D visualizations.
    *   **Identify gaps:** Is the model struggling with specific energy scales or spatial regions?

3.  **Refine Strategy and Metrics:**
    *   **Curriculum Learning (Hits Scheduling):** Gradually increase `num_hits` (e.g., from 256 to 512, then 1024) across iterations to increase the complexity of the local context.
    *   Update the **Current Strategy** section if a shift in approach is needed.
    *   Modify `train_example.py`, `dataset.py`, or `pixi.toml` accordingly.

4.  **Log and Commit:**
    *   Update `changelog.md` with a summary of the iteration outcome and changes made.
    *   Commit the changes to the codebase. **Ensure that `results/` folders are ignored.**

5.  **Archive Results:**
    *   Move `results/` to `results_iteration_N/`.

6.  **Repeat:**
    *   Repeat steps 1-5 for up to 20 iterations or until satisfactory convergence is achieved.

## Verification
*   Monitor the `loss.csv` and generated plots in each iteration's results folder.
