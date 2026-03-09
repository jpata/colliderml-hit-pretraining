# Convergence Study Instructions

This document outlines the procedure for the iterative convergence study of the Point Cloud Pretraining model.

## Goal
Achieve convergence of the validation loss for the Masked Point Modeling task.

## Current Strategy
*   **Base Model:** Transformer encoder with PointNet hit embeddings.
*   **Task:** Masked Point Modeling (reconstruct x, y, z, e of 75% masked hits).
*   **Loss:** SmoothL1Loss on masked hits.
*   **Optimization:** Adam with Cosine Annealing, LR=3e-4.
*   **Data:** `CalorimeterDataset` (**neighborhood=False**) with **512 hits** and 2000 events.
*   **Additional Metrics:** Monitor reconstruction fidelity vs density and 3D embedding clusters.

## Iterative Process (Up to 10 iterations)

1.  **Run Training:**
    Execute the training task using pixi:
    ```bash
    pixi run train
    ```
    This will run `train_example.py` with default parameters and save results to the `results/` directory.

2.  **Evaluate Convergence and Performance:**
    *   Inspect `results/loss.csv` to see the `train_loss` and `val_loss` over epochs.
    *   Check if `val_loss` is decreasing and stabilizing.
    *   Assess the density-fidelity correlation and 3D visualizations.
    *   **Identify gaps:** Are there specific behaviors (e.g., poor reconstruction of high-energy hits, lack of spatial coherence) not captured by current metrics?

3.  **Refine Strategy and Metrics:**
    *   Update the **Current Strategy** section if a shift in approach is needed.
    *   **Introduce new metrics:** If findings suggest a new way to measure success (e.g., energy conservation, cluster separation metrics), add them to the code and the strategy.
    *   Modify `train_example.py`, `dataset.py`, or `pixi.toml` accordingly.
    *   Possible changes include:
        *   Adjusting learning rate or scheduler.
        *   Modifying model architecture (e.g., `embed_dim`, number of layers/heads).
        *   Changing training hyperparameters (e.g., `batch_size`, `epochs`, `mask_ratio`).
        *   Adjusting data normalization or sampling strategy.

4.  **Log and Commit:**
    *   Update `changelog.md` with:
        *   A summary of the outcome of the previous iteration.
        *   The changes made in the current iteration and the justification.
        *   Any updates to the overall strategy.
    *   Commit the changes to the codebase with a meaningful message.

5.  **Archive Results:**
    *   Move the contents of the `results/` directory to a new folder named `results_iteration_N/` (where N is the iteration number).

6.  **Repeat:**
    *   Repeat steps 1-5 for up to 10 iterations or until satisfactory convergence is achieved.

## Verification
*   Monitor the `loss.csv` and generated plots in each iteration's results folder.
