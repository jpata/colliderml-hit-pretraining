# Changelog

## [Iteration 7]
- Increased `mask_ratio` from 0.5 to 0.75 to make the reconstruction task more challenging.
- Increased `max_events` from 1000 to 2000 for more data diversity.
- Increased learning rate from 1e-4 to 3e-4 for faster exploration.
- Modified `train_example.py` to support `mask_ratio` and `lr` as CLI arguments.

## [Iteration 6]
- Ran training with 1000 events (limited to avoid OOM).
- Observed quick convergence to a low loss (~0.001) within 2 epochs.
- Found strong negative correlation (~ -0.68) between local hit density and reconstruction MAE, suggesting dense regions are easier to reconstruct.
- Identified that the task may be too easy due to local clustering in the neighborhood dataset.

## [Iteration 5]
- Switched from `MSELoss` to `SmoothL1Loss` for more robust reconstruction.
- Added `weight_decay=1e-5` to `Adam` optimizer.
- Increased `embed_dim` to 128 and `epochs` to 20 for better capacity and learning time.

## [Iteration 4]
- Added random-initialized learned positional embeddings to `MaskedPointModel` to break permutation invariance for masked tokens. This allows the model to predict unique features for each masked hit.

## [Iteration 3]
- Added `LayerNorm` to `PointNetEncoder` and `reconstructor` for better stability.
- Increased `num_layers` of the Transformer from 4 to 8.

## [Iteration 2]
- Increased default `batch_size` from 4 to 16 for more stable gradients.
- Added `CosineAnnealingLR` scheduler to help convergence.

## [Iteration 1]
- Fixed validation loss calculation to be consistent with training loss (only on masked tokens).
- Increased default `embed_dim` from 16 to 64 for higher model capacity.
- Increased default `epochs` from 5 to 10.
- Fixed `pixi.toml` train task to provide a filename for `--output_loss`.
