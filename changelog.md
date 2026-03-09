# Changelog

## [Iteration 11]
- Implemented log-scaling for energy: `log10(energy + 1)`.
- Increased `num_hits` from 256 to 512 (Curriculum Learning).
- Observed dramatic improvement in `energy_loss` (~0.000003 on log-scale), now much lower than `coord_loss` (~0.0005).
- Correlation between density and MAE increased to ~ -0.79.
- Overall `val_loss` reached ~0.0011.

## [Iteration 10]
- Increased training to 20 epochs for better convergence.
- Observed that `energy_loss` (~0.0029) is significantly higher than `coord_loss` (~0.0003), indicating that energy reconstruction is the main bottleneck.
- Correlation between density and MAE remained stable at ~ -0.70.
- Overall `val_loss` converged to ~0.0035.

## [Iteration 9]
- Reverted to `neighborhood=True` (local neighborhood sampling) as per the convergence study strategy in `convergence.md`.
- Set `num_hits=256` and `mask_ratio=0.75`.
- Added granular logging for `coord_loss` (x, y, z) and `energy_loss` (e) to better understand reconstruction performance.
- Observed `val_loss` of ~0.0008 and strong negative correlation (-0.69) between density and MAE.

## [Iteration 8]
- Switched to `neighborhood=False` (random global sampling) to increase task difficulty and learn global dependencies.
- Increased `num_hits` from 256 to 512 to provide more context for global sampling.
- Kept `mask_ratio=0.75` and `lr=3e-4`.

## [Iteration 7]
- Increased `mask_ratio` from 0.5 to 0.75.
- Increased `max_events` to 2000.
- Observed that convergence remained very fast with low loss (~0.0006), reinforcing that the local neighborhood task is likely too simple.
- Strong negative correlation (~ -0.7) between density and MAE persisted.

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
