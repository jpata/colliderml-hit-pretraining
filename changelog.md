# Changelog

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
