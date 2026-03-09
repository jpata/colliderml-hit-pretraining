# Changelog

## [Iteration 2]
- Increased default `batch_size` from 4 to 16 for more stable gradients.
- Added `CosineAnnealingLR` scheduler to help convergence.

## [Iteration 1]
- Fixed validation loss calculation to be consistent with training loss (only on masked tokens).
- Increased default `embed_dim` from 16 to 64 for higher model capacity.
- Increased default `epochs` from 5 to 10.
- Fixed `pixi.toml` train task to provide a filename for `--output_loss`.
