
"""
Central configuration for the Masked Point Model.
All architectural hyperparameters should be modified here to ensure 
consistency between training, inference, and visualization.
"""

MODEL_CONFIG = {
    # --- Input ---
    # Number of hits per training sample (window size).
    "num_hits": 2048,

    # --- Encoder ---
    # Dimension of the latent space for patch tokens. 
    # Increasing this increases model capacity but uses more memory.
    "embed_dim": 64,

    # Number of patches to sample from each window using Farthest Point Sampling (FPS).
    # Higher values provide finer geometric resolution at the cost of computation.
    "n_patches": 128,

    # Number of Transformer blocks in the heavy encoder.
    "encoder_layers": 6,

    # Number of attention heads. embed_dim must be divisible by nhead.
    "nhead": 8,
    
    # --- Patch Tokenization (PointNet) ---
    # Number of nearest neighbors to group into each patch.
    # Defines the local receptive field size for each patch token.
    "k_neighbors": 256,
    
    # --- Decoder ---
    # Dimension of the decoder latent space. Usually smaller than embed_dim (lightweight decoder).
    "decoder_embed_dim": 64,

    # Number of Transformer blocks in the lightweight decoder.
    "decoder_layers": 4,
    
    # --- Output ---
    # Final feature dimension of the reconstructed points.
    # Currently: 5 raw features (x, y, z, energy, type) + 6 multi-scale density features.
    "output_dim": 11,
}

def get_model_config():
    """Returns the central model configuration dictionary."""
    return MODEL_CONFIG.copy()
