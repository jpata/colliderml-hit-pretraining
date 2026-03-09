import numpy as np
import torch

def hilbert_index_3d(coords, bits=10):
    """
    Computes Morton (Z-order) indices for 3D coordinates to provide spatial sorting.
    While Hilbert curves provide slightly better locality, Morton curves are 
    computationally efficient and robust for vectorized numpy operations.
    
    coords: (N, 3) tensor or numpy array in range [0, 1]
    bits: number of bits per dimension for quantization (default 10 = 1024^3 grid)
    """
    if isinstance(coords, torch.Tensor):
        coords_np = coords.detach().cpu().numpy()
    else:
        coords_np = coords
        
    # 1. Quantize coordinates to [0, 2^bits - 1]
    # We assume coords are normalized to [0, 1]
    max_val = (1 << bits) - 1
    xyz = (coords_np * max_val).astype(np.int64)
    xyz = np.clip(xyz, 0, max_val)
    
    N = xyz.shape[0]
    morton = np.zeros(N, dtype=np.int64)
    
    # 2. Interleave bits (Vectorized)
    # Bit i of x goes to position 3*i + 2
    # Bit i of y goes to position 3*i + 1
    # Bit i of z goes to position 3*i + 0
    for i in range(bits):
        morton |= ((xyz[:, 0] >> i) & 1) << (3 * i + 2)
        morton |= ((xyz[:, 1] >> i) & 1) << (3 * i + 1)
        morton |= ((xyz[:, 2] >> i) & 1) << (3 * i)
        
    return morton
