
import time
import torch
from torch.utils.data import DataLoader
from dataset import CalorimeterDataset
import numpy as np

def check_consistency(dataset):
    print("\nChecking consistency between different num_workers...")
    
    def get_first_batch(workers):
        # Fix ALL seeds
        torch.manual_seed(42)
        np.random.seed(42)
        # Use a very simple sampler to ensure same indices
        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=workers)
        return next(iter(loader))

    batch0 = get_first_batch(0)
    batch1 = get_first_batch(1)
    
    diff1 = torch.abs(batch0 - batch1).max().item()
    print(f"Max difference (0 vs 1 workers): {diff1}")
    
    if diff1 > 1e-6:
        print("Detailed comparison of first 2 hits:")
        print("Batch 0 (0 workers):", batch0[0, :2])
        print("Batch 1 (1 worker):", batch1[0, :2])
        
    if diff1 < 1e-6:
        print("Consistency check PASSED")
    else:
        print("Consistency check FAILED")

if __name__ == "__main__":
    ds = CalorimeterDataset(num_hits=1024, max_events=1000, verbose=False)
    check_consistency(ds)
