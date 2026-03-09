
"""
Dataset definitions for Calorimeter Hits.
Uses IterableDataset for efficient chunked loading and worker partitioning.
Memory optimized to avoid OOM by using manual slicing instead of explode().
"""

import torch
from torch.utils.data import IterableDataset, get_worker_info
import numpy as np
import os
import time
import polars as pl
from pathlib import Path

class CalorimeterDataset(IterableDataset):
    def __init__(self, num_hits=2048, max_events=None, verbose=False, chunk_size=20):
        self.verbose = verbose
        self.num_hits = num_hits
        self.max_events = max_events
        self.chunk_size = chunk_size
        
        # Identify shards dynamically from cache
        cache_root = Path("~/.cache/colliderml").expanduser()
        self.shard_dir = cache_root / "CERN__ColliderML-Release-1/ttbar_pu0_calo_hits/data/ttbar_pu0_calo_hits"
        
        if not self.shard_dir.exists():
            raise FileNotFoundError(f"Shard directory {self.shard_dir} not found.")
        
        self.shards = sorted(list(self.shard_dir.glob("train-*.parquet")))
        if not self.shards:
            raise FileNotFoundError(f"No parquet shards found in {self.shard_dir}")

        self.rows_per_shard = 1000
        self.coord_scale = 5000.0
        self.required_cols = ["x", "y", "z", "total_energy"]

    def __len__(self):
        total = len(self.shards) * self.rows_per_shard
        if self.max_events is not None:
            return min(self.max_events, total)
        return total

    def _process_chunk(self, df):
        """Processes a chunk of events into a list of hit arrays using manual conversion."""
        events = []
        # Convert columns to numpy once per chunk
        # Note: Polars Series of Lists to numpy returns a nested numpy array (object type)
        # We manually convert each list to a flat float32 array
        x_lists = df["x"].to_list()
        y_lists = df["y"].to_list()
        z_lists = df["z"].to_list()
        e_lists = df["total_energy"].to_list()
        
        for i in range(len(x_lists)):
            x = np.array(x_lists[i], dtype=np.float32) / self.coord_scale
            y = np.array(y_lists[i], dtype=np.float32) / self.coord_scale
            z = np.array(z_lists[i], dtype=np.float32) / self.coord_scale
            e = np.log10(np.array(e_lists[i], dtype=np.float32) / 1e-6 + 1.0)
            events.append(np.stack([x, y, z, e], axis=1))
            
        return events

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            shards = self.shards
            max_events = self.max_events
        else:
            # Partition shards among workers
            shards = self.shards[worker_info.id::worker_info.num_workers]
            if self.max_events is not None:
                # Divide max_events by number of workers, assigning remainders to first workers
                max_events = self.max_events // worker_info.num_workers
                if worker_info.id < self.max_events % worker_info.num_workers:
                    max_events += 1
            else:
                max_events = None
            
        events_yielded = 0
        
        for shard_path in shards:
            if self.verbose:
                print(f"Worker {os.getpid()} processing shard {shard_path.name}")
            
            # Use scan_parquet for efficient slicing
            lf = pl.scan_parquet(shard_path, low_memory=True)
            
            for start in range(0, self.rows_per_shard, self.chunk_size):
                if max_events is not None and events_yielded >= max_events:
                    return
                
                num_to_load = min(self.chunk_size, self.rows_per_shard - start)
                
                try:
                    # Efficiently load ONLY the chunk
                    chunk_df = lf.slice(start, num_to_load).collect()
                except Exception as e:
                    if self.verbose:
                        print(f"Error reading chunk from {shard_path}: {e}")
                    break
                
                # Process chunk into list of arrays
                raw_events = self._process_chunk(chunk_df)
                
                # Yield each event
                for raw_hits in raw_events:
                    if max_events is not None and events_yielded >= max_events:
                        return
                        
                    processed = self._finalize_event(raw_hits, events_yielded)
                    yield processed
                    events_yielded += 1

    def _finalize_event(self, hits, event_idx):
        n_hits = hits.shape[0]
        rng = np.random.default_rng(seed=event_idx)
        
        if n_hits >= self.num_hits:
            indices = rng.choice(n_hits, self.num_hits, replace=False)
            hits = hits[indices]
        else:
            padding = np.zeros((self.num_hits - n_hits, 4), dtype=np.float32)
            hits = np.concatenate([hits, padding], axis=0)
            
        return torch.from_numpy(hits)

    def get_full_event(self, idx):
        """Sparse access for visualization only."""
        shard_idx = idx // self.rows_per_shard
        rel_idx = idx % self.rows_per_shard
        shard_path = self.shards[shard_idx]
        
        df = pl.read_parquet(shard_path, columns=self.required_cols).slice(rel_idx, 1)
        
        x = np.concatenate(df["x"].to_numpy()) / self.coord_scale
        y = np.concatenate(df["y"].to_numpy()) / self.coord_scale
        z = np.concatenate(df["z"].to_numpy()) / self.coord_scale
        e = np.log10(np.concatenate(df["total_energy"].to_numpy()) / 1e-6 + 1.0)
        hits = np.stack([x, y, z, e], axis=1).astype(np.float32)
        
        return {
            "event_id": idx,
            "calo_hits": hits,
            "tracker_hits": np.zeros((0, 4), dtype=np.float32)
        }

class NeighborhoodCalorimeterDataset(CalorimeterDataset):
    def _finalize_event(self, hits, event_idx):
        n_hits = hits.shape[0]
        rng = np.random.default_rng(seed=event_idx)

        if n_hits <= self.num_hits:
            padding = np.zeros((self.num_hits - n_hits, 4), dtype=np.float32)
            result = torch.from_numpy(np.concatenate([hits, padding], axis=0))
        else:
            e = hits[:, 3]
            probs = e / (e.sum() + 1e-9)
            seed_idx = rng.choice(n_hits, p=probs)
            seed_pos = hits[seed_idx, :3]
            
            dists = np.sum((hits[:, :3] - seed_pos)**2, axis=1)
            neighbor_indices = np.argpartition(dists, self.num_hits)[:self.num_hits]
            neighbor_indices = neighbor_indices[np.argsort(dists[neighbor_indices])]
            
            result = torch.from_numpy(hits[neighbor_indices])
            
        return result

if __name__ == "__main__":
    ds = CalorimeterDataset(num_hits=1024, max_events=50, verbose=True)
    start = time.time()
    for i, event in enumerate(ds):
        if i % 10 == 0:
            print(f"Yielded event {i}, time since last: {time.time()-start:.4f}s")
            start = time.time()
