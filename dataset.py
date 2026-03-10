
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
from hilbert import hilbert_index_3d
import matplotlib.pyplot as plt

class CalorimeterDataset(IterableDataset):
    @staticmethod
    def compute_local_features(hits, radii=[0.01, 0.02, 0.05]):
        """
        Precomputes multi-scale local density and energy-weighted features using scipy.
        This is an offline version of the feature computation for use in dataset generation.
        hits: (N, 4) array of (x, y, z, e)
        Returns: (N, 2 * len(radii)) array
        """
        from scipy.spatial import KDTree
        tree = KDTree(hits[:, :3])
        features = []
        for r in radii:
            # Multi-scale Density (Log-scaled)
            counts = tree.query_ball_point(hits[:, :3], r, return_length=True)
            features.append(np.log10(counts.astype(np.float32) + 1.0))
            
            # Local Energy Sum (Log-scaled)
            # Find neighbors for each hit and sum their energies
            indices = tree.query_ball_point(hits[:, :3], r)
            e_sums = np.array([hits[idx, 3].sum() for idx in indices], dtype=np.float32)
            features.append(np.log10(e_sums + 1.0))
            
        return np.stack(features, axis=1)

    def __init__(self, dataset_name="ttbar", num_hits=2048, max_events=None, skip_events=0, verbose=False, chunk_size=200):
        self.verbose = verbose
        self.num_hits = num_hits
        self.max_events = max_events
        self.skip_events = skip_events
        self.chunk_size = chunk_size
        self.epoch = 0
        
        # Identify shards dynamically from cache
        cache_root = Path("~/.cache/colliderml").expanduser()
        release_root = cache_root / "CERN__ColliderML-Release-1"
        
        calo_name = f"{dataset_name}_pu0_calo_hits"
        tracker_name = f"{dataset_name}_pu0_tracker_hits"
        
        self.calo_dir = release_root / calo_name / "data" / calo_name
        self.tracker_dir = release_root / tracker_name / "data" / tracker_name
        
        if not self.calo_dir.exists():
            raise FileNotFoundError(f"Calo shard directory {self.calo_dir} not found.")
        
        self.calo_shards = sorted(list(self.calo_dir.glob("train-*.parquet")))
        if not self.calo_shards:
            raise FileNotFoundError(f"No parquet shards found in {self.calo_dir}")
        
        # We assume tracker shards match calo shards one-to-one
        self.tracker_shards = sorted(list(self.tracker_dir.glob("train-*.parquet"))) if self.tracker_dir.exists() else []

        self.rows_per_shard = 1000
        self.coord_scale = 5000.0
        self.tracker_energy_const = 2.6 # Approx mean log-energy of calo hits
        self.required_cols_calo = ["x", "y", "z", "total_energy"]
        self.required_cols_tracker = ["x", "y", "z"]

    def __len__(self):
        total = len(self.calo_shards) * self.rows_per_shard - self.skip_events
        if self.max_events is not None:
            return min(self.max_events, total)
        return total

    def _process_chunk(self, calo_df, tracker_df=None):
        """Processes a chunk of events into a list of hit arrays with hit type flag."""
        n_events = calo_df.height
        
        # Calo: (x, y, z, e)
        # Use explode() + to_numpy() for vectorized processing
        c_x = calo_df["x"].list.explode().to_numpy().astype(np.float32) / self.coord_scale
        c_y = calo_df["y"].list.explode().to_numpy().astype(np.float32) / self.coord_scale
        c_z = calo_df["z"].list.explode().to_numpy().astype(np.float32) / self.coord_scale
        c_e = np.log10(calo_df["total_energy"].list.explode().to_numpy().astype(np.float32) / 1e-6 + 1.0)
        c_counts = calo_df["x"].list.len().to_numpy()
        
        # Tracker: (x, y, z)
        if tracker_df is not None:
            t_x = tracker_df["x"].list.explode().to_numpy().astype(np.float32) / self.coord_scale
            t_y = tracker_df["y"].list.explode().to_numpy().astype(np.float32) / self.coord_scale
            t_z = tracker_df["z"].list.explode().to_numpy().astype(np.float32) / self.coord_scale
            t_counts = tracker_df["x"].list.len().to_numpy()
        else:
            t_counts = np.zeros(n_events, dtype=np.int64)

        events = []
        c_offset = 0
        t_offset = 0
        for i in range(n_events):
            cc = c_counts[i]
            tc = t_counts[i]
            
            # Calorimeter Hits (Type 0)
            cx = c_x[c_offset : c_offset + cc]
            cy = c_y[c_offset : c_offset + cc]
            cz = c_z[c_offset : c_offset + cc]
            ce = c_e[c_offset : c_offset + cc]
            ct = np.zeros(cc, dtype=np.float32)
            
            # Tracker Hits (Type 1)
            if tc > 0:
                tx = t_x[t_offset : t_offset + tc]
                ty = t_y[t_offset : t_offset + tc]
                tz = t_z[t_offset : t_offset + tc]
                te = np.full(tc, self.tracker_energy_const, dtype=np.float32)
                tt = np.ones(tc, dtype=np.float32)
                
                x = np.concatenate([cx, tx])
                y = np.concatenate([cy, ty])
                z = np.concatenate([cz, tz])
                e = np.concatenate([ce, te])
                h_type = np.concatenate([ct, tt])
            else:
                x, y, z, e, h_type = cx, cy, cz, ce, ct
            
            events.append(np.stack([x, y, z, e, h_type], axis=1))
            c_offset += cc
            t_offset += tc
            
        return events

    def set_epoch(self, epoch):
        """Used to update the epoch for randomization across epochs."""
        self.epoch = epoch

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        
        # Use SeedSequence to ensure unique, robust streams across workers/epochs
        ss = np.random.SeedSequence([self.epoch, worker_id])
        
        # Determine global max_events per worker
        if self.max_events is not None:
            max_events = self.max_events // num_workers
            if worker_id < self.max_events % num_workers:
                max_events += 1
        else:
            max_events = None

        # Partition shards among workers
        shard_indices = list(range(worker_id, len(self.calo_shards), num_workers))
        
        events_yielded = 0
        
        for s_idx in shard_indices:
            calo_path = self.calo_shards[s_idx]
            tracker_path = self.tracker_shards[s_idx] if self.tracker_shards else None
            
            # Global index bounds for this shard
            shard_global_start = s_idx * self.rows_per_shard
            shard_global_end = (s_idx + 1) * self.rows_per_shard
            
            # Skip if this shard is entirely before the skip threshold
            if shard_global_end <= self.skip_events:
                continue
            
            # Where to start within this shard
            start_within_shard = max(0, self.skip_events - shard_global_start)
            
            if self.verbose:
                print(f"Worker {worker_id} processing calo shard {calo_path.name}, skipping {start_within_shard} events.")
            
            l_calo = pl.scan_parquet(calo_path, low_memory=True)
            l_tracker = pl.scan_parquet(tracker_path, low_memory=True) if tracker_path else None
            
            for chunk_start in range(0, self.rows_per_shard, self.chunk_size):
                chunk_end = chunk_start + self.chunk_size
                
                # Entire chunk skipped
                if chunk_end <= start_within_shard:
                    continue
                
                # Slice logic within shard
                slice_start = max(chunk_start, start_within_shard)
                slice_len = min(self.chunk_size, chunk_end - slice_start)
                
                try:
                    c_chunk = l_calo.slice(slice_start, slice_len).collect()
                    t_chunk = l_tracker.slice(slice_start, slice_len).collect() if l_tracker is not None else None
                except Exception as e:
                    if self.verbose:
                        print(f"Error reading chunk in shard {s_idx}: {e}")
                    break
                
                raw_events = self._process_chunk(c_chunk, t_chunk)
                
                for raw_hits in raw_events:
                    if max_events is not None and events_yielded >= max_events:
                        return
                    
                    # Generate an independent RNG for this event
                    event_rng = np.random.default_rng(ss.spawn(1)[0])
                    processed = self._finalize_event(raw_hits, event_rng)
                    yield processed
                    events_yielded += 1

    def _sort_hits(self, hits):
        """Sorts hits based on Morton/Hilbert index for spatial locality."""
        # Avoid sorting the padding (0,0,0)
        coords = hits[:, :3]
        indices = hilbert_index_3d(coords)
        sort_idx = np.argsort(indices)
        return hits[sort_idx]

    def _finalize_event(self, hits, rng):
        n_hits = hits.shape[0]
        
        if n_hits >= self.num_hits:
            indices = rng.choice(n_hits, self.num_hits, replace=False)
            hits = hits[indices]
        else:
            padding = np.zeros((self.num_hits - n_hits, 5), dtype=np.float32)
            hits = np.concatenate([hits, padding], axis=0)
            
        # Sort for spatial locality
        hits = self._sort_hits(hits)
        return torch.from_numpy(hits)

    def get_full_event(self, idx):
        """Sparse access for visualization only."""
        shard_idx = idx // self.rows_per_shard
        rel_idx = idx % self.rows_per_shard
        
        c_path = self.calo_shards[shard_idx]
        t_path = self.tracker_shards[shard_idx] if self.tracker_shards else None
        
        c_df = pl.read_parquet(c_path, columns=self.required_cols_calo).slice(rel_idx, 1)
        t_df = pl.read_parquet(t_path, columns=self.required_cols_tracker).slice(rel_idx, 1) if t_path else None
        
        # Reuse process_chunk logic
        raw_hits = self._process_chunk(c_df, t_df)[0]
        
        # Split back for visualization API compatibility
        calo_mask = raw_hits[:, 4] == 0
        tracker_mask = raw_hits[:, 4] == 1
        
        return {
            "event_id": idx,
            "calo_hits": raw_hits[calo_mask, :4],
            "tracker_hits": raw_hits[tracker_mask, :4],
            "all_hits": raw_hits # (N, 5)
        }

class NeighborhoodCalorimeterDataset(CalorimeterDataset):
    def _finalize_event(self, hits, rng):
        n_hits = hits.shape[0]

        if n_hits <= self.num_hits:
            padding = np.zeros((self.num_hits - n_hits, 5), dtype=np.float32)
            hits = np.concatenate([hits, padding], axis=0)
        else:
            e = hits[:, 3]
            probs = e / (e.sum() + 1e-9)
            seed_idx = rng.choice(n_hits, p=probs)
            seed_pos = hits[seed_idx, :3]
            
            dists = np.sum((hits[:, :3] - seed_pos)**2, axis=1)
            neighbor_indices = np.argpartition(dists, self.num_hits)[:self.num_hits]
            hits = hits[neighbor_indices]
            
        # Sort for spatial locality
        hits = self._sort_hits(hits)
        return torch.from_numpy(hits)

if __name__ == "__main__":
    ds_full = CalorimeterDataset(num_hits=1024, max_events=None, verbose=False)
    print(f"Total events available in dataset: {len(ds_full)}")
    
    ds = CalorimeterDataset(num_hits=1024, max_events=100, verbose=True)
    print(f"Running validation on {len(ds)} events...")
    
    start = time.time()
    all_hits = []
    for i, event in enumerate(ds):
        if i % 10 == 0:
            print(f"Yielded event {i}, time since last: {time.time()-start:.4f}s")
            start = time.time()
        all_hits.append(event.numpy())
    
    all_hits = np.concatenate(all_hits, axis=0)
    # Mask out padding (0,0,0,0,0)
    mask = np.any(all_hits != 0, axis=1)
    valid_hits = all_hits[mask]
    
    print(f"Collected {len(valid_hits)} valid hits from {len(ds)} events.")
    
    # Compute additional precomputed features (density and energy density)
    def compute_density_np(hits, radii=[0.01, 0.02, 0.05]):
        # hits: (N, 4)
        hits_torch = torch.from_numpy(hits).unsqueeze(0) # (1, N, 4)
        coords = hits_torch[:, :, :3]
        energies = hits_torch[:, :, 3]
        dist_sq = torch.cdist(coords, coords)**2
        
        density_features = []
        energy_features = []
        for r in radii:
            mask = (dist_sq < r**2).float()
            d = torch.log10(mask.sum(dim=2) + 1.0)
            e_sum = torch.log10(torch.bmm(mask, energies.unsqueeze(-1)).squeeze(-1) + 1.0)
            density_features.append(d.squeeze(0))
            energy_features.append(e_sum.squeeze(0))
            
        return torch.stack(density_features + energy_features, dim=-1).numpy()

    # Let's re-collect hits and compute features per event
    all_hits_with_features = []
    sampled_calo_multiplicities = []
    sampled_tracker_multiplicities = []
    full_calo_multiplicities = []
    full_tracker_multiplicities = []
    
    print("Collecting full event multiplicities and computing features...")
    for i in range(len(ds)):
        # Get full event for true multiplicity
        full_event = ds.get_full_event(i)
        full_calo_multiplicities.append(len(full_event["calo_hits"]))
        full_tracker_multiplicities.append(len(full_event["tracker_hits"]))
    
    # Reset iterator to get sampled hits
    ds.epoch = 0 
    for i, event in enumerate(ds):
        hits = event.numpy()
        mask = np.any(hits != 0, axis=1)
        v_hits = hits[mask]
        
        # Collect sampled multiplicity (limited by num_hits)
        calo_count = np.sum(v_hits[:, 4] == 0)
        tracker_count = np.sum(v_hits[:, 4] == 1)
        sampled_calo_multiplicities.append(calo_count)
        sampled_tracker_multiplicities.append(tracker_count)

        if len(v_hits) > 0:
            densities = compute_density_np(v_hits[:, :4])
            all_hits_with_features.append(np.concatenate([v_hits, densities], axis=1))
    
    valid_hits_extended = np.concatenate(all_hits_with_features, axis=0)
    
    # Create validation plots
    os.makedirs("validation_plots", exist_ok=True)
    
    # Multiplicity Plots
    plt.figure(figsize=(15, 10))
    # Full Events
    plt.subplot(2, 2, 1)
    plt.hist(full_calo_multiplicities, bins=50, color='blue', alpha=0.7)
    plt.title("Full Event: Calorimeter Hit Multiplicity")
    plt.xlabel("Number of Hits")
    plt.subplot(2, 2, 2)
    plt.hist(full_tracker_multiplicities, bins=50, color='green', alpha=0.7)
    plt.title("Full Event: Tracker Hit Multiplicity")
    plt.xlabel("Number of Hits")
    # Sampled Chunks
    plt.subplot(2, 2, 3)
    plt.hist(sampled_calo_multiplicities, bins=50, color='blue', alpha=0.5)
    plt.title(f"Sampled Chunk (max={ds.num_hits}): Calo Multiplicity")
    plt.xlabel("Number of Hits")
    plt.subplot(2, 2, 4)
    plt.hist(sampled_tracker_multiplicities, bins=50, color='green', alpha=0.5)
    plt.title(f"Sampled Chunk (max={ds.num_hits}): Tracker Multiplicity")
    plt.xlabel("Number of Hits")
    plt.tight_layout()
    plt.savefig("validation_plots/multiplicity_distributions.png")
    plt.close()
    
    # Coordinates
    plt.figure(figsize=(15, 5))
    for i, label in enumerate(["x", "y", "z"]):
        plt.subplot(1, 3, i+1)
        plt.hist(valid_hits_extended[:, i], bins=100)
        plt.title(f"Distribution of {label}")
    plt.tight_layout()
    plt.savefig("validation_plots/coord_distributions.png")
    plt.close()
    
    # Energy
    plt.figure(figsize=(8, 6))
    plt.hist(valid_hits_extended[:, 3], bins=100)
    plt.title("Log10(Energy/1e-6 + 1.0) Distribution")
    plt.xlabel("Log10 Energy")
    plt.savefig("validation_plots/energy_distribution.png")
    plt.close()
    
    # Hit Densities (Radii: 0.01, 0.02, 0.05)
    plt.figure(figsize=(15, 5))
    for i, r in enumerate([0.01, 0.02, 0.05]):
        plt.subplot(1, 3, i+1)
        plt.hist(valid_hits_extended[:, 5+i], bins=100)
        plt.title(f"Hit Density (r={r})")
    plt.tight_layout()
    plt.savefig("validation_plots/hit_density_distributions.png")
    plt.close()
    
    # Energy Densities (Radii: 0.01, 0.02, 0.05)
    plt.figure(figsize=(15, 5))
    for i, r in enumerate([0.01, 0.02, 0.05]):
        plt.subplot(1, 3, i+1)
        plt.hist(valid_hits_extended[:, 8+i], bins=100)
        plt.title(f"Energy Density (r={r})")
    plt.tight_layout()
    plt.savefig("validation_plots/energy_density_distributions.png")
    plt.close()
    
    # Hit Types
    plt.figure(figsize=(8, 6))
    plt.hist(valid_hits_extended[:, 4], bins=3, range=(-0.5, 1.5))
    plt.xticks([0, 1], ["Calo (0)", "Tracker (1)"])
    plt.title("Hit Type Distribution")
    plt.savefig("validation_plots/hittype_distribution.png")
    plt.close()

    print("Validation plots (including densities) saved to 'validation_plots/' directory.")
