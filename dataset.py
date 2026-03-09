"""
Dataset definitions for Calorimeter Hits.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
from colliderml.core import load_tables, collect_tables

class CalorimeterDataset(Dataset):
    def __init__(self, num_hits=2048, max_events=None):
        cfg = {
            "dataset_id": "CERN/ColliderML-Release-1",
            "channels": "ttbar",
            "pileup": "pu0",
            "objects": ["particles", "tracker_hits", "calo_hits", "tracks"],
            "split": "train",
            "lazy": False,
            "max_events": max_events,
        }
        print(f"Loading data from ColliderML (max_events={max_events})...")
        tables = load_tables(cfg)
        self.frames = collect_tables(tables)
        # Group hits by event_id to get a list of events (DataFrames)
        self.events = self.frames["calo_hits"].partition_by("event_id", include_key=True)
        self.num_hits = num_hits
        
        # Pre-calculated normalization constants (approximate)
        self.coord_scale = 5000.0
        self.energy_scale = 0.1

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        event = self.events[idx]
        
        # Extract features (x, y, z, total_energy) and flatten if they are list columns
        x = np.concatenate(event["x"].to_numpy()) / self.coord_scale
        y = np.concatenate(event["y"].to_numpy()) / self.coord_scale
        z = np.concatenate(event["z"].to_numpy()) / self.coord_scale
        e = np.concatenate(event["total_energy"].to_numpy()) / self.energy_scale
        
        # Combine into a single hit array (N, 4)
        hits = np.stack([x, y, z, e], axis=1).astype(np.float32)
        
        # Fixed-size sampling/padding for batching
        n_hits = hits.shape[0]
        if n_hits >= self.num_hits:
            # Randomly sample num_hits
            indices = np.random.choice(n_hits, self.num_hits, replace=False)
            hits = hits[indices]
        else:
            # Pad with zeros
            padding = np.zeros((self.num_hits - n_hits, 4), dtype=np.float32)
            hits = np.concatenate([hits, padding], axis=0)
            
        return torch.from_numpy(hits)

    def get_full_event(self, idx):
        """Returns all calo and tracker hits for a given event index."""
        calo_event = self.events[idx]
        event_id = calo_event["event_id"][0]
        
        # Get tracker hits for the same event
        import polars as pl
        tracker_frames = self.frames["tracker_hits"]
        tracker_event = tracker_frames.filter(pl.col("event_id") == event_id)
        
        def process_hits(df, is_tracker=False):
            if len(df) == 0:
                return np.zeros((0, 4), dtype=np.float32)
            x = np.concatenate(df["x"].to_numpy()) / self.coord_scale
            y = np.concatenate(df["y"].to_numpy()) / self.coord_scale
            z = np.concatenate(df["z"].to_numpy()) / self.coord_scale
            if "total_energy" in df.columns:
                e = np.concatenate(df["total_energy"].to_numpy()) / self.energy_scale
            else:
                # Assign a small dummy energy for tracker hits if missing
                e = np.ones_like(x) * 0.01 
            return np.stack([x, y, z, e], axis=1).astype(np.float32)

        calo_hits = process_hits(calo_event)
        tracker_hits = process_hits(tracker_event, is_tracker=True)
        
        return {
            "event_id": event_id,
            "calo_hits": calo_hits,
            "tracker_hits": tracker_hits
        }

class NeighborhoodCalorimeterDataset(CalorimeterDataset):
    """
    Dataset that samples a 'neighborhood' around a randomly selected seed hit.
    This is better for learning local correlations in large events.
    """
    def __init__(self, num_hits=256, max_events=None):
        super().__init__(num_hits=num_hits, max_events=max_events)
        print(f"Loading data for NeighborhoodDataset (max_events={max_events})...")

    def __getitem__(self, idx):
        event = self.events[idx]
        
        x = np.concatenate(event["x"].to_numpy()) / self.coord_scale
        y = np.concatenate(event["y"].to_numpy()) / self.coord_scale
        z = np.concatenate(event["z"].to_numpy()) / self.coord_scale
        e = np.concatenate(event["total_energy"].to_numpy()) / self.energy_scale
        
        hits = np.stack([x, y, z, e], axis=1).astype(np.float32)
        n_hits = hits.shape[0]
        
        if n_hits <= self.num_hits:
            # If event is small, pad as usual
            padding = np.zeros((self.num_hits - n_hits, 4), dtype=np.float32)
            return torch.from_numpy(np.concatenate([hits, padding], axis=0))

        # 1. Select a random 'seed' hit (weighted by energy to focus on showers)
        probs = e / (e.sum() + 1e-9)
        seed_idx = np.random.choice(n_hits, p=probs)
        seed_pos = hits[seed_idx, :3]
        
        # 2. Find nearest neighbors
        dists = np.sum((hits[:, :3] - seed_pos)**2, axis=1)
        neighbor_indices = np.argsort(dists)[:self.num_hits]
        
        return torch.from_numpy(hits[neighbor_indices])

def validate_dataset(max_events=100, output_dir="validation_plots"):
    """
    Validates the dataset by plotting distributions and printing a summary.
    """
    dataset = CalorimeterDataset(num_hits=2048, max_events=max_events)
    num_events = len(dataset)
    
    # Collect calo hits info
    calo_frames = dataset.frames["calo_hits"]
    tracker_frames = dataset.frames["tracker_hits"]
    
    # Multiplicities per event
    import polars as pl
    if isinstance(calo_frames, pl.DataFrame):
        # Multiplicity is the length of the list columns since each row is one event
        calo_multiplicities = calo_frames["total_energy"].list.len().to_numpy()
        tracker_multiplicities = tracker_frames["x"].list.len().to_numpy()
    else:
        # Fallback if they are not polars (e.g. if they are already partitioned)
        calo_multiplicities = []
        for i in range(num_events):
            calo_multiplicities.append(len(dataset.events[i]))
        
        if hasattr(tracker_frames, "partition_by"):
            tracker_events = tracker_frames.partition_by("event_id", include_key=True)
            tracker_multiplicities = [len(ev) for ev in tracker_events]
        else:
            tracker_multiplicities = [0] * num_events

    # Collect energy values
    def flatten_column(df, col_name):
        series = df[col_name]
        import polars as pl
        if isinstance(series.dtype, pl.List):
            return np.concatenate(series.to_list())
        return series.to_numpy()

    calo_energies = flatten_column(calo_frames, "total_energy")
    
    tracker_energies = None
    if "total_energy" in tracker_frames.columns:
        tracker_energies = flatten_column(tracker_frames, "total_energy")

    print("\n--- Dataset Summary ---")
    print(f"Number of events: {num_events}")
    print(f"Avg Calo hits per event: {np.mean(calo_multiplicities):.2f}")
    print(f"Avg Tracker hits per event: {np.mean(tracker_multiplicities):.2f}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    features = ['x', 'y', 'z', 'total_energy']
    for i, feat in enumerate(features):
        data = flatten_column(calo_frames, feat)
        if feat in ['x', 'y', 'z']:
            data = data / dataset.coord_scale
        else:
            data = data / dataset.energy_scale
            
        axes[i].hist(data, bins=50, color='skyblue', edgecolor='black')
        axes[i].set_title(f'Calo Hit Distribution: {feat}')
        axes[i].set_xlabel('Normalized Value' if feat != 'total_energy' else 'Normalized Energy')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "calo_feature_distributions.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(calo_multiplicities, bins=30, alpha=0.5, label='Calo Hits', color='blue', edgecolor='black')
    plt.hist(tracker_multiplicities, bins=30, alpha=0.5, label='Tracker Hits', color='green', edgecolor='black')
    plt.title('Hit Multiplicity per Event')
    plt.xlabel('Number of Hits')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "multiplicity_distributions.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(calo_energies, bins=50, color='salmon', edgecolor='black')
    plt.title('Calo Hit Energy Distribution')
    plt.xlabel('Energy')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "calo_energy_distribution.png"))
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    tracker_coords = ['x', 'y', 'z']
    for i, feat in enumerate(tracker_coords):
        data = flatten_column(tracker_frames, feat)
        data = data / dataset.coord_scale
        axes[i].hist(data, bins=50, color='lightgreen', edgecolor='black')
        axes[i].set_title(f'Tracker Hit Distribution: {feat}')
        axes[i].set_xlabel('Normalized Value')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tracker_feature_distributions.png"))
    plt.close()

    print(f"\nValidation plots saved to {output_dir}/")

if __name__ == "__main__":
    validate_dataset(max_events=50)
