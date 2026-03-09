"""
Script to compute representations for ALL hits in an event using a trained model.
Uses a sliding window approach to handle large contexts.
"""

import torch
import torch.nn as nn
from dataset import CalorimeterDataset
from train_example import MaskedPointModel, compute_all_hit_representations
import numpy as np
from tqdm import tqdm
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_hits", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=16)
    parser.add_argument("--max_events", type=int, default=5)
    parser.add_argument("--output_file", type=str, default="full_event_embeddings.pt")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = MaskedPointModel(embed_dim=args.embed_dim).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    # Load dataset (raw, no fixed sampling)
    # We'll use a modified approach to get the full events
    dataset = CalorimeterDataset(num_hits=1, max_events=args.max_events)
    
    all_event_results = []
    
    print(f"Computing representations for {len(dataset.events)} full events...")
    for idx in tqdm(range(len(dataset.events))):
        # Extract all hits from the partition
        event = dataset.events[idx]
        x = np.concatenate(event["x"].to_numpy()) / dataset.coord_scale
        y = np.concatenate(event["y"].to_numpy()) / dataset.coord_scale
        z = np.concatenate(event["z"].to_numpy()) / dataset.coord_scale
        e = np.concatenate(event["total_energy"].to_numpy()) / dataset.energy_scale
        hits = torch.from_numpy(np.stack([x, y, z, e], axis=1).astype(np.float32)).to(device)
        
        embeddings = compute_all_hit_representations(model, hits, window_size=args.num_hits)
        all_event_results.append({
            "event_id": event["event_id"][0],
            "hits": hits.cpu().numpy(),
            "embeddings": embeddings.cpu().numpy()
        })
        
    torch.save(all_event_results, args.output_file)
    print(f"Saved full event embeddings to {args.output_file}")

if __name__ == "__main__":
    main()
