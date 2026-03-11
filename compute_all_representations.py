"""
Script to compute representations for ALL hits in an event using a trained model.
Uses a sliding window approach to handle large contexts.
"""

import torch
import torch.nn as nn
from dataset import CalorimeterDataset
from train_example import MaskedPointModel, compute_all_hit_representations
from model_config import get_model_config
import numpy as np
from tqdm import tqdm
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_hits", type=int, default=None, help="Window size for sliding window inference")
    parser.add_argument("--embed_dim", type=int, default=None, help="Embedding dimension used during training")
    parser.add_argument("--n_patches", type=int, default=None)
    parser.add_argument("--k_neighbors", type=int, default=None)
    parser.add_argument("--max_events", type=int, default=5)
    parser.add_argument("--output_file", type=str, default="full_event_embeddings.pt")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load central config
    config = get_model_config()
    if args.num_hits is not None: config["num_hits"] = args.num_hits
    if args.embed_dim is not None: config["embed_dim"] = args.embed_dim
    if args.n_patches is not None: config["n_patches"] = args.n_patches
    if args.k_neighbors is not None: config["k_neighbors"] = args.k_neighbors
    
    num_hits = config.pop("num_hits")
    
    # Load model
    model = MaskedPointModel(**config).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    # Load dataset (using base class to access get_full_event)
    dataset = CalorimeterDataset(num_hits=num_hits, max_events=args.max_events)
    num_events = len(dataset)
    
    all_event_results = []
    
    print(f"Computing representations for {num_events} full events...")
    for idx in tqdm(range(num_events)):
        # Extract all hits for the full event
        event_data = dataset.get_full_event(idx)
        all_hits = torch.from_numpy(event_data["all_hits"]).to(device)
        
        if all_hits.shape[0] == 0:
            continue
            
        # Compute embeddings for all hits in the event using the sliding window
        embeddings, coords, windows = compute_all_hit_representations(
            model, all_hits, window_size=num_hits, return_windows=True
        )
        
        all_event_results.append({
            "event_id": event_data["event_id"],
            "hits": all_hits.cpu().numpy(),
            "embeddings": embeddings.numpy(),
            "coords": coords.numpy(),
            "window_hits": [w.numpy() for w in windows]
        })
        
    torch.save(all_event_results, args.output_file)
    print(f"Saved full event embeddings to {args.output_file}")

if __name__ == "__main__":
    main()
