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
    parser.add_argument("--num_hits", type=int, default=256, help="Window size for sliding window inference")
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension used during training")
    parser.add_argument("--max_events", type=int, default=5)
    parser.add_argument("--output_file", type=str, default="full_event_embeddings.pt")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = MaskedPointModel(embed_dim=args.embed_dim).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    # Load dataset (using base class to access get_full_event)
    dataset = CalorimeterDataset(num_hits=args.num_hits, max_events=args.max_events)
    num_events = len(dataset)
    
    all_event_results = []
    
    print(f"Computing representations for {num_events} full events...")
    for idx in tqdm(range(num_events)):
        # Extract all hits for the full event
        event_data = dataset.get_full_event(idx)
        
        # Currently only calo hits are processed as tracker hits are empty in dataset.py
        calo_hits = torch.from_numpy(event_data["calo_hits"]).to(device)
        tracker_hits = torch.from_numpy(event_data["tracker_hits"]).to(device)
        
        all_hits = torch.cat([calo_hits, tracker_hits], dim=0)
        if all_hits.shape[0] == 0:
            continue
            
        # Compute embeddings for all hits in the event using the sliding window
        embeddings = compute_all_hit_representations(model, all_hits, window_size=args.num_hits)
        
        all_event_results.append({
            "event_id": event_data["event_id"],
            "hits": all_hits.cpu().numpy(),
            "embeddings": embeddings.cpu().numpy()
        })
        
    torch.save(all_event_results, args.output_file)
    print(f"Saved full event embeddings to {args.output_file}")

if __name__ == "__main__":
    main()
