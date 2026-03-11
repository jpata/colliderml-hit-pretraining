import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from dataset import CalorimeterDataset
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree
import umap
import os
from tqdm import tqdm

# Import central definitions to ensure consistency
from train_example import MaskedPointModel, compute_all_hit_representations

def visualize_event(model, event_data, event_type, output_dir, window_size=1024, overlap=None):
    model.eval()
    device = next(model.parameters()).device
    
    event_id = event_data["event_id"]
    all_hits_np = event_data["all_hits"]
    all_hits = torch.from_numpy(all_hits_np).to(device)
    
    if all_hits.shape[0] < 10:
        return

    # Ensure non-zero overlap for smooth window transitions
    if overlap is None:
        overlap = window_size // 4
        
    # Compute embeddings for all patches in the full event
    # window_size matches training num_hits and provides 3D-local context
    embeddings, centers = compute_all_hit_representations(model, all_hits, window_size=window_size, overlap=overlap)
    embeddings = embeddings.numpy()
    centers = centers.numpy()
    
    print(f"    - Event {event_id}: {all_hits_np.shape[0]} hits, {len(embeddings)} patches")
    
    # 1. UMAP projection of PATCH embeddings
    # We define the latent space structure using only the patches
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    umap_patch_coords = reducer.fit_transform(embeddings)
    
    # 2. DBSCAN clustering on patches
    clustering = DBSCAN(eps=0.5, min_samples=3).fit(umap_patch_coords)
    patch_labels = clustering.labels_
    
    # 3. Compute hit embeddings via IDW of N nearest patches
    N_neighbors = 5
    tree = KDTree(centers)
    distances, indices = tree.query(all_hits_np[:, :3], k=N_neighbors)
    
    # Avoid division by zero and compute weights
    distances = np.maximum(distances, 1e-8)
    weights = 1.0 / distances
    weights /= weights.sum(axis=1, keepdims=True)
    
    # Compute weighted average of patch embeddings for each hit
    neighbor_embeddings = embeddings[indices] # (N_hits, N_neighbors, D)
    hit_embeddings = np.sum(neighbor_embeddings * weights[:, :, np.newaxis], axis=1)
    
    # Project hits into the same UMAP space as patches
    print(f"    - Projecting {len(hit_embeddings)} hits to UMAP space...")
    umap_hit_coords = reducer.transform(hit_embeddings)
    
    # Use nearest patch for cluster assignment (coloring)
    hit_labels = patch_labels[indices[:, 0]]
    
    # --- Plotting ---
    fig = plt.figure(figsize=(20, 10))
    
    # Filter for non-padding/valid hits
    mask = np.any(all_hits_np[:, :3] != 0, axis=1)
    valid_hits_xyz = all_hits_np[mask, :3]
    valid_hit_labels = hit_labels[mask]
    valid_umap_hit_coords = umap_hit_coords[mask]
    
    # Left: 3D XYZ Plot (Hits + Patch Centers)
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Plot raw hits (small, higher alpha)
    hit_colors = [plt.cm.tab20(i % 20) if i != -1 else (0.8, 0.8, 0.8, 0.2) for i in valid_hit_labels]
    ax1.scatter(valid_hits_xyz[:, 0], valid_hits_xyz[:, 1], valid_hits_xyz[:, 2], 
                c=hit_colors, s=1, alpha=0.8)
    
    # Plot patch centers (larger, somewhat transparent)
    center_colors = [plt.cm.tab20(i % 20) if i != -1 else (0, 0, 0, 1) for i in patch_labels]
    ax1.scatter(centers[:, 0], centers[:, 1], centers[:, 2], 
                c=center_colors, s=30, edgecolors='black', linewidth=0.5, label='Patch Centers', alpha=0.6)
    
    ax1.set_title(f"3D Space: Hits & Patches (N_hits={np.sum(mask)}, N_patches={len(centers)})\n{event_type} event {event_id}")
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    
    # Right: 2D UMAP Projection (Patches + Hits)
    ax2 = fig.add_subplot(122)
    
    # Plot hits in UMAP space (background)
    ax2.scatter(valid_umap_hit_coords[:, 0], valid_umap_hit_coords[:, 1], 
                c=hit_colors, s=1, alpha=0.5, label='Hits')
    
    # Plot patches in UMAP space (foreground)
    ax2.scatter(umap_patch_coords[:, 0], umap_patch_coords[:, 1], 
                c=center_colors, s=40, edgecolors='black', linewidth=0.5, label='Patches', alpha=0.6)
    
    ax2.set_title(f"Latent Space: UMAP Projection\nColored by DBSCAN Clusters (IDW Hit Projection)")
    ax2.set_xlabel('UMAP 1'); ax2.set_ylabel('UMAP 2')
    ax2.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"embeddings_full_{event_type}_{event_id}.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"    - Saved plot to {plot_path}")

from model_config import get_model_config

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="results/test/checkpoint_h2048_patches.pth")
    parser.add_argument("--num_hits", type=int, default=None)
    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--n_patches", type=int, default=None)
    parser.add_argument("--k_neighbors", type=int, default=None)
    parser.add_argument("--num_events", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="validation_plots/embeddings_viz_full")
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
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    model = MaskedPointModel(**config).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    datasets = ["ttbar", "ggf"]
    
    for ds_name in datasets:
        print(f"Processing dataset: {ds_name}")
        ds = CalorimeterDataset(dataset_name=ds_name, num_hits=num_hits, max_events=args.num_events)
        
        for i in range(args.num_events):
            event_data = ds.get_full_event(i)
            visualize_event(model, event_data, ds_name, args.output_dir, window_size=num_hits)

if __name__ == "__main__":
    main()
