import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
import umap
import argparse
from tqdm import tqdm
import seaborn as sns

# Add project root and Panda to path
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "Panda"))

from scripts.train_panda import Sonata, Compose
from src.dataset import CalorimeterDataset

def visualize_embedding_features(umap_proj, energy, hit_type, event_type, event_id, output_dir):
    """Plot UMAP colored by physical features (Energy, Hit Type)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 1. Colored by Energy
    sc1 = ax1.scatter(umap_proj[:, 0], umap_proj[:, 1], c=energy, cmap='viridis', s=5, alpha=0.6)
    plt.colorbar(sc1, ax=ax1, label='Log10(Energy)')
    ax1.set_title(f"UMAP: Colored by Energy\n{event_type} event {event_id}")
    ax1.set_xlabel('UMAP 1'); ax1.set_ylabel('UMAP 2')
    
    # 2. Colored by Hit Type
    # 0: Calo, 1: Tracker
    unique_types = np.unique(hit_type)
    colors = ['blue', 'green'] # 0: blue (calo), 1: green (tracker)
    for t in unique_types:
        mask = hit_type == t
        label = "Calo" if t == 0 else "Tracker"
        ax2.scatter(umap_proj[mask, 0], umap_proj[mask, 1], c=colors[int(t)], label=label, s=5, alpha=0.6)
    
    ax2.set_title(f"UMAP: Colored by Hit Type\n{event_type} event {event_id}")
    ax2.set_xlabel('UMAP 1'); ax2.set_ylabel('UMAP 2')
    ax2.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"panda_features_{event_type}_{event_id}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"    - Saved feature-correlated UMAP to {plot_path}")

def visualize_distance_correlation(coords, embeddings, event_type, event_id, output_dir, n_samples=10000):
    """Plot physical distance vs latent distance for point pairs."""
    n_points = len(coords)
    if n_points < 2: return
    
    # Sample pairs
    idx1 = np.random.randint(0, n_points, n_samples)
    idx2 = np.random.randint(0, n_points, n_samples)
    
    # Physical distance (Euclidean in 3D)
    d_phys = np.sqrt(np.sum((coords[idx1] - coords[idx2])**2, axis=1))
    
    # Latent distance (Cosine distance or Euclidean)
    # Cosine distance: 1 - cosine_similarity
    emb1 = embeddings[idx1]
    emb2 = embeddings[idx2]
    
    # L2 distance in latent space
    d_latent = np.sqrt(np.sum((emb1 - emb2)**2, axis=1))
    
    plt.figure(figsize=(10, 8))
    plt.hexbin(d_phys, d_latent, gridsize=50, cmap='Blues', bins='log')
    plt.colorbar(label='log10(count)')
    plt.title(f"Distance Correlation: Physical vs. Latent Space\n{event_type} event {event_id}")
    plt.xlabel("Physical Distance (Euclidean)")
    plt.ylabel("Latent Distance (Euclidean)")
    
    # Add correlation coefficient
    corr = np.corrcoef(d_phys, d_latent)[0, 1]
    plt.annotate(f"Pearson r = {corr:.3f}", xy=(0.05, 0.95), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round", fc="w", alpha=0.5))
    
    plot_path = os.path.join(output_dir, f"panda_dist_corr_{event_type}_{event_id}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"    - Saved distance correlation plot to {plot_path}")

def visualize_cluster_correlations(labels, coords, embeddings, energy, hit_type, event_type, event_id, output_dir):
    """Correlate physical cluster properties with latent properties across many clusters."""
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1] # Ignore noise
    
    if len(unique_labels) < 2: 
        print(f"    - Skipping cluster correlations: only {len(unique_labels)} clusters found.")
        return
    
    cluster_stats = []
    for l in unique_labels:
        mask = labels == l
        c_coords = coords[mask]
        c_embs = embeddings[mask]
        c_energy = energy[mask]
        c_types = hit_type[mask]
        
        # 1. Physical Properties
        total_energy = np.sum(10**c_energy - 1.0) # Back to linear scale for sum
        log_total_energy = np.log10(total_energy + 1e-9)
        num_hits = len(c_coords)
        spatial_extent = np.sqrt(np.sum(np.std(c_coords, axis=0)**2)) # RMS spread
        
        # 2. Latent Properties
        centroid = np.mean(c_embs, axis=0)
        # Latent Cohesion: Mean L2 distance to centroid
        cohesion = np.mean(np.sqrt(np.sum((c_embs - centroid)**2, axis=1)))
        
        # 3. Composition
        p_tracker = np.mean(c_types == 1)
        
        cluster_stats.append({
            "log_energy": log_total_energy,
            "num_hits": num_hits,
            "spatial_extent": spatial_extent,
            "latent_cohesion": cohesion,
            "p_tracker": p_tracker
        })
    
    import pandas as pd
    df = pd.DataFrame(cluster_stats)
    
    # Plotting Correlations
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Color by Tracker Proportion
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
    # A. Num Hits vs Total Energy
    sns.scatterplot(data=df, x="num_hits", y="log_energy", hue="p_tracker", 
                    palette=cmap, ax=axes[0, 0], s=60, alpha=0.8)
    axes[0, 0].set_title("Cluster Size (Hits) vs. Total Energy")
    axes[0, 0].set_xlabel("Number of Hits")
    axes[0, 0].set_ylabel("Log10(Total Energy)")
    
    # B. Spatial Extent vs Total Energy
    sns.scatterplot(data=df, x="spatial_extent", y="log_energy", hue="p_tracker", 
                    palette=cmap, ax=axes[0, 1], s=60, alpha=0.8)
    axes[0, 1].set_title("Spatial Extent (RMS) vs. Total Energy")
    axes[0, 1].set_xlabel("Spatial Extent (m)")
    axes[0, 1].set_ylabel("Log10(Total Energy)")
    
    # C. Spatial Extent vs Latent Cohesion
    sns.scatterplot(data=df, x="spatial_extent", y="latent_cohesion", hue="p_tracker", 
                    palette=cmap, ax=axes[1, 0], s=60, alpha=0.8)
    axes[1, 0].set_title("Spatial Extent vs. Latent Cohesion")
    axes[1, 0].set_xlabel("Spatial Extent (m)")
    axes[1, 0].set_ylabel("Mean Latent Dist to Centroid")
    
    # D. Latent Cohesion vs Total Energy
    sns.scatterplot(data=df, x="latent_cohesion", y="log_energy", hue="p_tracker", 
                    palette=cmap, ax=axes[1, 1], s=60, alpha=0.8)
    axes[1, 1].set_title("Latent Cohesion vs. Total Energy")
    axes[1, 1].set_xlabel("Mean Latent Dist to Centroid")
    axes[1, 1].set_ylabel("Log10(Total Energy)")
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"panda_cluster_corrs_{event_type}_{event_id}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"    - Saved cluster correlation analysis to {plot_path}")

def visualize_panda_event(model, event_data, event_type, output_dir, device):
    model.eval()
    
    event_id = event_data["event_id"]
    all_hits_np = event_data["all_hits"] # (N, 5) -> x, y, z, energy, type
    
    if all_hits_np.shape[0] < 10:
        print(f"Skipping event {event_id}: too few hits ({all_hits_np.shape[0]})")
        return

    # 1. Preprocessing
    transform = Compose([
        dict(type="NormalizeCoord", center=[0.0, 0.0, 0.0], scale=1.0),
        dict(type="GridSample", grid_size=0.001, hash_type="fnv", mode="train", return_grid_coord=True),
    ])
    
    # hits: (x, y, z, e, type)
    processed = transform({
        "coord": all_hits_np[:, :3], 
        "origin_coord": all_hits_np[:, :3],
        "energy": all_hits_np[:, 3:4], 
        "hit_type": all_hits_np[:, 4:5], 
        "index_valid_keys": ["coord", "origin_coord", "energy", "hit_type"]
    })
    
    # Features: (x, y, z, energy, type)
    feat = np.concatenate([processed["coord"], processed["energy"], processed["hit_type"]], axis=1)
    
    batch = {
        "coord": torch.from_numpy(processed["coord"]).float().to(device),
        "origin_coord": torch.from_numpy(processed["origin_coord"]).float().to(device),
        "grid_coord": torch.from_numpy(processed["grid_coord"]).long().to(device),
        "feat": torch.from_numpy(feat).float().to(device),
        "offset": torch.tensor([processed["coord"].shape[0]], dtype=torch.long).to(device),
        "grid_size": torch.tensor([0.001]).to(device)
    }
    
    # 2. Forward pass to get embeddings
    with torch.no_grad():
        # We use student_backbone for embeddings as in train_panda.py
        point = model.student_backbone(batch, upcast=False)
        # Upcast to propagate embeddings back to original points (or grid points)
        # PointTransformerV3.forward with upcast=True does this, but we can call Sonata's up_cast
        point = model.up_cast(point)
        embeddings = point.feat.cpu().numpy()
        coords = point.origin_coord.cpu().numpy()

    print(f"    - Event {event_id}: {all_hits_np.shape[0]} hits -> {len(embeddings)} grid points")

    # 3. Plot embedding distance distribution to help pick eps
    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=5)
    nbrs = neigh.fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    
    # Sort distances to the 5th nearest neighbor
    dist_5th = np.sort(distances[:, 4])
    
    # Heuristic: Pick eps based on the "knee" of the k-distance plot
    # We'll use the point of maximum distance from the line connecting first and last points
    coords_k = np.vstack((np.arange(len(dist_5th)), dist_5th)).T
    first_pt = coords_k[0]
    last_pt = coords_k[-1]
    line_vec = last_pt - first_pt
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    vec_from_first = coords_k - first_pt
    scalar_proj = np.dot(vec_from_first, line_vec_norm)
    vec_proj = np.outer(scalar_proj, line_vec_norm)
    dist_to_line = np.sqrt(np.sum((vec_from_first - vec_proj)**2, axis=1))
    knee_idx = np.argmax(dist_to_line)
    suggested_eps = dist_5th[knee_idx]
    
    print(f"    - Suggested eps from k-distance knee: {suggested_eps:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(dist_5th, label="5th-NN Distance")
    plt.axhline(y=suggested_eps, color='r', linestyle='--', label=f"Suggested eps ({suggested_eps:.4f})")
    plt.axvline(x=knee_idx, color='g', linestyle=':', label=f"Knee index ({knee_idx})")
    plt.title(f"K-distance Plot (k=5)\n{event_type} event {event_id}")
    plt.xlabel("Points sorted by distance")
    plt.ylabel("5th Nearest Neighbor Distance")
    plt.legend()
    plt.grid(True)
    dist_plot_path = os.path.join(output_dir, f"panda_dist_{event_type}_{event_id}.png")
    plt.savefig(dist_plot_path)
    plt.close()
    print(f"    - Saved distance distribution plot to {dist_plot_path}")

    # 4. DBSCAN clustering on embeddings
    print(f"    - Optimizing DBSCAN on {len(embeddings)} embeddings...")
    # Use suggested_eps and some variations
    param_grid = [
        (suggested_eps * 0.5, 5),
        (suggested_eps, 5),
        (suggested_eps * 1.5, 5),
        (suggested_eps * 2.0, 5),
    ]
    
    best_labels = None
    best_params = None
    max_clusters = -1
    
    for eps, min_samples in param_grid:
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"      - eps={eps}, min_samples={min_samples}: {n_clusters} clusters")
        
        if n_clusters > max_clusters:
            max_clusters = n_clusters
            best_labels = labels
            best_params = (eps, min_samples)
            
    labels = best_labels
    eps, min_samples = best_params
    print(f"    - Selected best DBSCAN: eps={eps}, min_samples={min_samples} with {max_clusters} clusters")
    
    # 5. UMAP projection for visualization
    print(f"    - Running UMAP on {len(embeddings)} embeddings...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    umap_proj = reducer.fit_transform(embeddings)
    
    # 6. Plotting
    fig = plt.figure(figsize=(20, 10))
    
    # Left: 3D XYZ Plot
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Color mapping for clusters
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    label_to_color = {l: colors[i] if l != -1 else (0.8, 0.8, 0.8, 0.2) for i, l in enumerate(unique_labels)}
    hit_colors = [label_to_color[l] for l in labels]
    
    ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                c=hit_colors, s=5, alpha=0.8)
    
    ax1.set_title(f"3D Space: Grid Points (N={len(coords)})\n{event_type} event {event_id}")
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    
    # Right: 2D UMAP Projection
    ax2 = fig.add_subplot(122)
    ax2.scatter(umap_proj[:, 0], umap_proj[:, 1], 
                c=hit_colors, s=5, alpha=0.8)
    
    ax2.set_title(f"Latent Space: UMAP Projection\nColored by DBSCAN Clusters (eps={eps:.4f}, n_clusters={max_clusters})")
    ax2.set_xlabel('UMAP 1'); ax2.set_ylabel('UMAP 2')
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"panda_event_{event_type}_{event_id}.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"    - Saved plot to {plot_path}")

    # 7. Additional Embedding Visualizations
    print(f"    - Generating additional embedding visualizations...")
    energy = processed["energy"].flatten()
    hit_type = processed["hit_type"].flatten()
    
    # Ensure lengths match (they should if up_cast worked correctly)
    if len(embeddings) == len(energy):
        visualize_embedding_features(umap_proj, energy, hit_type, event_type, event_id, output_dir)
        visualize_distance_correlation(coords, embeddings, event_type, event_id, output_dir)
        visualize_cluster_correlations(labels, coords, embeddings, energy, hit_type, event_type, event_id, output_dir)
    else:
        print(f"    - Warning: Embedding size ({len(embeddings)}) does not match feature size ({len(energy)}). Skipping additional plots.")

def main():
    parser = argparse.ArgumentParser(description="Visualize PANDA model outputs")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to PANDA checkpoint")
    parser.add_argument("--num_events", type=int, default=5, help="Number of events per dataset")
    parser.add_argument("--output_dir", type=str, default="validation_plots/panda_viz", help="Output directory")
    parser.add_argument("--dataset", type=str, nargs="+", default=["ttbar", "ggf"], help="Datasets to visualize")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # PANDA Model Configuration (from train_panda.py)
    backbone_config = dict(
        in_channels=5, # (x, y, z, energy, type)
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(256, 256, 256, 256, 256),
        shuffle_orders=True,
        enable_flash=False,
        enc_mode=True,
        mask_token=True
    )
    
    # Instantiate Sonata model
    # Note: head_in_channels = 32 + 64 + 128 + 256 + 512 = 992
    model = Sonata(backbone_config, head_in_channels=992, 
                    head_num_prototypes=4096,
                    num_global_view=2,
                    num_local_view=6).to(device)
    
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    for ds_name in args.dataset:
        print(f"Processing dataset: {ds_name}")
        try:
            ds = CalorimeterDataset(dataset_name=ds_name)
        except Exception as e:
            print(f"Error loading dataset {ds_name}: {e}")
            continue
            
        for i in range(args.num_events):
            print(f"  Visualizing event {i}...")
            event_data = ds.get_full_event(i)
            visualize_panda_event(model, event_data, ds_name, args.output_dir, device)

if __name__ == "__main__":
    main()
