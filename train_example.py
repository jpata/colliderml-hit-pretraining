"""
Point Cloud Pretraining for Calorimeter Hits.
Uses a Masked Point Modeling (MPM) approach with an efficient Transformer encoder.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CalorimeterDataset, NeighborhoodCalorimeterDataset
import numpy as np
import awkward as ak
from tqdm import tqdm
import umap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.stats import entropy
import seaborn as sns
import os

def compute_all_hit_representations(model, hits, window_size=256, overlap=128):
    """
    Computes embeddings for all hits in an event using overlapping windows.
    hits: (N, 4) tensor
    """
    N = hits.shape[0]
    embed_dim = model.mask_token.shape[0]
    device = next(model.parameters()).device
    
    # Sort hits by Z coordinate to give some spatial order to the sliding window
    z_indices = torch.argsort(hits[:, 2])
    sorted_hits = hits[z_indices]
    
    # Storage for accumulated embeddings and counts
    accumulated_embeddings = torch.zeros((N, embed_dim), device=device)
    counts = torch.zeros((N, 1), device=device)
    
    step = max(1, window_size - overlap)
    
    model.eval()
    with torch.no_grad():
        for start in range(0, N, step):
            end = min(start + window_size, N)
            window_hits = sorted_hits[start:end]
            
            # Pad if window is smaller than window_size (for model compatibility if needed)
            curr_size = window_hits.shape[0]
            if curr_size < window_size:
                padding = torch.zeros((window_size - curr_size, 4), device=device)
                input_hits = torch.cat([window_hits, padding], dim=0).unsqueeze(0)
            else:
                input_hits = window_hits.unsqueeze(0)
                
            # Bypassing forward() masking logic
            hit_embeddings = model.hit_encoder(input_hits)
            latent = model.transformer(hit_embeddings)
            
            valid_latent = latent[0, :curr_size]
            accumulated_embeddings[start:end] += valid_latent
            counts[start:end] += 1
            if end == N: break
                
    final_sorted_embeddings = accumulated_embeddings / counts
    final_embeddings = torch.zeros_like(final_sorted_embeddings)
    final_embeddings[z_indices] = final_sorted_embeddings
    return final_embeddings

def compute_representation_metrics(all_embeddings, all_hits, epoch, output_dir):
    """
    Compute PCA, Clustering, and Correlation metrics to evaluate representation expressiveness.
    """
    metrics = {}
    
    # 1. PCA Explained Variance Entropy
    pca = PCA(n_components=min(all_embeddings.shape[0], all_embeddings.shape[1], 16))
    pca.fit(all_embeddings)
    var_ratio = pca.explained_variance_ratio_
    # Normalize variance ratio to sum to 1 and compute entropy as a measure of "effective dimensionality"
    metrics["pca_entropy"] = entropy(var_ratio)
    metrics["pca_top_1"] = var_ratio[0]
    metrics["pca_top_3"] = np.sum(var_ratio[:3])
    
    # 2. Embedding-Input Correlation
    # Compute correlation between each embedding dimension and (x, y, z, e)
    # Using only a subset of dimensions for summary metric
    corr_matrix = np.zeros((all_embeddings.shape[1], 4))
    for i in range(all_embeddings.shape[1]):
        for j in range(4):
            corr_matrix[i, j] = np.abs(np.corrcoef(all_embeddings[:, i], all_hits[:, j])[0, 1])
    
    # Mean max correlation: how much does each embedding dimension "copy" an input feature
    metrics["mean_max_input_corr"] = np.mean(np.max(corr_matrix, axis=1))
    
    # 3. Clustering Metrics (on a sample to keep it fast)
    sample_size = min(len(all_embeddings), 2000)
    indices = np.random.choice(len(all_embeddings), sample_size, replace=False)
    emb_sample = all_embeddings[indices]
    hits_sample = all_hits[indices]
    
    clustering = DBSCAN(eps=0.5, min_samples=3).fit(emb_sample)
    labels = clustering.labels_
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    metrics["n_clusters"] = n_clusters
    
    if n_clusters > 1:
        # Silhouette score of the embeddings
        valid_mask = labels != -1
        if np.sum(valid_mask) > n_clusters:
            metrics["silhouette"] = silhouette_score(emb_sample[valid_mask], labels[valid_mask])
            
            # Cluster Cohesion: Mean physical variance of hits within the same embedding cluster
            cluster_vars = []
            for l in unique_labels:
                if l == -1: continue
                c_hits = hits_sample[labels == l]
                if len(c_hits) > 1:
                    # Variance in x, y, z (normalized)
                    cluster_vars.append(np.mean(np.var(c_hits[:, :3], axis=0)))
            metrics["cluster_physical_cohesion"] = np.mean(cluster_vars) if cluster_vars else 0
        else:
            metrics["silhouette"] = 0
            metrics["cluster_physical_cohesion"] = 0
    else:
        metrics["silhouette"] = 0
        metrics["cluster_physical_cohesion"] = 0
        
    # Plot Correlation Heatmap
    plt.figure(figsize=(8, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='viridis', xticklabels=['X', 'Y', 'Z', 'E'])
    plt.title(f"Embedding-Feature Absolute Correlation - Epoch {epoch+1}")
    plt.ylabel("Embedding Dimension")
    plt.savefig(os.path.join(output_dir, f"feature_correlation_epoch_{epoch+1}.png"))
    plt.close()
    
    return metrics

def plot_metrics_history(history, output_dir):
    """
    Plot the evolution of representation metrics over epochs.
    """
    epochs = [h["epoch"] for h in history]
    
    fig, axs = plt.subplots(3, 2, figsize=(15, 18))
    
    # 1. Train and Validation Loss
    axs[0, 0].plot(epochs, [h["train_loss"] for h in history], marker='o', label='Train Loss', color='blue')
    axs[0, 0].plot(epochs, [h["val_loss"] for h in history], marker='o', label='Val Loss', color='orange')
    axs[0, 0].set_title("Training & Validation Loss")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_yscale('log')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # 2. PCA Entropy (Effective Dimension)
    axs[0, 1].plot(epochs, [h["pca_entropy"] for h in history if "pca_entropy" in h], marker='o', color='green')
    axs[0, 1].set_title("PCA Explained Variance Entropy (Effective Dim)")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].grid(True)
    
    # 3. Silhouette Score
    axs[1, 0].plot(epochs, [h["silhouette"] for h in history if "silhouette" in h], marker='o', color='red')
    axs[1, 0].set_title("Embedding Silhouette Score (DBSCAN Clusters)")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].grid(True)
    
    # 4. Density-Loss Log-Log Correlation
    corrs = [h["density_corr"] for h in history if "density_corr" in h]
    if corrs:
        axs[1, 1].plot(epochs[:len(corrs)], corrs, marker='o', color='purple')
    axs[1, 1].set_title("Density vs MAE Log-Log Correlation")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].grid(True)
    
    # 5. Cluster Physical Cohesion
    axs[2, 0].plot(epochs, [h["cluster_physical_cohesion"] for h in history if "cluster_physical_cohesion" in h], marker='o', color='brown')
    axs[2, 0].set_title("Mean Cluster Physical Spread (Lower is Better)")
    axs[2, 0].set_xlabel("Epoch")
    axs[2, 0].grid(True)
    
    # 6. Number of Clusters
    axs[2, 1].plot(epochs, [h["n_clusters"] for h in history if "n_clusters" in h], marker='o', color='gray')
    axs[2, 1].set_title("Number of DBSCAN Clusters Found")
    axs[2, 1].set_xlabel("Epoch")
    axs[2, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "representation_metrics_evolution.png"))
    plt.close()

def visualize_embeddings_3d(model, full_dataset, epoch, output_dir, n_events=2):
    """
    Visualize x, y, z point cloud of calo and tracker hits, 
    colored by their clustering in the embedding space.
    """
    model.eval()
    device = next(model.parameters()).device
    print(f"Generating 3D point cloud visualizations for {n_events} events...")

    for i in range(min(n_events, len(full_dataset))):
        event_data = full_dataset.get_full_event(i)
        calo_hits = torch.from_numpy(event_data["calo_hits"]).to(device)
        tracker_hits = torch.from_numpy(event_data["tracker_hits"]).to(device)
        
        all_hits = torch.cat([calo_hits, tracker_hits], dim=0)
        if all_hits.shape[0] == 0: continue
        
        # Compute embeddings for all hits in the event
        embeddings = compute_all_hit_representations(model, all_hits, window_size=256)
        embeddings_np = embeddings.cpu().numpy()
        
        # DBSCAN clustering: eps (radius) and min_samples
        # These may need tuning based on the embedding space scale
        clustering = DBSCAN(eps=0.5, min_samples=3).fit(embeddings_np)
        clusters = clustering.labels_
        
        # 2D UMAP Visualization as well
        if embeddings_np.shape[0] > 10:
            try:
                reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
                embedding_2d = reducer.fit_transform(embeddings_np)
                
                plt.figure(figsize=(10, 8))
                plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=clusters, cmap='tab20', s=10, alpha=0.6)
                plt.colorbar(label='DBSCAN Cluster ID')
                plt.title(f"UMAP Projection of Hit Embeddings - Event {i}, Epoch {epoch+1}")
                plt.savefig(os.path.join(output_dir, f"umap_epoch_{epoch+1}_ev{i}.png"))
                plt.close()
            except Exception as e:
                print(f"UMAP failed for event {i}: {e}")

        # 3D Plotting
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color normalization for clusters (using a discrete map)
        # Handle noise (-1) by making it a separate color (e.g., black)
        cmap = plt.get_cmap('tab20')
        unique_labels = np.unique(clusters)
        n_clusters_found = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        n_calo = calo_hits.shape[0]
        
        def get_colors(lbls):
            # Map -1 to black, others to discrete colors
            # Must return consistent format (all RGBA tuples) to avoid matplotlib errors
            cols = []
            for l in lbls:
                if l == -1:
                    cols.append((0, 0, 0, 1)) # Black as RGBA
                else:
                    cols.append(cmap(l % 20))
            return cols

        # Calo hits (circles)
        if n_calo > 0:
            ax.scatter(
                event_data["calo_hits"][:, 0], 
                event_data["calo_hits"][:, 1], 
                event_data["calo_hits"][:, 2],
                c=get_colors(clusters[:n_calo]), marker='o', s=15, alpha=0.7, label='Calo Hits'
            )
            
        # Tracker hits (triangles)
        if tracker_hits.shape[0] > 0:
            ax.scatter(
                event_data["tracker_hits"][:, 0], 
                event_data["tracker_hits"][:, 1], 
                event_data["tracker_hits"][:, 2],
                c=get_colors(clusters[n_calo:]), marker='^', s=8, alpha=0.5, label='Tracker Hits'
            )
            
        ax.set_xlabel('X (normalized)')
        ax.set_ylabel('Y (normalized)')
        ax.set_zlabel('Z (normalized)')
        ax.set_title(f'Event {event_data["event_id"]} - Epoch {epoch+1}\n(DBSCAN: {n_clusters_found} clusters found, black = noise)')
        ax.legend()
        
        # Save plot
        plot_path = os.path.join(output_dir, f"point_cloud_epoch_{epoch+1}_ev{i}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()

class PointNetEncoder(nn.Module):
    """Simple MLP-based encoder per hit."""
    def __init__(self, input_dim=4, embed_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.mlp(x)

class MaskedPointModel(nn.Module):
    def __init__(self, embed_dim=128, nhead=8, num_layers=8, max_len=1024):
        super().__init__()
        self.hit_encoder = PointNetEncoder(input_dim=4, embed_dim=embed_dim)
        
        # Learned mask token
        self.mask_token = nn.Parameter(torch.randn(embed_dim))
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
        
        # Efficient Transformer encoder using scaled_dot_product_attention
        # We use the standard TransformerEncoderLayer but ensure it uses the fast path
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, batch_first=True, activation='relu',
            norm_first=True # Better for deep transformers
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Reconstruction head (predicts x, y, z, e)
        self.reconstructor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 4)
        )

    def forward(self, hits, mask_ratio=0.5):
        # hits: (B, N, 4)
        B, N, C = hits.shape
        device = hits.device
        
        # Embed each hit
        hit_embeddings = self.hit_encoder(hits) # (B, N, embed_dim)
        
        # Random masking
        num_masked = int(N * mask_ratio)
        mask_indices = torch.rand(B, N, device=device).argsort(dim=1)[:, :num_masked]
        
        # Prepare tokens for transformer
        input_embeddings = hit_embeddings.clone()
        for b in range(B):
            input_embeddings[b, mask_indices[b]] = self.mask_token
            
        # Add positional embedding
        input_embeddings = input_embeddings + self.pos_embed[:, :N, :]
            
        # Transformer processes all tokens (automatically uses FlashAttention if available)
        latent = self.transformer(input_embeddings) # (B, N, embed_dim)
        
        # Reconstruct only the masked ones
        reconstructed = self.reconstructor(latent) # (B, N, 4)
        
        return reconstructed, mask_indices, latent

def compute_density(hits, radii=[0.01, 0.02, 0.05]):
    """
    Computes multi-scale local density for hits by averaging neighbor counts at different radii.
    hits: (N, 4) or (B, N, 4)
    radii: list of radii to compute density at
    """
    if len(hits.shape) == 2:
        hits = hits.unsqueeze(0)
    
    B, N, C = hits.shape
    coords = hits[:, :, :3] # (B, N, 3)
    
    # Pairwise distances
    dist_sq = torch.cdist(coords, coords)**2
    
    combined_density = torch.zeros((B, N), device=hits.device)
    for r in radii:
        combined_density += (dist_sq < r**2).float().sum(dim=2)
        
    return combined_density / len(radii)

def train(num_hits=256, embed_dim=16, max_events=None, epochs=1, batch_size=4, 
          output_dir="results", output_loss=None, use_neighborhood=True, 
          mask_ratio=0.5, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Config: num_hits={num_hits}, embed_dim={embed_dim}, max_events={max_events}, neighborhood={use_neighborhood}, mask_ratio={mask_ratio}, lr={lr}")
    
    os.makedirs(output_dir, exist_ok=True)

    # Dataset Selection
    if use_neighborhood:
        train_dataset = NeighborhoodCalorimeterDataset(num_hits=num_hits, max_events=max_events, verbose=False)
        val_dataset = NeighborhoodCalorimeterDataset(num_hits=num_hits, max_events=100, verbose=False) # Small fixed val
    else:
        train_dataset = CalorimeterDataset(num_hits=num_hits, max_events=max_events, verbose=False)
        val_dataset = CalorimeterDataset(num_hits=num_hits, max_events=100, verbose=False) # Small fixed val
    
    # For IterableDataset, shuffle MUST be handled inside the dataset or by not using it in DataLoader.
    # Worker partitioning is handled in the dataset's __iter__.
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, prefetch_factor=2)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
    
    model = MaskedPointModel(embed_dim=embed_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.SmoothL1Loss(reduction='none') # Need per-hit loss for density analysis

    # Training Loop
    epoch_losses = []
    metrics_history = []
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for i, batch_hits in enumerate(pbar):
            batch_hits = batch_hits.to(device)
            reconstructed, mask_indices, latent = model(batch_hits, mask_ratio=mask_ratio)
            
            # Extract masked targets and preds
            masked_targets = torch.cat([batch_hits[b, mask_indices[b]] for b in range(batch_hits.shape[0])])
            masked_preds = torch.cat([reconstructed[b, mask_indices[b]] for b in range(batch_hits.shape[0])])
            
            # Energy-weighted loss: weight each hit's loss by its energy (4th column, index 3)
            # The energy is already log-scaled in the dataset.
            weights = masked_targets[:, 3]
            
            # Compute per-hit loss (mean over x, y, z, e components) and apply weights
            loss_raw = nn.SmoothL1Loss(reduction='none')(masked_preds, masked_targets).mean(dim=1)
            loss = (loss_raw * weights).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        scheduler.step()
        num_batches = i + 1
        avg_train_loss = total_train_loss / num_batches
        print(f"Epoch {epoch+1} Average Train Loss: {avg_train_loss:.6f}")

        # Validation phase with Density Analysis and Representation Quality
        model.eval()
        total_val_loss = 0
        density_stats = [] # Store (density, loss) tuples
        
        val_embeddings_sample = []
        val_hits_sample = []
        
        with torch.no_grad():
            total_coord_loss = 0
            total_energy_loss = 0
            v_batches = 0
            for batch_hits in val_dataloader:
                v_batches += 1
                batch_hits = batch_hits.to(device)
                reconstructed, mask_indices, latent = model(batch_hits, mask_ratio=mask_ratio)
                
                # Collect a subset for representation analysis
                if v_batches <= 10: # Sample from first 10 batches
                    val_embeddings_sample.append(latent.cpu().numpy().reshape(-1, embed_dim))
                    val_hits_sample.append(batch_hits.cpu().numpy().reshape(-1, 4))

                # Extract masked targets and preds
                for b in range(batch_hits.shape[0]):
                    masked_targets = batch_hits[b, mask_indices[b]]
                    masked_preds = reconstructed[b, mask_indices[b]]
                    
                    # Separate coordinate (x,y,z) and energy (e) losses
                    coord_loss = nn.SmoothL1Loss()(masked_preds[:, :3], masked_targets[:, :3])
                    energy_loss = nn.SmoothL1Loss()(masked_preds[:, 3:], masked_targets[:, 3:])
                    
                    total_coord_loss += coord_loss.item()
                    total_energy_loss += energy_loss.item()
                
                loss = nn.SmoothL1Loss()(reconstructed, batch_hits) # Overkill but simple
                
                densities = compute_density(batch_hits) # (B, N)
                
                for b in range(batch_hits.shape[0]):
                    masked_targets = batch_hits[b, mask_indices[b]]
                    masked_preds = reconstructed[b, mask_indices[b]]
                    masked_densities = densities[b, mask_indices[b]]
                    
                    hit_losses = torch.mean(torch.abs(masked_targets - masked_preds), dim=1)
                    
                    for d, l in zip(masked_densities.cpu().numpy(), hit_losses.cpu().numpy()):
                        density_stats.append((d, l))
                
                # Global val loss
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / v_batches
        avg_coord_loss = total_coord_loss / (v_batches * batch_size)
        avg_energy_loss = total_energy_loss / (v_batches * batch_size)
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.6f} (Coord: {avg_coord_loss:.6f}, Energy: {avg_energy_loss:.6f})")
        
        # Consolidate metrics for history
        epoch_stats = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "coord_loss": avg_coord_loss,
            "energy_loss": avg_energy_loss
        }

        # Compute and log representation metrics
        if val_embeddings_sample:
            all_emb = np.concatenate(val_embeddings_sample, axis=0)
            all_hits = np.concatenate(val_hits_sample, axis=0)
            rep_metrics = compute_representation_metrics(all_emb, all_hits, epoch, output_dir)
            epoch_stats.update(rep_metrics)
            
            print(f"Epoch {epoch+1} Representation: PCA Entropy: {rep_metrics['pca_entropy']:.3f}, "
                  f"Silhouette: {rep_metrics['silhouette']:.3f}, Clusters: {rep_metrics['n_clusters']}, "
                  f"Physical Cohesion: {rep_metrics['cluster_physical_cohesion']:.4f}")

        # Plot Reconstruction Fidelity vs Density
        if density_stats:
            density_stats = np.array(density_stats)
            
            # Calculate correlation between log(density) and log(loss)
            # Ensure values are positive for log calculation
            valid_mask = (density_stats[:, 0] > 0) & (density_stats[:, 1] > 0)
            if np.any(valid_mask):
                log_density = np.log10(density_stats[valid_mask, 0])
                log_loss = np.log10(density_stats[valid_mask, 1])
                corr = np.corrcoef(log_density, log_loss)[0, 1]
                epoch_stats["density_corr"] = corr
                print(f"Epoch {epoch+1} Log-Log Correlation (Density vs MAE): {corr:.4f}")

            plt.figure(figsize=(10, 6))
            plt.hexbin(density_stats[:, 0], density_stats[:, 1], gridsize=100, cmap='YlOrRd', bins='log', xscale='log', yscale='log')
            plt.colorbar(label='Log10(Count)')
            plt.xlabel('Multi-Scale Local Hit Density (Avg neighbors at r=0.01, 0.02, 0.05)')
            plt.ylabel('Reconstruction MAE')
            plt.title(f'Fidelity vs Density - Epoch {epoch+1}')
            plt.savefig(os.path.join(output_dir, f"fidelity_vs_density_epoch_{epoch+1}.png"))
            plt.close()

        metrics_history.append(epoch_stats)
        plot_metrics_history(metrics_history, output_dir)
        epoch_losses.append((epoch + 1, avg_train_loss, avg_val_loss, avg_coord_loss, avg_energy_loss))

        # Generate 3D point cloud visualization with embedding-based coloring
        visualize_embeddings_3d(model, train_dataset, epoch, output_dir, n_events=1)

    # Save model
    save_path = os.path.join(output_dir, f"checkpoint_h{num_hits}_e{embed_dim}_neigh{use_neighborhood}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    if output_loss:
        loss_file_path = os.path.join(output_dir, output_loss)
        with open(loss_file_path, "w") as f:
            f.write("epoch,train_loss,val_loss,coord_loss,energy_loss\n")
            for e, t, v, c, en in epoch_losses:
                f.write(f"{e},{t:.6f},{v:.6f},{c:.6f},{en:.6f}\n")
        print(f"Losses saved to {loss_file_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_hits", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--max_events", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--output_loss", type=str, default=None)
    parser.add_argument("--neighborhood", type=str, choices=["True", "False"], default="True")
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    use_neighborhood = args.neighborhood == "True"
    
    train(
        num_hits=args.num_hits,
        embed_dim=args.embed_dim,
        max_events=args.max_events,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        output_loss=args.output_loss,
        use_neighborhood=use_neighborhood,
        mask_ratio=args.mask_ratio,
        lr=args.lr
    )
